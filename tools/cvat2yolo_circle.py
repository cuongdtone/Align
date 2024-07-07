import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import imgaug as ia
import cv2
from imgaug import augmenters as iaa
from functools import reduce
import cv2
import numpy as np
from pycocotools import mask as mask_utils



def cvat_rle_to_binary_image_mask(cvat_rle: dict, img_h: int, img_w: int) -> np.ndarray:
    # convert CVAT tight object RLE to COCO-style whole image mask
    rle = cvat_rle['rle']
    left = cvat_rle['left']
    top = cvat_rle['top']
    width = cvat_rle['width']

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    value = 0
    offset = 0
    for rle_count in rle:
        while rle_count > 0:
            y, x = divmod(offset, width)
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = 1 - value

    return mask

def binary_image_mask_to_cvat_rle(image: np.ndarray) -> dict:
    # convert COCO-style whole image mask to CVAT tight object RLE

    istrue = np.argwhere(image == 1).transpose()
    top = int(istrue[0].min())
    left = int(istrue[1].min())
    bottom = int(istrue[0].max())
    right = int(istrue[1].max())
    roi_mask = image[top:bottom + 1, left:right + 1]

    # compute RLE values
    def reduce_fn(acc, v):
        if v == acc['val']:
            acc['res'][-1] += 1
        else:
            acc['val'] = v
            acc['res'].append(1)
        return acc
    roi_rle = reduce(
        reduce_fn,
        roi_mask.flat,
        { 'res': [0], 'val': False }
    )['res']

    cvat_rle = {
        'rle': roi_rle,
        'top': top,
        'left': left,
        'width': right - left + 1,
        'height': bottom - top + 1,
    }

    return cvat_rle

def cvat_rle_to_coco_rle(cvat_rle: dict, img_h: int, img_w: int) -> dict:
    # covert CVAT tight object RLE to COCO whole image mask RLE
    binary_image_mask = cvat_rle_to_binary_image_mask(cvat_rle, img_h=img_h, img_w=img_w)
    return mask_utils.encode(np.asfortranarray(binary_image_mask))

def deserialize_cvat_rle(serialized_cvat_rle: dict) -> dict:
    return {
        'rle': list(map(int, serialized_cvat_rle['rle'].split(','))),
        'top': int(serialized_cvat_rle['top']),
        'left': int(serialized_cvat_rle['left']),
        'width': int(serialized_cvat_rle['width']),
        'height': int(serialized_cvat_rle['height']),
    }

def serialize_cvat_rle(cvat_rle: dict) -> dict:
    return {
        'rle': ', '.join(map(str, cvat_rle['rle'])),
        'top': str(cvat_rle['top']),
        'left': str(cvat_rle['left']),
        'width': str(cvat_rle['width']),
        'height': str(cvat_rle['height']),
    }


def cvat2poly(serialized_cvat_image: dict, serialized_cvat_rle: dict):
    img_w = int(serialized_cvat_image['width'])
    img_h = int(serialized_cvat_image['height'])

    # HWC BGR [0, 1] image for OpenCV, you can use cv2.imread() instead
    image = np.zeros((img_h, img_w, 3), np.float32)

    cvat_rle = deserialize_cvat_rle(serialized_cvat_rle)
    mask = cvat_rle_to_binary_image_mask(cvat_rle, img_h=img_h, img_w=img_w)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.0001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for p in approx:
            polygons.append(p[0].tolist())
    
    x_center = (cvat_rle['left'] + cvat_rle['width'])/2
    y_center = (cvat_rle['top'] + cvat_rle['height'])/2
    
    return polygons, x_center, y_center,  cvat_rle['width'], cvat_rle['height']

def draw(image, polygons, label):
    mask = np.zeros_like(image, dtype=np.uint8)

    polygons = np.array(polygons)
    points = polygons.reshape((-1, 1, 2))
    
    
    color = (0, 255, 0) if label == 0 else (0, 0, 255)
    
    cv2.polylines(image, [points.astype('int')], True, color, 2) # Draw Poly Lines
    
    # cv2.fillPoly(mask, [points], color=color)
    # image = cv2.addWeighted(image, 1, mask, 0.5, 0)
    return image


seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 0.3)),  # áp dụng Gaussian blur với sigma từ 0 đến 3.0
        iaa.AverageBlur(k=(2, 5)),         # áp dụng Average blur với kernel size từ 2 đến 7
        iaa.MedianBlur(k=(3, 5))          # áp dụng Median blur với kernel size từ 3 đến 11
    ])),
    iaa.Sometimes(0.7,
        [ iaa.Dropout((0.001, 0.06))]
    ),
    iaa.Sometimes(0.8, iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(5, 20)), 
        iaa.AdditiveGaussianNoise(scale=(5, 20), per_channel=True),
    ])),
    # iaa.Sometimes(0.5, iaa.OneOf([
    #     iaa.FastSnowyLandscape(), 
    #     iaa.Clouds(),
    #     iaa.Fog(),
    # ])),
])


def mask2poly(mask, complex = 0.0001):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = complex * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for p in approx:
            polygons.append(p[0].tolist())
    
    x, y, w, h = cv2.boundingRect(mask)
    x_center = x + w / 2
    y_center = y + h / 2
    obj_width = w
    obj_height = h

    
    return polygons, x_center, y_center, obj_width , obj_height


def normalize_coordinates(points, w_img, h_img):
    normalized_points = []
    for point in points:
        x_normalized = point[0] / w_img
        y_normalized = point[1] / h_img
        normalized_points += [x_normalized, y_normalized]
    return normalized_points

class CvatDataset(data.Dataset):
    def __init__(self, xml_path, img_root):
        # Parse the XML file
        tree = ET.parse(xml_path)
        # Get the root element
        root = tree.getroot()
        
        self.cls_names = {
            'line': 0,
            'edge': 1
        }


        self.labels = []
        # Access elements and attributes
        for child in root:
            # print(child.tag, child.attrib)
            img_path = child.get('name')
            if img_path is None:
                continue
            img_path = f'{img_root}/{img_path}'
            annotations = {'img': img_path,
                           'labels': [],
                           'width': child.get('width'), 
                           'height': child.get('height')
            }
            cx, cy, rx, ry = None, None, None, None
            for sub_child in child:
                if sub_child.tag == 'ellipse':
                    cx = sub_child.get('cx')
                    cy = sub_child.get('cy')
                    rx = sub_child.get('rx')
                    ry = sub_child.get('ry')
                if sub_child.tag == 'mask':
                    label = sub_child.get('label')
                    if label == 'bigchip':
                        rle = sub_child.get('rle')
                        left = sub_child.get('left')
                        top = sub_child.get('top')
                        width = sub_child.get('width')
                        height = sub_child.get('height')
                        
                        serialized_cvat_image = dict(
                            width = child.get('width'), 
                            height = child.get('height')
                        )
                        
                        serialized_cvat_rle = dict(
                            rle= rle,
                            left=left,
                            top=top,
                            width=width, 
                            height=height,
                            
                        )
                        polygon, x, y, w, h = cvat2poly(serialized_cvat_image, serialized_cvat_rle)
                        annotations['labels'].append([1, x, y, w, h, polygon])

            # over circle
            if cx is not None:
                mask = np.zeros((int(child.get('height')), int(child.get('width'))), dtype=np.uint8)
                center = (int(float(cx)), int(float(cy)))
                axes = (int(float(rx)), int(float(ry)))
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)
                polygon, x, y, w, h = mask2poly(mask)
                annotations['labels'].append([0, x, y, w, h, polygon])
    
            
            if len(annotations['labels']):
                self.labels.append(annotations)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]
    
    def run(self, out, prefix="A"):
        # out = f'{out}/{prefix}'
        # if os.path.exists(out):
        #     shutil.rmtree(out)

        # os.makedirs(out, exist_ok=True)

        for item in self.labels:
            
            img_path = item.get('img')
            w_img = int(item.get('width'))
            h_img = int(item.get('height'))

            yolo_data = ''
            image = cv2.imread(img_path)

            for l in item.get('labels'):
                cls_label, x, y, w, h, poly = l
                

                x = x / w_img
                y = y / h_img
                w = w / w_img
                h = h / h_img
                
                poly = np.asarray(poly, dtype=np.float64)
                poly /= [w_img, h_img] # scale
                # print(poly)
                
                yolo_poly = []
                for xx, yy in poly:
                    yolo_poly += [xx, yy]


                y_line = f'{cls_label} {" ".join(list(map(str, yolo_poly)))}\n'
                yolo_data += y_line
                
            # print(yolo_data)
            
            save_image_path =  prefix + os.path.basename(os.path.splitext(img_path)[0] + ".jpg")
            txt_path = prefix + os.path.basename(os.path.splitext(img_path)[0] + ".txt")
            
            cv2.imwrite(os.path.join(out, save_image_path), image)

            with open(os.path.join(out, txt_path), 'w') as f:
                f.write(yolo_data)


            for j in range(5):
                item = os.path.basename(os.path.splitext(img_path)[0])
                images_aug = seq(images=[image])
                cv2.imwrite(f'{out}/{prefix}{item}-{j}.jpg', images_aug[0][:, :, ::-1])
                shutil.copy2(os.path.join(out, txt_path), out + '/' + prefix + os.path.basename(os.path.splitext(img_path)[0] + f"-{j}.txt"))


import concurrent.futures


if __name__ == '__main__':
    executor =  concurrent.futures.ProcessPoolExecutor(max_workers=10)

    out = 'align-circle'
    if os.path.exists(out):
        shutil.rmtree(out)

    os.makedirs(out, exist_ok=True)


    data1 = CvatDataset(r'train-cvat-circle\set-failed-annotations.xml', r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\Failed\NGImage')
    print(len(data1))
    executor.submit(data1.run, out, "A")


    data2 = CvatDataset(r'train-cvat-circle\set1-100-annotations.xml', r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\SyntheticData2')
    print(len(data2))
    executor.submit(data2.run, out, "B")

    data3 = CvatDataset(r'train-cvat-circle\set1-100-jig2-annotations.xml', r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\OriginalImage_JIG2_240604')
    print(len(data3))
    executor.submit(data3.run, out, "C")


    data4 = CvatDataset(r'train-cvat-circle\set101-200-annotations.xml', r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\SyntheticData2')
    print(len(data4))
    executor.submit(data4.run, out, "D")

    data5 = CvatDataset(r'train-cvat-circle\D-first.xml', r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\SyntheticData')
    print(len(data5))
    executor.submit(data5.run, out, "E")

