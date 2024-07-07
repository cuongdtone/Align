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
import random

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



def rotate_image(image, angle):
    # Lấy kích thước ảnh
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def rotate_point(point, M):
    point_homogeneous = np.array([point[0], point[1], 1]).reshape((3, 1))
    rotated_point = np.dot(M, point_homogeneous)
    return (rotated_point[0, 0], rotated_point[1, 0])


def rotate_image_and_annotations(image_path, points, circle, output_dir, id_gen):
    # Read the image
    image_rot = cv2.imread(image_path)
    image_rot = seq(images=[image_rot[:, :, ::-1]])[0][:, :, ::-1]
    
    (h, w) = image_rot.shape[:2]
    
    # Rotate the image by 90 degrees clockwise
    rotated_image, M = rotate_image(image_rot, random.randint(-150, 150))

    yolo_data = ''
    for (cls, xx, yy) in points:
        # Calculate new coordinates after 90 degree clockwise rotation
        # new_xx = yy
        # new_yy = w - xx
        (xx, yy) = rotate_point((xx, yy), M)

        # Normalize coordinates and define box dimensions
        if cls == 4 or cls == 5:
            box_width = 55 / w
            box_height = 55 / h
        else:
            box_width = 100 / w
            box_height = 100 / h

        line = f'{cls} {xx/w} {yy/h} {box_width} {box_height}\n'
        yolo_data += line
    xc, yc = rotate_point((circle[0], circle[1]), M)
    line = f'6 {xc/w} {yc/h} {circle[2]/w} {circle[3]/h}'
    yolo_data += line

    # Save the rotated image
    output_image_path = f'{output_dir}/{os.path.splitext(os.path.basename(image_path))[0]}-{id_gen}.jpg'
    cv2.imwrite(output_image_path, rotated_image)

    # Save the YOLO annotations
    # annotation_path = f'{output_dir}/{idx:06d}-90.txt'
    annotation_path = f'{output_dir}/{os.path.splitext(os.path.basename(image_path))[0]}-{id_gen}.txt'

    with open(annotation_path, 'w') as f:
        f.write(yolo_data)


def normalize_coordinates(points, w_img, h_img):
    normalized_points = []
    for point in points:
        x_normalized = point[0] / w_img
        y_normalized = point[1] / h_img
        normalized_points += [x_normalized, y_normalized]
    return normalized_points


class CvatDataset(data.Dataset):
    def __init__(self, xml_path, root_img):
        # Parse the XML file
        tree = ET.parse(xml_path)
        # Get the root element
        root = tree.getroot()
        
        self.cls_names = {
            'line': 0,
            'edge': 1
        }
        self.clss_marked = {
            1: 0,
            2: 1,
            14: 2,
            13: 3,
            6: 4,
            8: 5,
            12: 6,
            11: 7,
            10: 8,
            9: 9
        }

        self.labels = []
        # Access elements and attributes
        for child in root:
            # print(child.tag, child.attrib)
            img_path = child.get('name')
            if img_path is None:
                continue
            img_path = f'{root_img}/{img_path}'
            annotations = {'img': img_path,
                           'labels': [],
                           'width': child.get('width'), 
                           'height': child.get('height')
            }
            for sub_child in child:
                if sub_child.tag == 'points':
                    label = sub_child.get('label')
                    points = sub_child.get('points')
                    idx_lmk = int(label[1:])
                    
                    label_yolo = idx_lmk - 1
                    if label_yolo is not None:
                        annotations['labels'].append([label_yolo, list(map(float, points.split(',')))])
                if sub_child.tag == 'ellipse':
                    label = sub_child.get('label')
                    if label == "circle":
                        cx = float(sub_child.get('cx'))
                        cy = float(sub_child.get('cy'))
                        rx = float(sub_child.get('rx'))
                        ry = float(sub_child.get('ry'))
                        hw = (rx + ry)
                        annotations['labels'].append([6, (cx, cy, hw, hw)])

            # print(annotations)
            if len(annotations['labels']):
                self.labels.append(annotations)

    def run(self, out, prefix="A"):
        X = []
        Y = []
        
        for idx, ano in enumerate(self.labels):
            x = ano.get('img')
            y = ano.get('labels')

            img = cv2.imread(x)
            h, w = img.shape[:2]
            
            y_scaled = []
            circle = []
            for cls, p in y:
                if cls == 6:
                    circle = p
                    continue
                xx, yy = p
                y_scaled.append((cls, xx, yy))
    
            x = f'{out}/{prefix}-{idx:06d}.jpg'
            cv2.imwrite(x, img)
                
            X.append(os.path.basename(x))
            Y.append(y_scaled)
            
            yolo_data = ''
            for idxx, p in enumerate(y_scaled):
                (cls, xx, yy) = p
                if cls == 4 or cls == 5:
                    line = f'{cls} {xx/w} {yy/h} {55/w} {55/w}\n'
                else:
                    line = f'{cls} {xx/w} {yy/h} {100/w} {100/w}\n'
                yolo_data += line
            
            print(circle)
            line = f'6 {circle[0]/w} {circle[1]/h} {circle[2]/w} {circle[3]/h}'
            yolo_data += line
            with open(f'{out}/{os.path.splitext(os.path.basename(x))[0]}.txt', 'w+') as f:
                f.write(yolo_data)
            
            # print(x)
            # print('==============')
            # image = cv2.imread(x)[:, :, ::-1]
            for j in range(8):
                rotate_image_and_annotations(x, y_scaled, circle, out, j)

        return X, Y
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]



out = 'align-lmk'
# if os.path.exists(out):
#     shutil.rmtree(out)

os.makedirs(out, exist_ok=True)

import concurrent.futures


if __name__ == '__main__':
    executor =  concurrent.futures.ProcessPoolExecutor(max_workers=10)

    data1 = CvatDataset(r'train-cvat-circle\set-failed-annotations.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\NGImage')
    print(len(data1))
    data1.run(out, "A")
    # executor.submit(data1.run, out, "A")


    # data2 = CvatDataset(r'train-cvat-circle\set1-100-annotations.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\SyntheticData2')
    # print(len(data2))
    # executor.submit(data2.run, out, "B")

    # data3 = CvatDataset(r'train-cvat-circle\set1-100-jig2-annotations.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\OriginalImage_JIG2_240604')
    # print(len(data3))
    # executor.submit(data3.run, out, "C")


    # data4 = CvatDataset(r'train-cvat-circle\set101-200-annotations.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\SyntheticData2')
    # print(len(data4))
    # executor.submit(data4.run, out, "D")

    # data5 = CvatDataset(r'train-cvat-circle\D-first.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\SyntheticData')
    # print(len(data5))
    # executor.submit(data5.run, out, "E")

    # data6 = CvatDataset(r'train-cvat-circle\set-night.xml', r'C:\Users\cuong\Desktop\Freelancer\Align\NightSet')
    # print(len(data6))
    # executor.submit(data6.run, out, "F")
