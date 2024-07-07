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
        epsilon = 0.001 * cv2.arcLength(contour, True)
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

    
# take these values from cvat xml file
# serialized_cvat_image = dict(
#  width="3200", height="1500"
# )

# serialized_cvat_rle = dict(
#     rle="89, 706, 111, 708, 109, 710, 107, 712, 105, 714, 104, 714, 103, 716, 101, 718, 99, 720, 97, 722, 95, 724, 94, 725, 92, 727, 90, 729, 88, 731, 86, 733, 84, 735, 83, 736, 81, 738, 79, 740, 77, 742, 75, 744, 73, 746, 71, 748, 70, 748, 69, 750, 67, 752, 66, 753, 65, 754, 64, 755, 63, 756, 63, 756, 62, 757, 61, 758, 60, 759, 59, 760, 59, 760, 58, 761, 57, 762, 56, 763, 55, 764, 54, 765, 54, 765, 53, 765, 53, 766, 52, 767, 51, 768, 51, 768, 50, 769, 49, 770, 48, 771, 47, 772, 46, 773, 46, 773, 45, 774, 44, 775, 44, 775, 43, 776, 43, 776, 42, 777, 42, 777, 41, 778, 41, 777, 41, 778, 41, 778, 40, 779, 40, 779, 40, 779, 39, 780, 39, 780, 38, 781, 38, 781, 37, 782, 37, 782, 36, 783, 36, 783, 35, 784, 35, 784, 34, 785, 34, 785, 33, 786, 33, 786, 32, 786, 33, 786, 32, 787, 32, 787, 31, 788, 31, 788, 30, 789, 30, 789, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 790, 29, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 789, 30, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 31, 788, 32, 71, 155, 561, 32, 68, 362, 357, 32, 65, 569, 153, 32, 62, 757, 59, 760, 56, 763, 53, 766, 53, 766, 52, 767, 52, 767, 51, 768, 51, 768, 51, 768, 50, 769, 50, 769, 49, 770, 49, 770, 48, 771, 48, 771, 47, 772, 47, 772, 46, 773, 46, 773, 46, 773, 45, 774, 45, 774, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 775, 44, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 776, 43, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 777, 42, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 41, 778, 42, 777, 42, 777, 43, 776, 43, 776, 43, 776, 44, 775, 44, 775, 45, 774, 45, 774, 45, 774, 46, 773, 46, 773, 46, 773, 47, 772, 47, 772, 48, 771, 48, 771, 48, 771, 49, 770, 49, 770, 50, 769, 51, 768, 52, 767, 54, 765, 56, 763, 58, 761, 60, 759, 62, 757, 63, 756, 65, 754, 67, 752, 69, 750, 71, 748, 219, 600, 516, 303, 814, 5, 815, 4, 814, 5, 814, 6, 813, 6, 813, 6, 813, 6, 812, 7, 812, 7, 812, 7, 812, 7, 812, 7, 812, 7, 811, 8, 811, 8, 811, 8, 811, 8, 811, 8, 810, 9, 810, 9, 810, 9, 810, 9, 810, 9, 810, 9, 809, 10, 809, 10, 809, 10, 809, 10, 809, 10, 808, 11, 808, 11, 808, 11, 808, 11, 808, 11, 808, 11, 807, 12, 807, 12, 807, 12, 807, 12, 807, 12, 806, 13, 806, 13, 806, 13, 806, 13, 806, 13, 806, 13, 805, 14, 805, 14, 805, 14, 805, 14, 805, 14, 804, 15, 804, 15, 804, 15, 804, 15, 804, 15, 804, 15, 803, 16, 803, 16, 803, 16, 803, 16, 803, 16, 802, 17, 802, 18, 801, 19, 800, 19, 800, 20, 799, 21, 797, 23, 796, 24, 795, 24, 795, 25, 794, 26, 792, 28, 791, 29, 790, 29, 790, 30, 789, 31, 787, 33, 786, 34, 785, 34, 785, 35, 784, 36, 783, 37, 781, 38, 781, 39, 780, 40, 779, 41, 778, 42, 776, 43, 776, 44, 775, 45, 774, 46, 773, 47, 772, 48, 770, 50, 769, 51, 768, 52, 767, 53, 766, 54, 764, 57, 762, 58, 761, 59, 760, 60, 759, 61, 758, 62, 756, 64, 755, 65, 754, 66, 753, 67, 752, 69, 749, 71, 748, 72, 747, 73, 746, 74, 745, 75, 744, 76, 742, 78, 741, 79, 740, 80, 739, 82, 737, 83, 735, 86, 733, 89, 730, 93, 726, 96, 723, 100, 719, 103, 715, 108, 711, 111, 708, 115, 704, 118, 701, 122, 696, 357, 462, 666, 153, 25",
#     left="73",
#     top="491",
#     width="819", 
#     height="508",
    
# )
# test(serialized_cvat_image, serialized_cvat_rle)