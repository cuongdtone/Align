import cv2
import numpy as np
from random import randint
import os

root = 'align-lmk'

for i in os.listdir(root):
    if i.endswith('.txt'):
        continue
    
    item = os.path.basename(os.path.splitext(i)[0])
    
    with open(f'{root}/{item}.txt', 'r') as f:
        labels = f.read().splitlines()
        
    img = cv2.imread(f'{root}/{item}.jpg')

    h,w = img.shape[:2]

    for label in labels:
        class_id, center_x, center_y, bbox_width, bbox_height = map(float, label.split(' '))

        center_x = int(center_x * w)
        center_y = int(center_y * h)
        bbox_width = int(bbox_width * w)
        bbox_height = int(bbox_height * h)
        
        top_left_x = int(center_x - bbox_width / 2)
        top_left_y = int(center_y - bbox_height / 2)
        bottom_right_x = int(center_x + bbox_width / 2)
        bottom_right_y = int(center_y + bbox_height / 2)
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
        cv2.putText(img, str(int(class_id)), (int(center_x)+5, int(center_y)+5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # cv2.resizeWindow('image', 1280, 1280)
    img = cv2.resize(img, (928, 928))
    print(i)
    # cv2.imwrite(f"review_data/{i}", img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

