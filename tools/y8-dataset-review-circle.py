import cv2
import numpy as np
from random import randint
import os


for i in os.listdir("align-circle")[::-1]:
    if i.endswith(".txt"):
        continue
    print(i)
    item = os.path.basename(os.path.splitext(i)[0])
    
    with open(f'align-circle/{item}.txt', 'r') as f:
        labels = f.read().splitlines()
        
    img = cv2.imread(f'align-circle/{item}.jpg')

    h,w = img.shape[:2]

    for label in labels:
        class_id, *poly = label.split(' ')
        
        poly = np.asarray(poly, dtype=np.float16).reshape(-1,2) # Read poly, reshape
        # print(poly)
        poly *= [w,h] # Unscale
        
        cv2.polylines(img, [poly.astype('int')], True, (randint(0,255),randint(0,255),randint(0,255)), 2) # Draw Poly Lines
        # cv2.fillPoly(img, [poly.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area

    cv2.imshow('image', cv2.resize(img, (920, 920)))
    cv2.waitKey(0)

