import cv2
import os
import json
img=cv2.imread("1.jpg")
annotations = json.load(open(os.path.join("../autodl-tmp/datasets_keypoint/STARE", 
                                                       'annotations', "train.json")))
for i in annotations:
    if (i["image_name"])=='12.ppm':
        x,y=i["keypoints"][:2]
        print(x,y)
        cv2.circle(img, (100,200), 8, (255, 0, 0), -1)
        cv2.imwrite('2.jpg',img)