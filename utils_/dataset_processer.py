import os
import json
import pandas as pd
from .utils_ import contour_to_bbox
import glob
import xml.etree.ElementTree as ET
import shutil

def xyxy2xywh(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + (width / 2)
    y_center = ymin + (height / 2)
    
    return x_center, y_center, width, height
def create_coco_annotation(image_id, annotation_id, x_center, y_center):
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [],
        "area": 0,
        "bbox": [x_center, y_center,2],
        "iscrowd": 0
    }
    return annotation

class BaseDB:
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

    def process_split(self, split, split_name):
        raise NotImplementedError("Subclasses must implement this method.")

    def parse(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
class DRIONS_DB(BaseDB):
    '''
    DRIONS-DB
            │
            ├── experts_annotation
            │   ├── anotExpert1_001.txt
            │   ├── anotExpert1_002.txt
            │   ├── ...
            │
            └── images
                ├── image_001.jpg
                ├── image_002.jpg
                ├── ...
    '''

    def process_split(self, split, split_name):
        image_file = os.path.join(self.data_path, 'images')
        annotation_file = os.path.join(self.data_path, 'experts_anotation')

        coco_annotations = []

        for idx, split_number in enumerate(split):
            annotation = contour_to_bbox(os.path.join(annotation_file, f"anotExpert1_{split_number}.txt"))
            x_center, y_center, _, _ = annotation
            coco_annotations.append(create_coco_annotation(int(split_number), idx, x_center, y_center))

            # Copy the original image to target_path/images
            shutil.copy(os.path.join(image_file, f"{split_number}.jpg"),
                        os.path.join(self.target_path, 'images', f"{idx}.jpg"))

        with open(os.path.join(self.target_path, 'annotations', f"{split_name}.json"), 'w') as f:
            json.dump(coco_annotations, f)

    def parse(self):
        train_split = [f"{number:03}" for number in range(1, 101)]
        val_split = [f"{number:03}" for number in range(101, 106)]
        test_split = [f"{number:03}" for number in range(106, 111)]

        self.process_split(train_split, 'train')
        self.process_split(val_split, 'valid')
        self.process_split(test_split, 'test')


class HRF_DB(BaseDB):
    '''HRF
    │
    ├── annotations.xml
    └── images
        ├── 01_dr.JPG
        ├── 02_dr.JPG
        ├── ... '''
    def process_split(self, split, split_name):
        annotations = pd.read_excel(os.path.join(self.data_path, 'annotation.xls'), engine='openpyxl')

        coco_annotations = []

        for idx, row in annotations.iterrows():
            if idx not in split:
                continue

            image_name = row['image']
            center_x = row['Pap. Center x']
            center_y = row['Pap. Center y']

            coco_annotations.append(create_coco_annotation(int(image_name), idx, center_x, center_y))

            # Copy the original image to target_path/images
            shutil.copy(os.path.join(self.data_path, 'images', f"{image_name}.jpg"),
                        os.path.join(self.target_path, 'images', f"{idx}.jpg"))

        with open(os.path.join(self.target_path, 'annotations', f"{split_name}.json"), 'w') as f:
            json.dump(coco_annotations, f)
    def parse(self):
        total_images = 45
        train_ratio = 0.7
        val_ratio = 0.2

        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        indices = list(range(total_images))

        train_split = indices[:train_count]
        val_split = indices[train_count:train_count + val_count]
        test_split = indices[train_count + val_count:]

        self.process_split(train_split, 'train')
        self.process_split(val_split, 'valid')
        self.process_split(test_split, 'test')

class STARE_DB(BaseDB):
    ''' STARE
        │
        ├── image
        │   ├── im001.ppm
        │   ├── ...
        │   └── im319.ppm
        │
        └── annotation.txt'''
    def process_split(self, split, split_name):
        coco_annotations = []

        for idx, xml_file in enumerate(split):
            filename, xmin, ymin, xmax, ymax = self.parse_xml(xml_file)
            x_center, y_center, _,_ = xyxy2xywh(xmin, ymin, xmax, ymax)

            coco_annotations.append(create_coco_annotation(int(filename), idx, x_center, y_center))

            # Copy the original image to target_path/images
            shutil.copy(os.path.join(self.data_path, 'images', f"{filename}.jpg"),
                        os.path.join(self.target_path, 'images', f"{idx}.jpg"))

        with open(os.path.join(self.target_path, 'annotations', f"{split_name}.json"), 'w') as f:
            json.dump(coco_annotations, f)

    def parse(self):
        total_images = 319
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        indices = list(range(1, total_images + 1))

        train_split = indices[:train_count]
        val_split = indices[train_count:train_count + val_count]
        test_split = indices[train_count + val_count:]

        self.process_split(train_split, 'train')
        self.process_split(val_split, 'valid')
        self.process_split(test_split, 'test')


class ODVOC_DB(BaseDB):
    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            return filename, xmin, ymin, xmax, ymax

    def process_split(self, split, split_name):
        coco_annotations = []

        for idx, xml_file in enumerate(split):
            filename, xmin, ymin, xmax, ymax = self.parse_xml(xml_file)
            x_center, y_center, _,_ = xyxy2xywh(xmin, ymin, xmax, ymax)

            shutil.copy(os.path.join(self.data_path, 'images', f"{filename}.jpg"),
                        os.path.join(self.target_path, 'images', f"{idx}.jpg"))
            coco_annotations.append(create_coco_annotation(int(filename), idx, x_center, y_center))

        with open(os.path.join(self.target_path, 'annotations', f"{split_name}.json"), 'w') as f:
            json.dump(coco_annotations, f)

    def parse(self):
        xml_files = glob.glob(os.path.join(self.data_path, 'annotations', '*.xml'))
        
        total_images = len(xml_files)
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        train_split = xml_files[:train_count]
        val_split = xml_files[train_count:train_count + val_count]
        test_split = xml_files[train_count + val_count:]

        self.process_split(train_split, 'train')
        self.process_split(val_split, 'valid')
        self.process_split(test_split, 'test')

import os
import json
from PIL import Image

class GY_DB:
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

        if not os.path.exists(os.path.join(target_path, 'images')):
            os.makedirs(os.path.join(target_path, 'images'))
        if not os.path.exists(os.path.join(target_path, 'annotations')):
            os.makedirs(os.path.join(target_path, 'annotations'))

    def process(self):
        image_files = [f for f in os.listdir(os.path.join(self.data_path, 'images')) if f.endswith('.jpg')]

        for img_name in image_files:
            img_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(self.data_path, 'images', img_name)
            label_path = os.path.join(self.data_path, 'labels', f"{img_id}.txt")

            img = Image.open(img_path)
            width, height = img.size

            coco_annotations = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split(' '))
                    x_center, y_center = int(x * width), int(y * height)
                    coco_annotations.append(create_coco_annotation(img_id, len(coco_annotations), x_center, y_center))

            with open(os.path.join(self.target_path, 'annotations', f"{img_id}.json"), 'w') as f:
                json.dump(coco_annotations, f)

            img.save(os.path.join(self.target_path, 'images', f"{img_id}.png"))
