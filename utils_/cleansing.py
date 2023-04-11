import os
import json
import pandas as pd
from .utils_ import contour_to_bbox
import glob
import xml.etree.ElementTree as ET
from PIL import Image
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
        "bbox": [x_center, y_center, 0, 0],
        "iscrowd": 0
    }
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
            image_path = os.path.join(image_file, f"image_{split_number}.jpg")
            annotation = contour_to_bbox(os.path.join(annotation_file, f"anotExpert1_{split_number}.txt"))

            # Load the image and find its dimensions
            image = Image.open(image_path)
            width, height = image.size

            # Normalize the bounding box coordinates and dimensions
            x_center, y_center, bbox_width, bbox_height = annotation
            x_center /= width
            y_center /= height
            bbox_width /= width
            bbox_height /= height

            coco_annotations.append(create_coco_annotation(int(split_number), idx, x_center, y_center))

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
        # Read the 'annotation.xls' file
        annotations = pd.read_excel(os.path.join(self.data_path, 'annotation.xls'), engine='openpyxl')

        coco_annotations = []

        for idx, row in annotations.iterrows():
            if idx not in split:
                continue

            image_name = row['image']
            image_path = os.path.join(self.data_path, 'images', f"{image_name}.jpg")

            image = Image.open(image_path)
            width, height = image.size

            # Extract the optic disc center coordinates
            center_x = row['Pap. Center x']/width
            center_y = row['Pap. Center y']/height

            coco_annotations.append(create_coco_annotation(int(image_name), idx, center_x, center_y))

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
        with open(os.path.join(self.data_path, 'annotation.txt'), 'r') as f:
            coco_annotations = []

            for idx, line in enumerate(f):
                image_name, center_x, center_y = line.strip().split()

                if int(image_name[2:]) not in split:
                    continue
                
                image_path = os.path.join(self.data_path, 'image', f"{image_name}.ppm")
                image = Image.open(image_path)
                width, height = image.size
                center_x /= width
                center_y /= height

                coco_annotations.append(create_coco_annotation(int(image_name[2:]), idx, center_x, center_y))

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

            image_path = os.path.join(self.data_path, 'images', f"{filename}.png")
            image = Image.open(image_path)
            width, height = image.size
            x_center, y_center, bbox_width, bbox_height = xyxy2xywh(xmin, ymin, xmax, ymax)
            x_center /= width
            y_center /= height
            bbox_width /= width
            bbox_height /= height

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
