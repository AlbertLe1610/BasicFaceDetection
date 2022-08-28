import numpy as np
import cv2
import os
import torch
from PIL import Image
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def ExecuteRawData(annotate_file):
    string1 = '.jpg'
    data = {}
    image_name = ''
    value = 0
    bounding_box = []
    index = 0
    for line in annotate_file:
        index += 1
        # checking string is present in line or not
        if string1 in line:
            image_name = line
            continue
        if ' ' not in line:
            number_of_boardingbox = int(line)
            boarding_box = annotate_file[index:index + number_of_boardingbox]

            boarding_box = [(i.split(' '), 1)[0] for i in boarding_box]
            boarding_box_coordinate = []
            for i in range(0, number_of_boardingbox):
                boarding_box_coordinate.append([int(i) for i in boarding_box[i][:4]])
            data.update({image_name.strip(): boarding_box_coordinate})
    bounding_box_coordinate = list(data.items())
    final_datalist = list(filter(lambda x: 0 < len(x[1]) < 2, bounding_box_coordinate))
    final_datalist = [d for d in final_datalist if np.sum(d[1]) > 0]
    return final_datalist


def conver_xywh_xy(coor):
    """
    coor: [[x, y, w, h]]
    return [[x1, y1, x2, y2]]
    """
    x = coor[0][0]
    y = coor[0][1]
    w = coor[0][2]
    h = coor[0][3]
    x2 = x + w
    y2 = y + h
    return [[x, y, x2, y2]]



def ResizePadding(img, image_size=224):
    ratio = image_size / max(img.shape)
    img_resize = cv2.resize(img, None, fx=ratio, fy=ratio)
    k = abs(img_resize.shape[0] - img_resize.shape[1])  # padd bao nhieu hang cot
    if img_resize.shape[0] < img_resize.shape[1]:  # kiem tra xem padd hang hay pad cot
        imgPadding = cv2.copyMakeBorder(img_resize, 0, k, 0, 0, cv2.BORDER_CONSTANT, None, value=0)
    else:
        imgPadding = cv2.copyMakeBorder(img_resize, 0, 0, 0, k, cv2.BORDER_CONSTANT, None, value=0)
    return imgPadding


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotateFilepath, image_size=224):
        with open(annotateFilepath) as fp:
            annotate_file = fp.readlines()
        final_datalist = ExecuteRawData(annotate_file)
        imageAndBoundingBoxCoordinate = [(d[0], conver_xywh_xy(d[1])) for d in final_datalist]
        imageAndBoundingBoxCoordinate = dict(imageAndBoundingBoxCoordinate)
        self.name_list = list(imageAndBoundingBoxCoordinate.keys())  # name of image
        self.imageAndBoundingBoxCoordinate = imageAndBoundingBoxCoordinate
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_dir = img_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        image_name = self.name_list[idx]
        image_coordinate = self.imageAndBoundingBoxCoordinate.get(image_name)
        img = cv2.imread(os.path.join(self.image_dir, image_name))
        resize_padd_img = ResizePadding(img)
        pil_image = Image.fromarray(resize_padd_img)
        transformed_resize_padd_img = self.transform(pil_image)
        resize_bounding_box_coordinate = np.array(image_coordinate) / max(img.shape)
        resize_bounding_box_coordinate = torch.from_numpy(resize_bounding_box_coordinate)
        return resize_bounding_box_coordinate, transformed_resize_padd_img

if __name__ == '__main__':
    test = CustomImageDataset("D:\AI programme\Machine Learning final project\Dataset\Train Image\WIDER_train\images", "D:\AI programme\Machine Learning final project\Dataset\Annotation\wider_face_split/wider_face_train_bbx_gt.txt")
    print(test[23][1].shape)

