import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy import io
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms



diseases_list = [{"id": 0,  "name": "Apple Scab"},
                 {"id": 1,  "name": "Apple Black Rot"},
                 {"id": 2,  "name": "Apple Cedar Rust"},
                 {"id": 3,  "name": "Apple Healthy"},
                 {"id": 4,  "name": "Blueberry Healthy"},
                 {"id": 5,  "name": "Cherry Healthy"},
                 {"id": 6,  "name": "Cherry Powdery Mildew"},
                 {"id": 7,  "name": "Corn Gray Leaf Spot"},
                 {"id": 8,  "name": "Corn Common Rust"},
                 {"id": 9,  "name": "Corn Healthy"},
                 {"id": 10, "name": "Corn Northern Leaf Blight"},
                 {"id": 11, "name": "Grape Black Rot"},
                 {"id": 12, "name": "Grape Black Measles"},
                 {"id": 13, "name": "Grape Healthy"},
                 {"id": 14, "name": "Grape Leaf Blight"},
                 {"id": 15, "name": "Orange Huanglongbing"},
                 {"id": 16, "name": "Peach Bacterial Spot"},
                 {"id": 17, "name": "Peach Healthy"},
                 {"id": 18, "name": "Bell Pepper Bacterial Spot"},
                 {"id": 19, "name": "Bell Pepper Healthy"},
                 {"id": 20, "name": "Potato Early Blight"},
                 {"id": 21, "name": "Potato Healthy"},
                 {"id": 22, "name": "Potato Late Blight"},
                 {"id": 23, "name": "Raspberry Healthy"},
                 {"id": 24, "name": "Soybean Healthy"},
                 {"id": 25, "name": "Squash Powdery Mildew"},
                 {"id": 26, "name": "Strawberry Healthy"},
                 {"id": 27, "name": "Strawberry Leaf Scorch"},
                 {"id": 28, "name": "Tomato Bacterial Spot"},
                 {"id": 29, "name": "Tomato Early Blight"},
                 {"id": 30, "name": "Tomato Late Blight"},
                 {"id": 31, "name": "Tomato Leaf Mold"},
                 {"id": 32, "name": "Tomato Septoria Leaf Spot"},
                 {"id": 33, "name": "Tomato two Spotted Spider Mite"},
                 {"id": 34, "name": "Tomato Target Spot"},
                 {"id": 35, "name": "Tomato Mosaic Virus"},
                 {"id": 36, "name": "Tomato Yellow Leaf Curl Virus"},
                 {"id": 37, "name": "Tomato Healthy"}]


class ClassificationData(data.Dataset):

    def __init__(self, root, image_list):
        self.root = root
        filename, ext = image_list.split('.')
        self.foldername = filename + "_set"
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Scale(240),
            transforms.ToTensor(),
        ])

        with open(os.path.join(self.root, image_list)) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, index):
        #img_id, ext = self.image_names[index].split('.')
        img_id, ext = os.path.splitext(os.path.basename(self.image_names[index]))

        tgr, num = img_id.split('_')

        img = Image.open(os.path.join(self.root, self.foldername, img_id + '.jpg')).convert('RGB')

        img = self.transform(img)

        target = int(tgr)
        name = diseases_list[target]["name"]

        #target_labels = (target, name)
        target_labels = target

        return img, target_labels

    def __len__(self):
        return len(self.image_names)
