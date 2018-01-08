import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy import io
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

root = "../PlantVillage_Dataset/raw"
target = "../partitioned_dataset"

plant_type = "Plant_type_set"
disease_type = "Disease_type_set"
train_set = "Training_set"
val_set = "Validation_set"
test_set = "Test_set"

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

        pickle_in = open(os.path.join(target, "segmented/plant_type_dict.pickle"), "rb")
        plant_type_dict = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open(os.path.join(target, "segmented/segmented_type_dict.pickle"), "rb")
        disease_type_dict = pickle.load(pickle_in)
        pickle_in.close()

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
