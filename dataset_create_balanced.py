## CREATING BALANCED DATASET BY REMOVING "RASPBERRY" CLASS BECAUSE OF LOW NUMBER OF IMAGES 
## AND RANDOMLY SELECTING 1500 IMAGES FROM EACH PLANT TYPE CLASS

import glob, os
import shutil
import numpy as np
import cPickle as pickle
from operator import add
import random

root = "../partitioned_dataset/segmented"

target = "../partitioned_dataset_balanced/segmented"

plant_type_dir = "Plant_type_set"

training_set = "Training_set"
validation_set = "Validation_set"
test_set = "Test_set"

dirs = os.walk(os.path.join(root, plant_type_dir)).next()[1]

files_num_list = [0 for x in range(14)]

files_per_plant_class = 1500

num_val_files_per_class = files_per_plant_class / 5
num_test_files_per_class = files_per_plant_class / 5
num_train_files_per_class = files_per_plant_class - ( num_val_files_per_class + num_test_files_per_class )

pickle_in = open(os.path.join(root, "plant_type_dict.pickle"), "rb")
plant_type_dict = pickle.load(pickle_in)
pickle_in.close()

print(dirs)

for directory in dirs:
    folder_path = os.path.join(root, plant_type_dir, directory)
    files = os.walk(folder_path).next()[2]
    print(len(files))
    random.shuffle(files)

    temp = [0 for x in range(14)]
    counter = 0
    src_file_path = os.path.join(root, plant_type_dir, directory)
    dst_file_path = os.path.join(target, plant_type_dir, directory)

    if not os.path.exists(dst_file_path):
        os.makedirs(dst_file_path)

    if directory == validation_set:
        file_limit = num_val_files_per_class
    elif directory == test_set:
        file_limit = num_test_files_per_class
    else:
        file_limit = num_train_files_per_class

    txt_file_name = str(directory) + ".txt"

    with open(os.path.join(target, plant_type_dir, txt_file_name), "w") as f:
        for file in files:
            plant_id, number = file.split("_")

            if ((int(plant_id) != 12) and (temp[int(plant_id)] < file_limit)):
                temp[int(plant_id)] += 1
                file_name = str(plant_id) + "_" + str(counter)
                shutil.copy2(os.path.join(src_file_path, file), os.path.join(dst_file_path, file_name + '.jpg'))
                f.write(file_name + '\n')
                counter += 1

    temp.sort()
    print(temp)
    files_num_list = map(add, temp, files_num_list)

files_num_list.sort()
print(files_num_list)

del plant_type_dict['Raspberry']
pickle_out = open(os.path.join(target, "plant_type_dict.pickle"), "wb")
pickle.dump(plant_type_dict, pickle_out)
pickle_out.close()

with open(os.path.join(target, "Plant_type_set.txt"), "w") as f:
    for key in plant_type_dict:
        if key != 'Raspberry':
            f.write(key + "=" + plant_type_dict[key] + "\n")