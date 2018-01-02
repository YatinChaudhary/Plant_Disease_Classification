import os 
import numpy as np 

data_dir = "/home/kishan/Desktop/datasets/"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
# dirs_in_folder = [ for d in os.listdir(raasta) if os.path.isdir(os.path.join(raasta, d))]
dirs_in_train_folder = os.walk(train_dir).next()

for directori in dirs_in_train_folder[1]:
    if not os.path.exists(os.path.join(val_dir, directori)):
        os.makedirs(os.path.join(val_dir, directori))
    
    if not (os.walk(os.path.join(val_dir, directori)).next()[2]):
        current_dir  = os.path.join(train_dir, directori)

        current_class_files = os.walk(current_dir).next()[2]
        current_class_files_count = len(current_class_files)

        val_files_count = current_class_files_count//5
        indices_to_move = np.random.sample(range(current_class_files_count), val_files_count)
        
        for i in indices_to_move:
            os.rename(os.path.join(train_dir, directori, current_class_files[i]), 
                      os.path.join(val_dir, directori, current_class_files[i]))