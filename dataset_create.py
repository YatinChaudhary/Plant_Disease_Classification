import glob, os
import shutil
import numpy as np
import cPickle as pickle

root = "../PlantVillage_Dataset/raw"
#target = "../partitioned_dataset"

### FOR DEVELOPMENT DATASET
target = "../partitioned_dataset_dev"

if not os.path.exists(target):
	os.makedirs(target)

def create_dataset(root, f_name):
	plant_type = "Plant_type_set"
	disease_type = "Disease_type_set"
	train_set = "Training_set"
	val_set = "Validation_set"
	test_set = "Test_set"

	rootdir = os.path.join(root, f_name)
	target_dir = os.path.join(target, f_name)

	plant_type_set_path = os.path.join(target_dir, plant_type)
	train_plant_type_set_path = os.path.join(plant_type_set_path, train_set)
	val_plant_type_set_path = os.path.join(plant_type_set_path, val_set)
	test_plant_type_set_path = os.path.join(plant_type_set_path, test_set)

	disease_type_set_path = os.path.join(target_dir, disease_type)
	train_disease_type_set_path = os.path.join(disease_type_set_path, train_set)
	val_disease_type_set_path = os.path.join(disease_type_set_path, val_set)
	test_disease_type_set_path = os.path.join(disease_type_set_path, test_set)

	if not os.path.exists(plant_type_set_path):
		os.makedirs(plant_type_set_path)
		os.makedirs(train_plant_type_set_path)
		os.makedirs(val_plant_type_set_path)
		os.makedirs(test_plant_type_set_path)
	
	if not os.path.exists(disease_type_set_path):
		os.makedirs(disease_type_set_path)
		os.makedirs(train_disease_type_set_path)
		os.makedirs(val_disease_type_set_path)
		os.makedirs(test_disease_type_set_path)

	plant_type_txt = os.path.join(target_dir, plant_type+".txt")
	disease_type_txt = os.path.join(target_dir, disease_type+".txt")

	plant_names = []
	disease_names = []

	dirs_rootdir = os.walk(rootdir).next()[1]
	for dir in dirs_rootdir:
		plant_name, disease_name = dir.split("___")
		plant_names.append(plant_name)
		disease_names.append(disease_name)

	plant_names_list = list(set(plant_names))
	disease_names_list = list(set(disease_names))

	print(len(plant_names_list), len(disease_names_list))
	write_list_in_file(plant_type_txt, plant_names_list)
	write_list_in_file(disease_type_txt, disease_names_list)

	plant_type_dict = {}
	disease_type_dict = {}

	with open(plant_type_txt, "r") as f:
		for line in f:
			key, value = line.strip().split("=")
			plant_type_dict[key] = value

	with open(disease_type_txt, "r") as f:
		for line in f:
			key, value = line.strip().split("=")
			disease_type_dict[key] = value

	print(plant_type_dict, disease_type_dict)
	pickle_path = os.path.join(target_dir, "plant_type_dict.pickle")
	pickle_out = open(pickle_path, "wb")
	pickle.dump(plant_type_dict, pickle_out)
	pickle_out.close()

	pickle_path = os.path.join(target_dir, "disease_type_dict.pickle")
	pickle_out = open(pickle_path, "wb")
	pickle.dump(disease_type_dict, pickle_out)
	pickle_out.close()

	plant_counter_val = 0
	plant_counter_test = 0
	plant_counter_train = 0
	disease_counter_val = 0
	disease_counter_test = 0
	disease_counter_train = 0

	for dir in dirs_rootdir:
		plant_name, disease_name = dir.split("___")
		print(plant_name, disease_name)
		plant_name_id = plant_type_dict[plant_name]
		disease_name_id = disease_type_dict[disease_name]

		current_dir = os.path.join(rootdir, dir)
		files_current_dir = os.walk(current_dir).next()[2]
		files_count_current_dir = len(files_current_dir)
		
		#val_set_files_count = files_count_current_dir // 5
		#test_set_files_count = files_count_current_dir // 5
		#train_set_files_count = files_count_current_dir - (val_set_files_count + test_set_files_count)

		### FOR DEVELOPMENT DATASET
		val_set_files_count = files_count_current_dir // 20
		test_set_files_count = files_count_current_dir // 80
		train_set_files_count = files_count_current_dir - (val_set_files_count + test_set_files_count)
		
		index_array = np.arange(files_count_current_dir)
		np.random.shuffle(index_array)

		print(files_count_current_dir, val_set_files_count, test_set_files_count, train_set_files_count, len(index_array))

		try:
			val_set_file_indices = index_array[0: val_set_files_count]
			test_set_file_indices = index_array[val_set_files_count: val_set_files_count + test_set_files_count]
			train_set_file_indices = index_array[val_set_files_count + test_set_files_count :]
			#print "chk-1"
			with open(os.path.join(plant_type_set_path, val_set + ".txt"), "a") as f:
				for index in val_set_file_indices:
					file_name_plant = str(plant_name_id) + '_' + str(plant_counter_val)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(val_plant_type_set_path, file_name_plant + '.jpg'))
					f.write(file_name_plant + "\n")
					plant_counter_val += 1

			#print "chk-2"
			with open(os.path.join(disease_type_set_path, val_set + ".txt"), "a") as f:
				#print "chk-2-1"
				for index in val_set_file_indices:
					file_name_disease = str(disease_name_id) + '_' + str(disease_counter_val)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(val_disease_type_set_path, file_name_disease + '.jpg'))
					f.write(file_name_disease + "\n")
					disease_counter_val += 1

			#print "chk-3"
			with open(os.path.join(plant_type_set_path, test_set + ".txt"), "a") as f:
				for index in test_set_file_indices:
					file_name_plant = str(plant_name_id) + '_' + str(plant_counter_test)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(test_plant_type_set_path, file_name_plant + '.jpg'))
					f.write(file_name_plant + "\n")
					plant_counter_test += 1

			with open(os.path.join(disease_type_set_path, test_set + ".txt"), "a") as f:
				for index in test_set_file_indices:
					file_name_disease = str(disease_name_id) + '_' + str(disease_counter_test)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(test_disease_type_set_path, file_name_disease + '.jpg'))
					f.write(file_name_disease + "\n")
					disease_counter_test += 1
			
			### COMMENT BELOW TWO FOR LOOPS FOR DEVELOPMENT DATASET
			"""
			with open(os.path.join(plant_type_set_path, train_set + ".txt"), "a") as f:
				for index in train_set_file_indices:
					file_name_plant = str(plant_name_id) + '_' + str(plant_counter_train)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(train_plant_type_set_path, file_name_plant + '.jpg'))
					f.write(file_name_plant + "\n")
					plant_counter_train += 1

			with open(os.path.join(disease_type_set_path, train_set + ".txt"), "a") as f:
				for index in train_set_file_indices:
					file_name_disease = str(disease_name_id) + '_' + str(disease_counter_train)
					file_path = os.path.join(current_dir, files_current_dir[index])
					#if not os.path.exists(os.path.join(val_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, os.path.join(train_disease_type_set_path, file_name_disease + '.jpg'))
					f.write(file_name_plant + "\n")
					disease_counter_train += 1
			"""
		except Exception:
			print("error in ", plant_name)


def write_list_in_file(file_path, list):
	with open(file_path, "w") as f:
		for i in range(len(list)):
			f.write(list[i] + "=" + str(i) + "\n")

	
def create_dataset_root(root):
	create_dataset(root,"segmented")
	#create_dataset(root,"color")
	#create_dataset(root,"grayscale")

create_dataset_root(root)

