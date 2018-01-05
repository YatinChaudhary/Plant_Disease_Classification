import glob, os
import shutil
import numpy as np

root = "../PlantVillage_Dataset/raw"
target = "../partitioned_dataset"

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

	for plant_name in plant_names_list:
		if not os.path.exists(os.path.join(train_plant_type_set_path, plant_name)):
			os.makedirs(os.path.join(train_plant_type_set_path, plant_name))
			os.makedirs(os.path.join(val_plant_type_set_path, plant_name))
			os.makedirs(os.path.join(test_plant_type_set_path, plant_name))

	for disease_name in disease_names_list:
		if not os.path.exists(os.path.join(train_disease_type_set_path, disease_name)):
			os.makedirs(os.path.join(train_disease_type_set_path, disease_name))
			os.makedirs(os.path.join(val_disease_type_set_path, disease_name))
			os.makedirs(os.path.join(test_disease_type_set_path, disease_name))

	print len(plant_names_list), len(disease_names_list)
	write_list_in_file(plant_type_txt, plant_names_list)
	write_list_in_file(disease_type_txt, disease_names_list)

	for dir in dirs_rootdir:
		plant_name, disease_name = dir.split("___")
		print plant_name, disease_name

		current_dir = os.path.join(rootdir, dir)
		files_current_dir = os.walk(current_dir).next()[2]
		files_count_current_dir = len(files_current_dir)
		
		val_set_files_count = files_count_current_dir // 5
		test_set_files_count = files_count_current_dir // 5
		train_set_files_count = files_count_current_dir - (val_set_files_count + test_set_files_count)
		
		index_array = np.arange(files_count_current_dir)
		np.random.shuffle(index_array)

		print files_count_current_dir, val_set_files_count, test_set_files_count, train_set_files_count, len(index_array)

		try:
			val_set_file_indices = index_array[0: val_set_files_count]
			test_set_file_indices = index_array[val_set_files_count: val_set_files_count + test_set_files_count]
			train_set_file_indices = index_array[val_set_files_count + test_set_files_count :]
			#print "chk-1"
			val_plant_set_path = os.path.join(val_plant_type_set_path, plant_name)
			val_disease_set_path = os.path.join(val_disease_type_set_path, disease_name)
			for index in val_set_file_indices:
				file_path = os.path.join(current_dir, files_current_dir[index])
				if not os.path.exists(os.path.join(val_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, val_plant_set_path)
				if not os.path.exists(os.path.join(val_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, val_disease_set_path)
				#shutil.copyfile(file_path, os.path.join(val_plant_set_path, files_current_dir[index]))
				#shutil.copyfile(file_path, os.path.join(val_disease_set_path, files_current_dir[index]))
			#print "chk-2"
			test_plant_set_path = os.path.join(test_plant_type_set_path, plant_name)
			test_disease_set_path = os.path.join(test_disease_type_set_path, disease_name)
			for index in test_set_file_indices:
				file_path = os.path.join(current_dir, files_current_dir[index])
				if not os.path.exists(os.path.join(test_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, test_plant_set_path)
				if not os.path.exists(os.path.join(test_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, test_disease_set_path)
				#shutil.copyfile(file_path, os.path.join(test_plant_set_path, files_current_dir[index]))
				#shutil.copyfile(file_path, os.path.join(test_disease_set_path, files_current_dir[index]))
			#print "chk-3"
			train_plant_set_path = os.path.join(train_plant_type_set_path, plant_name)
			train_disease_set_path = os.path.join(train_disease_type_set_path, disease_name)
			for index in train_set_file_indices:
				file_path = os.path.join(current_dir, files_current_dir[index])
				if not os.path.exists(os.path.join(train_plant_set_path, files_current_dir[index])):
					shutil.copy2(file_path, train_plant_set_path)
				if not os.path.exists(os.path.join(train_disease_set_path, files_current_dir[index])):
					shutil.copy2(file_path, train_disease_set_path)
				#shutil.copyfile(file_path, os.path.join(train_plant_set_path, files_current_dir[index]))
				#shutil.copyfile(file_path, os.path.join(train_disease_set_path, files_current_dir[index]))
			#print "chk-4"
		except Exception:
			print "error in " + plant_name



	"""
	for i in range(0,38):
		folder_name = "c_" + str(i)
		curr_root = os.path.join(path_source, folder_name)

		for filename in glob.iglob(os.path.join(curr_root, r'*.jpg')):
			title, ext = os.path.splitext(os.path.basename(filename))
			new_filename = str(i)+"_"+str(image_index)
			src_file = os.path.join(curr_root,new_filename + ext)			
			os.rename(filename, src_file)

			shutil.copy2(src_file,path_destination)
			image_index +=1
			with open(root_txt, "a") as f:
				f.write(new_filename+"\n")

		for filename in glob.iglob(os.path.join(curr_root, r'*.JPG')):
			title, ext = os.path.splitext(os.path.basename(filename))
			new_filename = str(i)+"_"+str(image_index)
			ext = ".jpg"
			src_file = os.path.join(curr_root,new_filename + ext)			
			os.rename(filename, src_file)

			shutil.copy2(src_file,path_destination)
			image_index +=1
			with open(root_txt, "a") as f:
				f.write(new_filename+"\n")
	"""

def write_list_in_file(file_path, list):
	with open(file_path, "w") as f:
		for i in range(len(list)):
			f.write(list[i] + "=" + str(i) + "\n")

	
def create_dataset_root(root):
	create_dataset(root,"segmented")
	#create_dataset(root,"color")
	#create_dataset(root,"grayscale")

create_dataset_root(root)

