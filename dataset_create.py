
import glob, os
import shutil

root = "/home/kishan/Desktop/datasets/train"
def create_dataset(root,f_name):
	root_set_path = os.path.join(root, f_name + "_set") 
	if not os.path.exists(root_set_path):
		os.makedirs(root_set_path)
	
	root_txt = os.path.join(root,f_name+".txt")
	image_index = 1;
	path_source = os.path.join(root,f_name)
	path_destination = root_set_path

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

def create_dataset_root(root):
	create_dataset(root,"train")
	create_dataset(root,"val")
	create_dataset(root,"test")

create_dataset_root(root)

