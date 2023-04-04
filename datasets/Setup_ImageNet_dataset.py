import sys
import os
import wget
from zipfile import ZipFile
import pandas as pd

path = "./ImageNet/"



# ==== Organize validation data folder in Imagenet to make it compatible with pytorch.
# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir    = os.path.join(path, 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val')
val_labels_file = path+"ILSVRC2017_devkit/ILSVRC/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"
map_clsloc_file = path+"ILSVRC2017_devkit/ILSVRC/devkit/data/map_clsloc.txt"

# Create mapping dictionary
if os.path.exists(map_clsloc_file):
   map_data = pd.read_csv(map_clsloc_file, sep=" ", header=None)
   map_data.columns = ["Class Name", "Object ID", "Object Name"]
   ObjectID_ClassName_ObjectName_dic = {}
   for i in range(len(map_data)):
      ObjectID_ClassName_ObjectName_dic[map_data["Object ID"][i]]= [map_data["Class Name"][i], map_data["Object Name"][i]]


if os.path.exists(val_labels_file):
   list_of_val_imgs_labels = pd.read_csv(val_labels_file, sep=" ", header=None)

if os.path.exists(val_img_dir):
   list_of_val_imgs = sorted(os.listdir(val_img_dir))

if not os.path.isdir(val_img_dir+"/"+list_of_val_imgs[5]):

   for i, img_file in enumerate(list_of_val_imgs):
      
      # Get image path
      img_path = val_img_dir+"/"+img_file
      
      # Get label
      label_id = list_of_val_imgs_labels[0][i]

      # Find class from label
      img_class_name, _ = ObjectID_ClassName_ObjectName_dic[label_id]

      # Check if directory with class name exists, if not create it.
      class_name_dir = val_img_dir+"/"+img_class_name
      if not os.path.exists(class_name_dir):
         os.makedirs(class_name_dir)
      # Move image to its class name directory.
      if os.path.exists(os.path.join(val_img_dir, img_file)):
         os.rename(os.path.join(val_img_dir, img_file), os.path.join(class_name_dir, img_file))
         
   print("Re-oragnised Imagenet val to Pytorch format.")

else:

   print("Imagenet val already oragnised to Pytorch format. Assumed because " +val_img_dir+" directory only contains other directories.")

