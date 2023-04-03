import sys
import os
import wget
from zipfile import ZipFile

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
path = "./TinyImageNet/"

if not os.path.isdir(path):
   os.makedirs(path)

if len(os.listdir(path))==0:
   wget.download(url,out = path)
   print("Downloaded TinyImagnet zip file.")
        
if len(os.listdir(path))<2:
   with ZipFile(path+"tiny-imagenet-200.zip", "r") as file:
       file.extractall(path)
   print("Extracted TinyImagnet files.")
else:
   print("TinyImagnet files already exist.")



# ==== Organize validation data folder in Tiny Imagenet to make it compatible with pytorch.
# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_dir    = os.path.join(path, 'tiny-imagenet-200/val')
if os.path.exists(val_dir+"/images"):
   val_img_dir = os.path.join(val_dir, 'images')

   # Open and read val annotations text file
   fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
   data = fp.readlines()

   # Create dictionary to store img filename (word 0) and corresponding
   # label (word 1) for every line in the txt file (as key value pair)
   val_img_dict = {}
   for line in data:
      words = line.split('\t')
      val_img_dict[words[0]] = words[1]
   fp.close()

   # Create subfolders (if not present) for validation images based on label,
   # and move images into the respective folders
   for img, folder in val_img_dict.items():
      newpath_imgs       = (os.path.join(val_dir, folder,"images"))

      if not os.path.exists(newpath_imgs):
         os.makedirs(newpath_imgs)

      if os.path.exists(os.path.join(val_img_dir, img)):
         os.rename(os.path.join(val_img_dir, img), os.path.join(newpath_imgs, img))
         
   # Delete old images folder after finishing oraginsiign the images
   if os.path.exists(val_img_dir):
      os.rmdir(val_img_dir)
   print("Re-oragnised TinyImagenet val to Pytorch format.")

else:
   print("TinyImagenet already oragnised to Pytorch format. Assumed because images folder does not exist.")

