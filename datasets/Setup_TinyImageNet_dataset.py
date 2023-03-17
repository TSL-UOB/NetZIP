import sys
import os
import wget

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