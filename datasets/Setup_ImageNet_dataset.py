import sys
import os
import wget

url = "https://image-net.org/data/ILSVRC/2017/ILSVRC2017_DET.tar.gz"
path = "./ImageNet/"

if not os.path.isdir(path):
   os.makedirs(path)

if len(os.listdir(path))==0:
   wget.download(url,out = path)
   print("Downloaded ImageNet zip file.")
        
# if len(os.listdir(path))<2:
#    with ZipFile(path+"tiny-imagenet-200.zip", "r") as file:
#        file.extractall(path)
#    print("Extracted ImageNet files.")
# else:
#    print("ImagNet files already exist.")
