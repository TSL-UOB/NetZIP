import sys
import os
import wget

url = "https://image-net.org/data/ILSVRC/2017/ILSVRC2017_DET.tar.gz"
path = "./ImageNet/"

if not os.path.isdir(path):
   os.makedirs(path)

wget.download(url,out = path)
