import os
import random

import time
import copy
import numpy as np

import shutil

import wget
from zipfile import ZipFile

import pandas as pd

import cv2

from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# =====================================================
# == Set random seeds
# =====================================================
def set_random_seeds(random_seed=0):
   np.random.seed(random_seed)
   random.seed(random_seed)

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.savefig("temp.png")

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
   for f in list_of_files:
      try:
         shutil.move(f, destination_folder)
      except:
         print(f)
      #    assert False

# =====================================================
# == Get COCO dataset 
# =====================================================

random_seed = 0
num_classes = 200

set_random_seeds(random_seed=random_seed)

def get_and_prepare_coco_dataset():
   path = "./COCO/"

   if not os.path.isdir(path):
        os.makedirs(path)


   # Download images files
   url = "http://images.cocodataset.org/zips/"
   fs= ['train2017.zip','val2017.zip']#, 'test2017.zip']
   for f in fs:        
      if not os.path.exists(path+f):
         wget.download(url+f,out = path)
         print("Downloaded COCO " + f + " zip file.")
      else:
         print("COCO " + f + " already exist")

   # Download COCO dataset annotations in json format
   # url = "http://images.cocodataset.org/annotations/"
   # fs= ["annotations_trainval2017.zip", "image_info_test2017.zip"]
   # for f in fs:        
   #    if not os.path.exists(path+f):
   #       wget.download(url+f,out = path)
   #       print("Downloaded COCO " + f + " zip file.")
   #    else:
   #       print("COCO " + f + " already exist")

   # Download COCO dataset annotations in yolo format
   url= "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
   f= 'coco2017labels.zip' 
   if not os.path.exists(path+f):
       wget.download(url+f,out = path)
       print("Downloaded COCO annotations in yolo format zip file.")
   else:
       print("COCO labels already exist")


   # Extract files
   fs= ['train2017.zip', 'val2017.zip', 'coco2017labels.zip']
   for f in fs:        
      if not os.path.exists(path+f[:-4]):
         with ZipFile(path+f, "r") as file:
            print("Extracting COCO " + f + " zip file, please wait ...")
            file.extractall(path)
         print("Extracted COCO " + f + " zip file.")
      else:
         print("COCO " + f + " already extracted")

   # Make images and labels directories
   if not os.path.exists(path+"images"):
      os.mkdir(path+"images")

   if not os.path.exists(path+"labels"):
      os.mkdir(path+"labels")

   temp_paths = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
   for temp_path in temp_paths:
      temp_path = path + temp_path
      if not os.path.isdir(temp_path):
              os.makedirs(temp_path)


   # Read images and annotations. Note did work around here to ignore images that do not have an annotation file.
   images = [os.path.join(path+'train2017', x[:-3]+"jpg") for x in os.listdir(path+'coco/labels/train2017')]
   annotations = [os.path.join(path+'coco/labels/train2017', x) for x in os.listdir(path+'coco/labels/train2017') if x[-3:] == "txt"]

   val_images = [os.path.join(path+'val2017', x[:-3]+"jpg") for x in os.listdir(path+'coco/labels/val2017')]
   val_annotations = [os.path.join(path+'coco/labels/val2017', x) for x in os.listdir(path+'coco/labels/val2017') if x[-3:] == "txt"]


   images.sort()
   annotations.sort()

   val_images.sort()
   val_annotations.sort()

   # Split the dataset into train-valid-test splits 
   train_images, test_images, train_annotations, test_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)

   # Move the splits into their folders
   move_files_to_folder(train_images, path+'images/train')
   move_files_to_folder(val_images, path+'images/val/')
   move_files_to_folder(test_images, path+'images/test/')
   move_files_to_folder(train_annotations, path+'labels/train/')
   move_files_to_folder(val_annotations, path+'labels/val/')
   move_files_to_folder(test_annotations, path+'labels/test/')

   # Remove files
   fs = ["coco", "test2017", "train2017", "val2017"]
   for f in fs:
      if os.path.exists(path+f):
         shutil.rmtree(path + f)


# =====================================================
# == Main
# =====================================================
setup_coco =  True
if setup_coco == True:
   get_and_prepare_coco_dataset()
