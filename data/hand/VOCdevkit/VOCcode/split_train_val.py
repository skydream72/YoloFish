def getTargetIds(jsonData, img_name):
    for dest in jsonData:
        path_list = dest['filename'].split(os.sep)
        path_list_len = len(path_list)
        if(path_list[path_list_len - 1] == img_name):
            #print("filename:", path_list[path_list_len - 1])
            return dest['annotations']
            
def find_values(id, json_repr):
    results = []

    def _decode_dict(a_dict):
        try: results.append(a_dict[id])
        except KeyError: pass
        return a_dict

    json.loads(json_repr, object_hook=_decode_dict)  # return value ignored
    return results

def create_annontations(img_name, source, fishId, hsuDataAugmentation, rotationAugmentation):
    im=Image.open(source)
    width, height = im.size
    
    json_tag = os.path.join(path_root_images, FishJsonTags[fishId])
    #print('# img name: {}, # source: {}, # fishId: {}, # fish tag file: {}, #width: {}, #height:{}'.format(img_name, source, fishId, json_tag, width, height))
    
    jdata = json.loads(open (json_tag).read())
    annotations = getTargetIds(jdata, img_name)
    #print("annotations:", annotations)
    annotation_size = len(annotations)
    #print("annotations_size:", annotation_size)
    
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOCFish"
    ET.SubElement(root, "filename").text = img_name
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Kggle"
    owner = ET.SubElement(root, "owner")
    ET.SubElement(owner, "name").text = "Leo"
    
    image_size = ET.SubElement(root, "size")
    ET.SubElement(image_size, "width").text = str(width)
    ET.SubElement(image_size, "height").text = str(height)
    ET.SubElement(image_size, "depth").text = '3'
    
    ET.SubElement(root, "segmented").text = "0"
    
    for i in range(annotation_size):
        fish_object = ET.SubElement(root, "object")
        ET.SubElement(fish_object, "name").text = FishNamesLow[fishId]
        ET.SubElement(fish_object, "pose").text = "Unspecified"
        ET.SubElement(fish_object, "truncated").text = '0'
        ET.SubElement(fish_object, "difficult").text ='0'
        
        x_start = annotations[i]['x']
        x_start = 1 if x_start <= 0 else x_start
        x_end = annotations[i]['x'] + annotations[i]['width']
        x_end = width-1 if x_end >= width else x_end
        y_start = annotations[i]['y']
        y_start = 1 if y_start <= 0 else y_start
        y_end =  annotations[i]['y'] + annotations[i]['height']
        y_end = height-1 if y_end >= height else y_end
        
        bndbox = ET.SubElement(fish_object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x_start)
        ET.SubElement(bndbox, "ymin").text = str(y_start)
        ET.SubElement(bndbox, "xmax").text = str(x_end)
        ET.SubElement(bndbox, "ymax").text = str(y_end)
        
    tree = ET.ElementTree(root)
    annotationsfile = os.path.join(dir_images_annotations_full, os.path.splitext(img)[0] + ".xml")
    tree.write(annotationsfile)
    
    if(hsuDataAugmentation):
        annotationsfile = os.path.join(dir_images_annotations_full, 'v'+ os.path.splitext(img)[0] + ".xml")
        tree.write(annotationsfile)
        
    if(rotationAugmentation):
        for j in range(3):
            if(j==1):
                for i in range(annotation_size):
                    fish_object = ET.SubElement(root, "object")
                    ET.SubElement(fish_object, "name").text = FishNamesLow[fishId]
                    ET.SubElement(fish_object, "pose").text = "Unspecified"
                    ET.SubElement(fish_object, "truncated").text = '0'
                    ET.SubElement(fish_object, "difficult").text ='0'

                    x_start = annotations[i]['x']
                    x_start = 1 if x_start <= 0 else x_start
                    x_end = annotations[i]['x'] + annotations[i]['width']
                    x_end = width-1 if x_end >= width else x_end
                    y_start = annotations[i]['y']
                    y_start = 1 if y_start <= 0 else y_start
                    y_end =  annotations[i]['y'] + annotations[i]['height']
                    y_end = height-1 if y_end >= height else y_end

                    bndbox = ET.SubElement(fish_object, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(x_start)
                    ET.SubElement(bndbox, "ymin").text = str(y_start)
                    ET.SubElement(bndbox, "xmax").text = str(x_end)
                    ET.SubElement(bndbox, "ymax").text = str(y_end)
           
                annotationsfile = os.path.join(dir_images_annotations_full, 'r' + str(j) + os.path.splitext(img)[0] + ".xml")
                tree.write(annotationsfile)
                
def data_augmentation_hsu(target_dir, img):
    im = Image.open(os.path.join(target_dir, img))
    ld = im.load()
    width, height = im.size
    for y in range(height):
        for x in range(width):
            r,g,b = ld[x,y]
            h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            h = (h + -90.0/360.0) % 1.0   # hue
            s = s**0.25                   # saturation
            r,g,b = colorsys.hsv_to_rgb(h, s, v)
            ld[x,y] = (int(r * 255.9999), int(g * 255.9999), int(b * 255.9999))
    #im.show()
    ####################################
    # To save the image:
    im.save(os.path.join(target_dir, 'v' + img))
    
def data_augmentation_rotation(target_dir, img):
    im = Image.open(os.path.join(target_dir, img))
    for j in range(3):
        if(j==1):
            im_r = im.rotate(-90*(j+1), expand=True)
            im_r.save(os.path.join(target_dir, 'r' + str(j) + img))
        
import os
import numpy as np
import shutil
from PIL import Image, ImageDraw
import json
import xml.etree.cElementTree as ET
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import colorsys

from PIL import Image

dir_images_train_full = '../VOCHand/JPEGImages/'
dir_images_annotations_full = '../VOCHand/Annotations/'
dir_images_layout_full = '../VOCHand/ImageSets/Main/'

FishNames = ['Hand']
FishNamesLow = ['hand']

# the sum of training, val, test proportion,  must equal 1
split_train_proportion = 0.7
split_val_proportion = 0.1
split_test_proportion = 0.2

if not os.path.exists(dir_images_train_full):
    os.makedirs(dir_images_train_full)
if not os.path.exists(dir_images_annotations_full):
    os.makedirs(dir_images_annotations_full)    
if not os.path.exists(dir_images_layout_full):
    os.makedirs(dir_images_layout_full)

total_train_images = []
total_val_images = []
total_test_images = []
total_trainval_images = []

total_images = os.listdir(dir_images_train_full)
for fish_img in total_images:
    fish = fish_img.split('.')[0]
    path_ann_data = os.path.join(dir_images_annotations_full, fish+'.xml')
    #print('path_ann_data = ', path_ann_data)    
    if not os.path.exists(path_ann_data):
        print('remove = ', os.path.join(dir_images_train_full, fish_img))                          
        os.remove(os.path.join(dir_images_train_full, fish_img))
total_images = os.listdir(dir_images_train_full)     
size = len(total_images)
        
nbr_train = int(size* split_train_proportion)
nbr_val = int(size * split_val_proportion)
nbr_test = int(size * split_test_proportion)
    
random.shuffle(total_images,random.random)
train_images = total_images[:nbr_train]
val_images = total_images[nbr_train:(nbr_train+nbr_val)]
test_images = total_images[(nbr_train+nbr_val):(nbr_train+nbr_val+nbr_test)]
trainval_images = total_images[:(nbr_train+nbr_val)]
    
total_train_images.extend(train_images)
total_val_images.extend(val_images)
total_test_images.extend(test_images)
total_trainval_images.extend(trainval_images)

random.shuffle(total_train_images,random.random)
random.shuffle(total_val_images,random.random)
random.shuffle(total_test_images,random.random)
random.shuffle(total_trainval_images,random.random)

trainfile = open(os.path.join(dir_images_layout_full, 'train.txt'), 'w')
valfile = open(os.path.join(dir_images_layout_full, 'val.txt'), 'w')
trainvalfile = open(os.path.join(dir_images_layout_full, 'trainval.txt'), 'w')
testfile = open(os.path.join(dir_images_layout_full, 'test.txt'), 'w')

for img in total_train_images:
    trainfile.write("%s\n" % os.path.splitext(img)[0])
for img in total_val_images:
    valfile.write("%s\n" % os.path.splitext(img)[0])
for img in total_test_images:
    testfile.write("%s\n" % os.path.splitext(img)[0])
for img in total_trainval_images:
    trainvalfile.write("%s\n" % os.path.splitext(img)[0])
        
trainfile.close()
valfile.close()
testfile.close()
trainvalfile.close()

print('Finish splitting train and val images!')