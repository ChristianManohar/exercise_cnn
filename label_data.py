"""
This file creates a csv file of images sorted into train, validate, and test sets
"""

import os
import pandas as pd
import random
from PIL import Image


"""def resize(height, width, path, dirs):
    for item in dirs:
        if os.path.isfile(path + item):
            img = Image.open(path + item)
            new_image = img.resize((width, height))
            new_file_name = 'resized-' + item
            new_image.save(path + new_file_name)
"""

#Define train, validation, and test percent
train_percent = 0.6
validation_percent = 0.2

#Defines the base directory which contains subdirectories
base_dir = 'archive/'
exercise_dirs = []

#Iterate through subdirectories and create 2D vector of image files
im_files = []
for subdir, dirs, files in os.walk(base_dir):
    exercise_dirs = dirs
    for dir in dirs:
        im_files.append(sorted(os.listdir(base_dir+str(dir))))
    break

#Initialize dataframe of images, with four columns defined
df = pd.DataFrame(columns=['image_files', 'exercise_label', 'numeric_labels', 'partition'])


numeric_label = float(0)
for f, dir_ in zip(im_files, exercise_dirs):

    files = sorted(os.listdir(base_dir + dir_))


    for im in files:

        res = random.random()
        if res <= train_percent: 
            partition = 'train'
        elif res <= train_percent + validation_percent:
            partition = 'validation'
        elif res <= 1:
            partition = 'test'
        else:
            assert False


        #insert row of data into DF individually
        data = [im, dir_, numeric_label, partition]

        df.loc[-1] = data
        df.index = df.index + 1
        
    #increment numeric label to differentiate exercises
    numeric_label += 1

#df.sort_index(ascending=False)

df.to_csv('ims.csv')