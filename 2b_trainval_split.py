#!/usr/bin/env python
# coding: utf-8

# ### Split dataprocessedv0 dataset into train val and store in dataprocessedv2

# In[ ]:


# Run this
#!pip install split-folders tqdm


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import skimage.draw
import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
import warnings
import random
warnings.filterwarnings('ignore')


get_ipython().run_line_magic('matplotlib', 'inline')

# Helpers

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    
# Root directory of the project
ROOT_DIR = os.path.abspath(".")
print(ROOT_DIR)

# Training file directory
IMAGES_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv0", "images/")
MASKS_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv0", "images/")
print(IMAGES_FOLDER, MASKS_FOLDER)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Split data_processed0 into train val and store in data_processedv2

# In[ ]:


IN_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv0") # raw images
OUT_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv2")
IN_FOLDER, OUT_FOLDER


# In[ ]:





# In[ ]:





# In[ ]:


import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio(IN_FOLDER, output=OUT_FOLDER, seed=1337, ratio=(.8, .2)) # default values # .8 .2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# NOT USING THIS NOW!


# ### Split data_processedv1(patches) into train val and store in data_processedv3

# In[ ]:


#IN_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv1") # raw images
#OUT_FOLDER = os.path.join(ROOT_DIR, "dataset", "data_processedv3")
#IN_FOLDER, OUT_FOLDER


# In[ ]:


#import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#split_folders.ratio(IN_FOLDER, output=OUT_FOLDER, seed=1337, ratio=(.8, .2)) # default values


# In[ ]:




