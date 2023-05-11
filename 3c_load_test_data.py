#!/usr/bin/env python
# coding: utf-8

# ### Save testing images in one folder named `test_images`

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
        
        
        
def read_nuclei(path):
    "Read raw data"

    # Load 4-channel image
    if len(path) == 0:
        return None
    
    img = skimage.io.imread(path)
    
    # input image
    if len(img.shape) > 2:
        img = img[:,:,:3]
    # mask
    else:
        # do nothing
        pass
        
    return img



def save_nuclei(path, img):
    "save image"
    skimage.io.imsave(path, img)
    
    
    
label_map = {'Epithelial':1,
             'Lymphocyte':2,
             'Macrophage':4,
             'Neutrophil':3,
            }


# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
# Training file directory
IMAGES_FOLDER = os.path.join(ROOT_DIR, "dataset", "Testing images/")
print(IMAGES_FOLDER)
IMAGES_SUB_FOLDER = [os.path.join(IMAGES_FOLDER, i) for i in sorted(next(os.walk(IMAGES_FOLDER))[1])]
print(IMAGES_SUB_FOLDER[:5])
IMAGES_DEST =  os.path.join(ROOT_DIR, "dataset", "test_images")
print(IMAGES_DEST)
# Create folders
create_directory(IMAGES_DEST)


# In[ ]:


# STORE IMAGES in test folder together

raw_ct = 0
for ct in tqdm(range(len(IMAGES_SUB_FOLDER[:]))):
    
    #print(ct)
    
    # Read all raw images in image sub folder
    all_imgs = sorted(glob(IMAGES_SUB_FOLDER[ct] + '/*.tif'))
    
    paths = [s.split('.')[0][-25:] for s in all_imgs]
    
    
    # Iterate over the individual raw images
    for i in range(len(all_imgs)):
        
        # Read test image
        #print(all_imgs[i])
        img = read_nuclei(all_imgs[i])

        # Save it
        save_nuclei(IMAGES_DEST+ "/{}.png".format(paths[i]), img)
        raw_ct+=1


# In[ ]:


image_fns = sorted(next(os.walk(IMAGES_DEST))[2])
image_fns[:3]


# ### Get data stats

# In[ ]:


w = []
h = []

for i in range(len(IMAGES_DEST)):
    image = skimage.io.imread(os.path.join(IMAGES_DEST, image_fns[i]))
    w.append(image.shape[1])
    h.append(image.shape[0])
    
w = np.array(w)
h = np.array(h)

print(w.shape, h.shape)


# In[ ]:


# Mean of hieght
np.mean(h), np.std(h)
#(607.6363636363636, 534.3626403529337)


# In[ ]:


# Mean of hieght
np.mean(w), np.std(w)
#(569.4727272727273, 401.6142814621763)


# In[ ]:


np.max(h), np.min(h)
#(2500, 82)


# In[ ]:


np.max(w), np.min(w)
#(1987, 35)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def process(image, mask):
    f, axarr = plt.subplots(1,2, figsize=(16, 16))
    axarr[0].imshow(image)
    axarr[1].imshow(mask, cmap='gray')


def sliding_window(image, step, window):
    x_loc = []
    y_loc = []
    cells = []
    
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            cells.append(image[y:y + window[1], x:x + window[0]])
            x_loc.append(x)
            y_loc.append(y)
    return x_loc, y_loc, cells


# ### Process images with sliding window

# In[ ]:


image_fns = sorted(next(os.walk(IMAGES_DEST))[2])

image = skimage.io.imread(os.path.join(IMAGES_DEST, image_fns[random.randrange(len(image_fns))]))

# Get locations
x_pos, y_pos, cells = sliding_window(image, 8, (64, 64))

# Array for storing predictions
pred = np.zeros((image.shape[0], image.shape[1]))

for (x, y, cell) in tqdm(zip(x_pos, y_pos, cells)):
    
    # Get patch
    patch = image[y:y + 64, x:x + 64]
    
    # Get size
    raw_dim = (patch.shape[1], patch.shape[0]) # W, H
    #print(raw_dim)
    #print(patch.shape)
    
    
    if raw_dim != (64, 64):
        
        
        # Resize to 64x64
        patch = cv2.resize(patch, (64, 64), interpolation = cv2.INTER_AREA)
        
        
        # Do stuffffff
        processed = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        
        # Resize back to original shape
        processed = cv2.resize(processed, raw_dim, interpolation = cv2.INTER_AREA)
    
    else:
        
        # Do stuffffff
        processed = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    
    # Add in dummy image
    pred[y:y + 64, x:x + 64] = processed    

print(image.shape, pred.shape)
process(image, pred)


# In[ ]:




