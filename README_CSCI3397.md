# CSCI3397 Biomedical Imaging Final Project: Multi-Organ Nuclei Segmentation and Classification Challenge

The instructions for running the project are adapted from the previous project. We have added some steps which we found necessary to setup the environment and run all of the code locally. 

### Our Contributions

All group members (Callie, Emma and Austin) attempted (for a substantial amount of time) to set up this project in Colab. However, as addressed above, Callie discovered a fix which worked on her local machine (editing the efficientnet distribution file, discussed below), and we were unable to find a similar solution for the Colab environment. Additionally, we did not have write-access to this file on the GPUs on the cslab server. Due to the fact that Callie was the only student in the group with sufficient memory/ hardware to run the project locally, all coding was done on her computer. However, we collaborated on all deicsions of how to modeify and adapt the code. Emma was able to sucessfully upload all of the processed data for training and testing as well. 

Code - related contributions: 

Our main contribution was our discussion and analysis of the sigma value for median filtering used within post-processing to generate the segmentation masks. This modification was made within the 4b_inference, and 3b_validate_and_compute_PQ files.

Additionally, several modeifications had to be made to the original code in order for it to run (original code had errors). This included re-writing the instance segmentation function in 4b_inference.py (found starting at line 341), and in 3b_validate_and_compute_PQ.py (found starting at line 411).

Callie also created test_change_sigma.py and 3b_pq_change_sigma to test the effect of changing sigma within the training of the model, as opposed to solely in post-processing, and to do so during computation of the PQ metric, respectively.

### Report Breakdown

Again, all group members collaborated to understand and write all sections of the report. The sections were written by group members as follows:

Callie: Abstract, Introduction, Model, U-net Architecture, Experiments, Results, Further Work

Emma: Dataset, ... (add sections you worked on)

Austin: Realted Work, Loss Function, ... (add sections you worked on)


### Requirements
* Python: 3.6
* Tensorflow: 2.0.0
* Keras: 2.3.1
* [segmentation_models](https://segmentation-models.readthedocs.io/en/latest/install.html).
* OpenCV

### Environment installations

Run this command to make environment

```
conda env create -f environment.yml
```

*OR* you can make you own environment by:

```
conda create -n yourenvname python=3.6 anaconda
```

Then install the packages

```
conda install -c anaconda tensorflow-gpu=2.0.0
conda install -c conda-forge keras
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
```

The run `conda activate yourenvname`.

NOTE: `segmentation_models` does not have conda distribution. You can install by running `pip install -U --pre segmentation-models --user` inside your environment. More details at [segmentation_models](https://segmentation-models.readthedocs.io/en/latest/install.html).

### Additional Step Needed
* Within efficientnet python distribution package (located at a path akin to: /usr/local/lib/python3.8/dist-packages/efficientnet/): open the keras.py file and change both instances of 'init_keras_custom_objects' to 'init_tfkeras_custom_objects'. Save this file, and any errors concerning efficientnet (& versioning) should be fixed.

### Dataset versions

* `MoNuSAC_images_and_annotations` : original dataset which has patient's whole slide images (WSI) and ground truth. (Given by challenge organizers)
* `MoNuSAC_masks` : contains binary masks generated from `get_mask.ipynb`.
* `Testing Images`: contains test images, without annotations. (Given by challenge organizers)
* `data_processedv0` : contains all raw images and the ground truth masks.
* `data_processedv4` : sliding window patchwise data from original images and masks in `data_processedv0`.
* `data_processedv5` : 80/20 trainval split from `data_processedv4`.

### Getting started

0. Clone repository (obviously!)
1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `0_get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `1_data_process_MoNuSAC.ipynb` to get raw images and their ground truth masks in `data_processedv0`. 
5. Run `2_extract_patches.ipynb` to get patches of images and gt masks from the previous raw version to get `data_processedv4` and the 80/20 split `data_processedv5`.
6. Run `3_train.iynb`. It trains *PatchEUNet* on `data_processedv5`.
7. Put `Testing Images` in `dataset` folder.
8. Run `3c_load_test_data.ipynb`. Outputs `test_images` folder with all test images.
8. Run `4_inference.ipynb` to get final prediction masks from `test_images`. (For visualization)
9. Run `4b_inference.ipynb` to get final prediction masks according to MoNuSAC submission format.




