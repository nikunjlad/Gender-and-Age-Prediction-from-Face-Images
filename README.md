# Gender-and-Age-Prediction-from-Face-Images &nbsp; ![](https://img.shields.io/badge/release-v1.0-orange)

![](https://img.shields.io/badge/license-MIT-blue) &nbsp;<br><br><br>
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/output/predictions/multi-person.gif">

## Architecture &nbsp;
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/assets/age_gender.png">

## Description &nbsp; 
[![](https://img.shields.io/badge/GitHub-taoxugit-red)](https://github.com/davidstap/AttnGAN) &nbsp;
[![](https://img.shields.io/badge/arXiv-AttnGAN-lightgrey)](https://arxiv.org/abs/1711.10485)

Convolutional Neural Networks are heavily used for image classification tasks. Recently, various techniques were tried for trying to predict age and gender in humans. While it's fairly easy to predict whether a person is a Male or a Female since difference between both is pretty obvious, it becomes difficult to predict an age of a person seeing just their face. <b>Looks can be deceiving</b> as is rightly said and sometimes some people seem to be old but in reality they are not, and likewise, some people seem to be young but they are actually quite old. Various works have been done on this in literature right from using localizing facial features based on their size and ratios to applying constraints for age estimation like aging, etc. However, such constrained techniques do not generalize well on in-the-wild images of human faces. For this project, we have explored the [Adience Dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html) which represents highly unconstrained and complex in-the-wild human faces. 

The problem of Age and Gender classification was initially solved by [Tal Hassner, et al.](https://scholar.google.com/citations?hl=en&user=ehe5pyIAAAAJ) in their 2015 CVPR paper on [Age and Gender Classification using Convolutional Neural Networks](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf). This project is an implementation of their method along with a few customizations as part of my research work.

## Dependencies &nbsp;
![](https://img.shields.io/badge/python-3.6-yellowgreen) &nbsp; ![](https://img.shields.io/badge/install%20with-pip-orange)

You can either create a virtualenv or a docker container for running the project. We went about creating a virtualenv <br>
Necessary libraries include the following and it is </b>recommended</b> to use pip for installation, although conda can also be used.

torch>=1.5.0 </br>
torchvision>=0.6.0 </br>
opencv-python>=4.2.0.34 </br>
opencv-contrib-python>=4.2.0.34 </br>
numpy>=1.18.3 </br>
Pillow>=7.1.1 </br>
h5py>=2.10.0 </br>
matplotlib>=3.2.1 </br>
tqdm>=4.45.0 </br>

Use the [requirements.txt](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/configs/requirements.txt) file for installing dependencies in your virtual environment. Once inside your virtualenv run the following command to install packages for this project.

```bash
pip install -r requirements.txt
```

## Dataset &nbsp;  

Dataset consists of ~19,000 aligned images. We used the aligned face images along with 5 helper data fold files - <b>fold_0_data.txt</b>, <b>fold_1_data.txt</b>, <b>fold_2_data.txt</b>, <b>fold_3_data.txt</b> and <b>fold_4_data.txt</b>. Since data was not aggregated into a single package, a small [Process](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/utils/process.py) script was written to read data using <b>OpenCV</b>, covert images from BGR2RGB (more about this in this post by [Satya Mallick](https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/)) and split our datasets into train and test. We used 90% data for training and 10% for testing. We randomly shuffled our data and neatly packaged the training, testing data along with the gender and age labels into a single .h5 file. To download the h5 file, use this [link](https://drive.google.com/drive/folders/1gGPHYopXyW9SYRUMI4DpswCuw2ZWFnOk?usp=sharing).

While 2 categories of genders were used throughout the dataset, <b>Male</b> and <b>Female</b>, the ages were chunked into smaller age groups. The Adience dataset had 8 age groups - <b>(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)</b>. As we can observe, not all age categories are covered and a lot of data had categories excluding the above mentioned 8. We introduced four more age categories - <b>(21-24), (33-37), (44-47) and (54-59) </b>, since ~1200 images were mislabelled. Our [Process](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/utils/process.py) script takes care of that and reassigns correct labels while removing unwanted samples. We now have 12 classes as compared to the original 4 classes. While the paper published by Tal Hassner performed well with 8 classes, due to imbalanced distribution of classes and lack of samples for the newly added classes, our model has some bias in it. 

(Download the <b>aligned</b> images and place in the adience directory downloaded from above drive link). Command to process script from raw images and export an h5 file : 

```bash
python process.py --path=data/adience --save=data/adience/adience.h5
```

## System Specifications &nbsp;
![](https://img.shields.io/badge/Discovery%20-HPC-yellow)
![](https://img.shields.io/badge/NVidia-v100:sxm2-red)

### Cluster Information
<b>Cluster</b>: Discovery High-Performance Computing Cluster </br>
<b>Reservation</b>: CSYE7374_GPU </br>
<b>Reservation memory</b>: 95 GB </br>
<b>Time reserved</b>: 10 hrs. </br>

If you have access to Discovery, use following command to access reservation (You are supposed to use your reservation in place of CSYE7374_GPU)
```bash
srun -p reservation --reservation=CSYE7374_GPU --gres=gpu:v100-sxm2:4 --pty --mem=95000 --time=10:00:00 /bin/bash
```

### GPU Information
<b>Model</b>: Nvidia V100-SXM2 </br>
<b>GPU count</b>: 0 – 4 </br>
<b>GPU Memory</b>: 32.4805 GB / GPU </br>
<b>Clock</b>: 1.290Ghz with max boost of 1.530GHz </br>

Use 
```bash
watch -n 0.1 nvidia-smi
```
to keep a check on GPU usage during training. For getting GPU specific information use, 
```bash
nvidia-smi -q -d CLOCK
```

## Train &nbsp;

We have maintained a [config.yaml](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/configs/config.yaml) file which holds the training configuration. If you are using NVidia GPU, mention the device list under the <b>[GPU][DEVICES]</b> tag. (Note, always provide a list and not a single integer. For single GPU use [0]; for any more GPU for DataParallelism, populate the DEVICE list with more GPU ids). 

We have curated a common script to train both Age and Gender models. To train the model for Gender classification use -
```bash
python main.py --age-gender=gender
```
and to train on age use
```bash
python main.py --age-gender=age
```

Model outputs are save in the <b>output</b> folder. For every run of training, output of the run will be saved in a folder named - <BATCH_SIZE>_output_<NUM_GPUS>, where <BATCH_SIZE> is batch size during current run and <NUM_GPUS> is number of GPUs used for training, example [64_output_3](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/tree/master/src/output/64_output_3)

Every training outputs a [statistics](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/output/64_output_3/age_stats_64_3.json) file giving runtime parameter dictionary, logs along with accuracy and loss curves and best model. 

For our case we will have 2 statistics file, 1 each for gender and age classification and 2 sets of accuracy and loss curves along with 2 models giving best parameters for corresponding runs.

NOTE: If you don't have GPU, set <b>[GPU][STATUS]</b> flag as False. However, our implementation keeps a default check of GPU and automatically switches to CPU in absence of GPU.

## Observations &nbsp;

| Model | Batch Size | # GPUs | # Epochs | Train Acc | Val Acc | Test Acc | Train Loss | Valid Loss | Test Loss |
|-------|------------|--------|----------|-----------|---------|----------|------------|------------|-----------|
| Age   |  64        |   3    |  60      |  0.976946 |0.702474 | 0.630011 |  0.067266  |  1.616894  |  1.208523 |
| Gender|  64        |   3    |  60      |  0.999731 |0.934218 | 0.886597 |  0.001065  |  0.367484  |  0.886597 |

Loss and Accuracy curves for both models are present in the [64_output_3](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/tree/master/src/output/64_output_3) directory in the output folder.

Note: More experimentation is required as we yet not conclude the above results are state-of-the art.

## Test &nbsp;

### Static Images
To generate sample images, use
```bash
python sample.py --input=../images/woman.jpg --output=woman.png
``` 
We get really good predictions both for Age and Gender as can be seen in below images
</br>
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/assets/true.png">
</br></br>
Here is another case of a picture of me and my brother back during our younger days to the times when we were older
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/assets/transition.png">
</br></br>
However, its not perfect always. Here is a case where it gave pretty weird results
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/output/predictions/friends.png">

## Developers &nbsp;

[![](https://img.shields.io/badge/Nikunj-Lad-yellow)](https://github.com/nikunjlad)
