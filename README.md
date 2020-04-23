# Gender-and-Age-Prediction-from-Face-Images &nbsp; ![](https://img.shields.io/badge/release-v1.0-orange)

![](https://img.shields.io/badge/license-MIT-blue) &nbsp;<br>

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

Use the [requirements.txt](https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/requirements.txt) file for installing dependencies in your virtual environment. Once inside your virtualenv run the following command to install packages for this project.

```bash
pip install -r requirements.txt
```

## Dataset &nbsp; 
![](https://img.shields.io/badge/Ubuntu-18.04-blueviolet) 
![](https://img.shields.io/badge/Discovery%20-HPC-yellow)
![](https://img.shields.io/badge/NVidia-v100:sxm2-red)

Dataset consists of ~19,000 aligned images. We used the aligned face images along with 5 helper data fold files - fold_0_data.txt, fold_1_data.txt, fold_2_data.txt, fold_3_data.txt and fold_4_data.txt. Since data was not aggregated into a single package, a small [Process](https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/src/utils/process.py) script was written to read data using <b>OpenCV</b>, covert images from BGR2RGB (more about this in this post by [Satya Mallick](https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/)) and split our datasets into train and test. We used 90% data for training and 10% for testing. We randomly shuffled our data and neatly packaged the training, testing data along with the gender and age labels into a single .h5 file. To download the h5 file, use this link.

While 2 categories of genders were used throughout the dataset, <b>Male</b> and <b>Female</b>, the ages were chunked into smaller age groups. The Adience dataset had 8 age groups - <b>(0-2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)</b>. As we can observe, not all age categories are covered and a lot of data had categories excluding the above mentioned 8. We introduced for more age categories and 

#### DAMSM Models

Download the pretrained [DAMSM Encoders and Decoders](https://drive.google.com/drive/u/0/folders/1KlhVPPRtczelfKkGDhcjkKtJMssi9DIZ), extract and place the birds directory in the <b>DAMSMencoders</b> directory. This directory contains the RNN Text Encoder and the CNN Image Encoder models.


## Attention Maps &nbsp;

Attention maps are generated along the course of training either while using the pretrained models or while training AttnGAN from scratch. These are stored in the <b>/output/birds_DAMSM_<timestamp>/</b> directory. <br>
  
<img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/attention_maps0.png">

## Run and Evaluate &nbsp;

Pre-train DAMSM models: 
  - For bird dataset, download the pretrained model of [AttnGAN](https://drive.google.com/file/d/1UbTP2Y4Bx9jHgLQEUJ-D6qqYIt3jHbGz/view?usp=sharing) and place it in the <b>models</b> directory. Use the following command to use pretrained model to generate images.
  ```python 
  python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
  ```
- Train AttnGAN models:
  - For bird dataset, to train the entire AttnGAN from scratch, use the following command. 
  ```python 
  python main.py --cfg cfg/bird_attn2.yml --gpu 2
  ```
- `*.yml` files are example configuration files for training/evaluation our models.

Run 
```python
python main.py --cfg cfg/eval_bird.yml --gpu 1
```
to generate examples from captions in files listed in <b>"/data/birds/example_filenames.txt"</b>. Results are saved to <b>/output/samples/</b> directory. 
- Input your own sentence in <b>"/data/birds/example_captions.txt"</b> if you wannt to generate images from customized sentences. 

NOTE: Use -1 value for the gpu argunment, if you don't have GPU on your system and want to use CPU. Else, mention the GPU id on which to execute your code. In either case, your system will fall back to CPU if no GPU's are detected.

## Results &nbsp;

Given the following 3 statements to the model, we get some realistic bird images as results.

- A red bird with long beak and black wings having a long tail. <br>

  <img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird1.png" width="600" height="500">
- this bird has a dark light overall body color, with long neck and short legs. <br>

  <img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird2.png" width="600" height="500">
- A bird with yellow wings and dark eyes and black beak. <br>

  <img src="https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/assets/bird3.png" width="600" height="500">

We documented our work as a presentation. Feel free to check out the presentation [here](https://github.com/nikunjlad/Text-to-Image-Metamorphosis-using-GANs/blob/master/docs/Text-to-Image-Metamorphosis-using-GANs.pptx).

## Developers &nbsp;

[![](https://img.shields.io/badge/Nikunj-Lad-yellow)](https://github.com/nikunjlad)
