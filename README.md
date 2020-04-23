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

torch>=1.5.0
torchvision>=0.6.0
opencv-python>=4.2.0.34
opencv-contrib-python>=4.2.0.34
numpy>=1.18.3
Pillow>=7.1.1
h5py>=2.10.0
matplotlib>=3.2.1
tqdm>=4.45.0

Use the [requirements.txt](https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/requirements.txt) file for installing dependencies in your virtual environment. Once inside your virtualenv run the following command to install packages for this project.

```bash
pip install -r requirements.txt
```

## Project Structure &nbsp; 
![](https://img.shields.io/badge/Ubuntu-18.04-blueviolet) 
![](https://img.shields.io/badge/Google%20-Cloud-yellow)
![](https://img.shields.io/badge/NVidia-TeslaT4-red)

Developed and configured this project on MAC using PyCharm IDE and trained the model on Google Cloud using NVidia Tesla T4 GPUs <br>

#### Birds data directory
The data lies in the data directory. Download the [bird.zip](https://drive.google.com/file/d/1v0CIe8psyLI0Yle2YylrL91QnEdjdSzP/view?usp=sharing) file and place the extracted bird directory in the data directory in this repository. Following are the files in the bird.zip file.

1. <b>/birds/CUB_200_2011/</b> : It contains an images directory which holds 200 bird class directories. The bird dataset has 200 classes of birds each class has ~60 images of birds.
2. <b>/birds/CUB_200_2011/bounding_boxes.txt</b> : file which contains bounding box information of the birds in the true images. The annotations are in [top-left-x, top-left-y, width, height] format.
3. <b>/birds/CUB_200_2011/classes.txt</b> : file containing names of all the 200 classes.
4. <b>/birds/CUB_200_2011/image_class_labels.txt</b> : file containing class labels. First 60 lines belong to image 1, next 60 lines belong to image 2 and so and so forth.
5. <b>/birds/CUB_200_2011/images.txt</b> : file containing all the image names in the dataset. this is about ~12000 images, 60 images per class.

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
