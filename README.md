# Gender-and-Age-Prediction-from-Face-Images &nbsp; ![](https://img.shields.io/badge/release-v1.0-orange)

![](https://img.shields.io/badge/license-MIT-blue) &nbsp;<br>

## Architecture &nbsp;
<img src="https://github.com/nikunjlad/Gender-and-Age-Prediction-from-Face-Images/blob/master/assets/age_gender.png">

## Description &nbsp; 
[![](https://img.shields.io/badge/GitHub-taoxugit-red)](https://github.com/davidstap/AttnGAN) &nbsp;
[![](https://img.shields.io/badge/arXiv-AttnGAN-lightgrey)](https://arxiv.org/abs/1711.10485)

Generative Adversarial Networks are used heavily for generating synthetic data and for upsampling an unbalanced dataset. However, it has more to it and one of it's application can be observed in this repository. Text-to-Image Metamorphosis is translation of a text to an Image. Essentially, it is inverse of Image Captioning. In Image Captioning, given an image, we develop a model to generate a caption for it based on the underlying scene. Text-to-Image Metamorphosis, generates an image from a corresponding text by understanding the language semantics. Various works have been done in this domain, the most notable being developing an Attentional GAN model to develop images given a local word feature vector and a global sentence vector. Currently we have only worked on [AttnGAN](https://arxiv.org/abs/1711.10485). Further up, we intend to implement [MirrorGAN](https://arxiv.org/abs/1903.05854), an extension of AttnGAN to generate images from sentences and reconstruct the sentences from the generated image, so as to see how similar are the input and output sentences. Concretely, we would like a input and output sentences to be as close to each other (like a mirror) so as to conclude, the underlying generation is close to ground truth.

## Dependencies &nbsp;
![](https://img.shields.io/badge/python-3.6-yellowgreen) &nbsp; ![](https://img.shields.io/badge/install%20with-pip-orange)

You can either create a virtualenv or a docker container for running the project. We went about creating a virtualenv <br>
Necessary libraries include but not limited to the following and are installed using pip.

python==3.6.5 <br>
numpy=>=1.18.1 <br>
pandas>=1.0.1 <br>
nltk>=3.4.5 <br>
torch>=1.4.0 <br>
torchvision>=0.5.0 <br>

For an entire list of libraries for this project refer the [requirements.txt](https://github.com/nikunjlad/Text-to-Image-Metamorphosis/blob/master/requirements.txt) file. <br>

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
