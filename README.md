# AppleGAN

A PyTorch implementation of a DCGAN trained on the kmkinney/GANGothicArchitecture dataset, which is treated here as a collection of apple images.  
The model supports resumable training, checkpointing, and image generation at any time.

## Overview

This repository contains a minimal, reproducible GAN pipeline for generating synthetic apple images.  
The architecture follows the standard DCGAN design with transposed-convolution upsampling in the generator and strided-convolution downsampling in the discriminator.

The training loop is interruption-safe. Training can be stopped and resumed at any point without losing progress.  
A simple command-line interface allows both training and sample generation.

## Features

- DCGAN architecture implemented in PyTorch  
- Treats the entire dataset as unlabeled apple images  
- Safe to interrupt training at any time  
- Automatic checkpoint saving and restoration  
- GPU support  
- Command-line interface for sample generation

## File Structure

AppleGAN/
gan.py
data/
<your dataset images here>
checkpoints/
generated/
README.md

`gan.py` contains all model definitions, training logic, checkpoint handling, and generation utilities.

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- numpy  
- pillow  

Install dependencies with:

pip install torch torchvision numpy pillow

## Training

Place your dataset inside the `data/` directory.

To start training:


## Training

Place your dataset inside the `data/` directory.

To start training:

python gan.py --train

vbnet
Copy code

The script uses the GPU automatically if one is available.

To resume training:

python gan.py --train

powershell
Copy code

## Generating Images

To generate N samples using the latest generator checkpoint:

python gan.py --generate N

diff
Copy code

Images will be saved into the `generated/` directory.

## Checkpoints

The following files are saved after every epoch:

- generator.pth  
- discriminator.pth  
- optim_G.pth  
- optim_D.pth  
- training_state.json

These allow full restoration of training state.

## Notes

- All images are resized to 64×64 and normalized to [-1, 1].  
- This implementation is meant to be simple and reproducible, not optimized.  
- Any folder-structured dataset containing images can be substituted.
