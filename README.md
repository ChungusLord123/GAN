AppleGAN

A PyTorch implementation of a DCGAN trained on the kmkinney/GANGothicArchitecture dataset, which in this project is treated as a collection of apple images.
The model supports resumable training, checkpointing, and image generation at any time.

Overview

This repository contains a minimal, reproducible GAN pipeline for generating synthetic apple images.
The architecture follows the standard DCGAN design with transposed-convolution upsampling in the generator and strided-convolution downsampling in the discriminator.

The training loop is designed to be interruption-safe. Training can be stopped at any time and resumed without loss of progress.
A command-line interface makes it easy to either train the model or generate new samples from a saved checkpoint.

Features

DCGAN architecture implemented in PyTorch

Works with any folder-structured dataset; here it is assumed all images are apples

Training can be interrupted safely (Ctrl-C)

Automatic checkpoint saving and resumption

GPU support when available

Command-line generation of image samples

File Structure

This repository uses the following structure:

AppleGAN/
    gan.py
    data/
        <your dataset images here>
    checkpoints/
    generated/
    README.md


gan.py contains all training, model, and inference logic in a single script.

Requirements

Python 3.8+

PyTorch

torchvision

numpy

pillow

Install dependencies:

pip install torch torchvision numpy pillow

Training

Place your dataset inside the data/ directory.
All images will be loaded recursively.

To start training:

python gan.py --train


Training will automatically detect and use a GPU if available.
Checkpoints are written to checkpoints/ at the end of each epoch.

To resume training after interruption:

python gan.py --train


The script will load the latest checkpoint and continue from the correct epoch.

Generating Images

To generate new apple images using the latest saved generator:

python gan.py --generate 64


This outputs 64 generated samples to the generated/ directory.

Checkpoints

The following files are saved after each epoch:

generator.pth

discriminator.pth

optim_G.pth

optim_D.pth

training_state.json (epoch and batch index)

These files allow full restoration of the training session.

Notes

All images are automatically resized to 64×64 and normalized to [-1, 1].

This implementation is intended as a clean, minimal reference rather than an optimized production model.
