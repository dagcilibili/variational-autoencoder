# Variational Autoencoder

This is a Variational Autoencoder (VAE) implementation using [Tensorflow](https://www.tensorflow.org/) on Python. It uses of convolutional layers and fully connected layers in encoder and decoder. The loading functions are designed to work with CIFAR-10 dataset. New loading functions need to be written to handle other datasets. In deep learning literature, it is reported that VAEs produce blurry images when trained on CIFAR-10 dataset. We have written this code to experiment on ways to generate realistic-looking images.

*Thanks:* While writing this code, we got inspired by [J. H. Metzen's implementation](https://jmetzen.github.io/2015-11-27/vae.html), and got help from the data loading scripts distributed in [UC Berkeley's CS 294-129](https://bcourses.berkeley.edu/courses/1453965/pages/cs294-129-designing-visualizing-and-understanding-deep-neural-networks) and [Stanford's CS231n](http://cs231n.stanford.edu/) courses.

*Authors:* [Orhan Ocal](https://github.com/dagcilibili) and [Raaz Dwivedi](https://github.com/rzrsk).