# Variational Autoencoder

This is a Variational Autoencoder (VAE) implementation using [https://www.tensorflow.org/ Tensorflow] on Python. It uses of convolutional layers and fully connected layers in encoder and decoder. The loading functions are designed to work with CIFAR-10 dataset. New loading functions need to be written to handle other datasets. In deep learning literature, it is reported that VAEs produce blurry images when trained on CIFAR-10 dataset. We have written this code to experiment on ways to generate realistic-looking images.

*Thanks:* While writing this code, we got inspired by [https://jmetzen.github.io/2015-11-27/vae.html J. H. Metzen's implementation], and got help from the data loading scripts distributed in [https://bcourses.berkeley.edu/courses/1453965/pages/cs294-129-designing-visualizing-and-understanding-deep-neural-networks UC Berkeley's CS 294-129] and [http://cs231n.stanford.edu/ Stanford's CS231n] courses.

*Authors:* [https://github.com/dagcilibili Orhan Ocal] and [https://github.com/rzrsk Raaz Dwivedi].