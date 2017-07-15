---
layout: post
title:  "Gap between DNN and HMM"
date:  2017-07-15 09:53:41 -0700
categories: speech basics
---

# This is a blog explaining the bridge between DNN and HMM in speech recognition

This is my first blog. I have been planning to write blog long before. However it didn't happen until recently. Hopefully I could keep updating this blog regularly.

In this first blog, I will write something about a question in speech recognition which has been confusing me a lot recently. The issue is how people apply Deep Neural Network (DNN) on top of HMM to perform speech recognition. 

Before going into the question, let's look at some technical background in speech recognition

## Technical background

Automatic speech recognition has been a hot topic for decades. The original most successful modeling framework is GMM/HMM model: Gaussian mixture models (GMM) for acoustic distributions and HMM models for word decoding. Usually the recognition task is performed as follows

* Raw speech audio is converted into acoustic features by some type of feature engineering techniques. For example the spectrogram has been used to represent the audio
	
* Once we have acoustic features, a HMM framework is used to estimate a *hidden* state sequence for the inputs
	
* The hidden state sequence are further used to perform decoding to generate word sequence, known as script

The above process can be formulated as follows. Let's use $(y_1, \dots, y_T)$ to denote the input audio. The first step applies a feature transformation $f(\cdot)$ to the audio frame to get its feature space representation: $(f_1, \ldots, f_T)$, where $f_i = f(y_i)$. The second step uses a -/HMM framework (- could be GMM or DNN) to estimate a latent state representation of the feature sequence. After this point we have a state sequence $(x_1, \ldots, x_T)$. The last step is the decoding step which produces the words sequence $(w_1, \ldots, w_N)$. Note that the sequence in the last step has a different length.

In this blog I am going to try to understand how the last two steps are performed.




Some references [^ref1], [^ref2], [^ref3], [^ref4]





[^ref1]: A tutorial on hidden Markov models and selected applications in speech recognition

[^ref2]: An introduction to hybrid HMM/Connectionist continuous speech recognition

[^ref3]: Deep neural networks for acoustic modeling in speech recognition

[^ref4]: From HMMs to segment models: a unified view of stochastic modeling for speech recognition


