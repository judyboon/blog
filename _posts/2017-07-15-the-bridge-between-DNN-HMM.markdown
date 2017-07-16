---
layout: post
title:  "Gap between DNN and HMM"
date:  2017-07-15 09:53:41 -0700
categories: speech basics
---

This is my first blog. I have been planning to write blog long
before. However it didn't happen until recently. Hopefully I could
keep updating this blog regularly.

In this first blog, I will write something about a question in speech
recognition which has been confusing me a lot recently. The issue is
how people apply Deep Neural Network (DNN) on top of HMM to perform
speech recognition.

Before going into the question, let's look at some technical
background in speech recognition

# Technical background

Automatic speech recognition has been a hot topic for decades. The
original most successful modeling framework is GMM/HMM model: Gaussian
mixture models (GMM) for acoustic distributions and HMM models for
word decoding. Usually the recognition task is performed as follows

1. Raw speech audio is converted into acoustic features by some type
   of feature engineering techniques. For example the spectrogram has
   been used to represent the audio
	
2. Once we have acoustic features, a HMM framework is used to estimate
   a *hidden* state sequence for the inputs
	
3. The hidden state sequence are further used to perform decoding to
   generate word sequence, known as script

The above process can be formulated as follows. Let's use $(y_1,
\dots, y_T)$ to denote the input audio. The first step applies a
feature transformation $\phi(\cdot)$ to the audio frame to get its
feature space representation: $(f_1, \ldots, f_T)$, where $f_t =
\phi(y_t)$. The second step uses a -/HMM framework (- could be GMM or
DNN) to estimate a latent state representation of the feature
sequence. After this point we have a state sequence $(x_1, \ldots,
x_T)$. The last step is the decoding step which produces the words
sequence $(w_1, \ldots, w_N)$. Note that the sequence in the last step
has a different length.

In this blog I am going to try to understand how the last two steps
are performed.

# Hidden Markov model

The most important component in speech recognition is the HMM part
(which might not be the case with recent emergence of temporary neural
network such as LSTM). A HMM consists of following ingredients

* Emission distribution

* State transition distribution

The emission distribution describes the distribution of observation
given latent state $g(f_t \| x_t)$ (I am using $f_t$ to denote the
observation to make consistent with previous speech recognition
process). The transition distribution describes the evolving law of
the latent state $h(x_t \| x_{t-1})$. These two distributions are
assumed to follow certain parametric form with some unknown parameters
$\theta = (\theta_g, \theta_h)$.

The three classic problems in HMM [^ref1] are 

1. Given observation sequence $(f_1, \ldots, f_T)$ and parameters
   $\theta$, how to calculate the likelihood of $p(f_1, \ldots, f_T \|
   \theta)$ with latent states marginalized out

2. Given observation and parameters, how to find the latent state
   sequence $(x_1, \ldots, x_T)$ that best explains the observation

3. Given observation only, how to perform parameter estimation of
   $\theta$

Above problems can be solved efficiently with complexity $O(TK^2)$
where $K$ is the number of latent states in the model [^ref1]. The key
additional variables that are introduced to solve these problems are

1. Forward pass variable $\alpha_t(s) = p(f_1, \ldots, f_t, x_t = s \|
   \theta)$ with recursion

	$$\alpha_{t+1}(s') = \sum_s \big( \alpha_t(s) h(x_{t+1} = s' | x_t
    = s) \big) g(f_{t+1} | x_{t+1} = s')$$

2. Backward pass variable $\beta_t(s) = p(f_{t+1}, \ldots, f_T \| x_t
   = s, \theta)$ with recursion

	$$\beta_{t-1}(s') = \sum_s \big( h(x_t = s | x_{t-1} = s')
    \beta_t(s) g(f_t | x_t = s) \big)$$

3. Conditional state variable $\gamma_t(s) = p(x_t = s \| f_1, \ldots,
   f_T, \theta)$ with
   
   $$\gamma_t(s) = \frac{\alpha_t(s) \beta_t(s)}{ p(f_1, \ldots, f_T |
   \theta )} = \frac{\alpha_t(s) \beta_t(s)}{ \sum_{s'} \alpha_t(s')
   \beta_t(s') }$$
   
4. Viterbi variable $\delta_t(s) = \max_{x_1, \cdots, x_{t-1}} p(f_1,
   \ldots, f_t, x_1, \cdots, x_{t-1}, x_t = s \| \theta)$ (Note the
   difference of $\delta_t(s)$ and $\alpha_t(s)$ is just substituting
   summation with maximization)

5. Baum-Welch variable $\xi_t(s, s') = p(x_t = s, x_{t+1} = s' \| y_1,
   \ldots, y_T, \theta)$ with

	$$\xi_t(s, s') = \frac{\alpha_t(s) h(x_{t+1} = s' | x_t = s)
    g(y_{t+1} | x_{t+1} = s') \beta_{t+1}(s')}{ p(y_1, \ldots, y_T |
    \theta)} $$

For the last variable, it is used (combined with $\gamma_t(s)$) for
the EM algorithm (the famous Baum-Welch algorithm) for estimating
$\theta$. Note that $\xi_t(s, s')$ is *expected number of transitions
from $s$ to $s'$* and $\gamma_t(s)$ is *expected number of transitions
from $s$*. Then the re-estimation of model transition distribution
($h(\cdot | \cdot))$ can be derived accordingly [^ref1].

The estimation of emission parameter can be derived using a process
similar to EM estimation of standard GMM. The conditional state
variable $\gamma_t(s)$ is analogous to the expectation of component
indicator in GMM. By plugging $\gamma_t(s)$ into the *E* step of
GMM, the emission distribution parameter can be updated in the *M*
step. The emission distribution is usually assumed to be Gaussian or
mixtures of Gaussian to enable closed form update in Baum-Welch
algorithm.

In speech recognition, it is usually the case where an individual HMM
is fitted to some type of **unit**. The definition of unit could be a
word or some other fine-grained linguistic units such as phones or
syllables [^ref1]. Assuming the unit is in word level. Then each HMM
is trained to fit a particular word. When unlabeled new word audio
signal comes, the task reduces to finding which model gives the
maximum likelihood of observing the audio features.

# Modeling emission distribution using DNN

Once the feature space representations of audio signal $(f_1, \cdots,
f_T)$ are obtained, the emission distribution in the HMM framework is
often assumed to be multinomial (for discrete features), Gaussian or
mixture of Gaussian (for continuous features). Recently there has been
an emergence where the emission distribution is replaced with a deep
neural network, leading to DNN/HMM framework. 


In the DNN/HMM framework, a DNN is trained to estimate $p(x_t \| f_t)$
by training on a labeled corpus. Suppose we have feature sequence
$(f_1, \cdots, f_T)$ as well as their corresponding labels $(c_1,
\cdots, c_T)$. We could train a neural network to approximate $p(c_t |
f_t)$. For example we could fit a convolutional network with certain
loss function using back propagation algorithms. Once the mapping from
$f_t$ to $c_t$ is trained, we use the output of from the trained DNN
as the latent state estimate for a testing sequence. According to
Bayes's law, we have

$$ p(f_t | x_t) = \frac{p(x_t | f_t) * p(f_t)}{p(x_t)}. $$

This equation suggests that the emission distribution in HMM could be
replaced by the DNN model, by dividing the "posterior" state
distribution $p(x_t \| f_t)$ with the "prior" state distribution
$p(x_t)$ [^ref2]. The factor $p(f_t)$ is common across all different
states, therefore it is canceled in the Forward-Backward pass
algorithms in HMM. 


There are several advantages of using a pre-trained model as the
emission distribution. First, we removed the parametric assumption of
the emission distribution, leading to a more general training
framework. Second, the feature mapping $f_t = \phi(y_t)$ mentioned
previously could be simplified. This greatly reduced our labor in
choosing the "correct"" mapping function, since the high level
features are automatically learned in a DNN model [^ref4].


This idea leads following iterative optimization
procedure

1. Train a DNN/HMM based on a labeled data set, e.g. TIMIT

2. Perform a Viterbi alignment to a test data set to find best state
   sequence using the estimated $p(f_t | x_t)$

3. Use the best state sequence to train another DNN

4. Repeat step 2 and 3 until convergence


# Conclusion

In this blog I have learned how the DNN is used in an HMM learning
framework in speech recognition tasks. There is still a missing
connection between how HMM models in unit level (e.g. phone) is used
to estimate the text script given a audio input. In [^ref1] a method
called **level building HMM** was mentioned. Probably will be the
topic for my next blog.


# Some references


[^ref1]: Lawrance R Rabiner, *A tutorial on hidden Markov models and
    selected applications in speech recognition*, 1989

[^ref2]: Nelson Morgan and Herve Bourlard, *Connectionist speech
    recognition: a hybrid approach*, 2012

[^ref3]: Geoffrey Hinton *et. al* *Deep neural networks for acoustic
    modeling in speech recognition*, 2012
	
[^ref4]: Ossama Abdel-Hamid *et. al* *Applying Convolutional Neural
    Networks concepts to hybrid NN-HMM model for speech recognition*
    2012



