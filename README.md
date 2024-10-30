# Symmetry in Machine Learning Using Vector Fields

## Introduction

Continuous symmetry discovery has recently been approached using large models such as GANs. This project aims to reformulate the problem of continuous symmetry discovery in terms of discovering tangent vector fields which Lie-differentiate given smooth functions to zero, which functions include probability distributions, regression/classification functions, or other so-called "machine learning functions." The discovered vector fields represent continuous symmetries, since their associated flows are 1-parameter transformations under which $X$-invariant functions are invariant. By reformulating continuous symmetry discovery in this way, the computational complexity of continuous symmetry discovery is reduced. Additionally, where previous methods have seemingly only discovered symmetries which are affine transformations, a vector field approach allows one to easily extend the search space of symmetries to symmetries of far greater complexity.

To discover symmetry in data, a function which characterizes the data must first be given (or estimated). This can take the form of estimating the underlying data distribution, applying level set estimation, or metric tensor estimation. Other ways of characterizing/summarizing data in terms of smooth functions may also be applied.


## Associated Papers

[Symmetry Discovery Beyond Affine Transformations](https://arxiv.org/abs/2406.03619) (NeurIPS 2024)
