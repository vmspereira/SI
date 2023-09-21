![CI-CD](https://github.com/vmspereira/si/actions/workflows/main.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/415842359.svg)](https://zenodo.org/badge/latestdoi/415842359)
[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
# Intelligent Systems for Bioinformatics / Sistemas Inteligentes para Bioinformática

A library of algorithms to grasp essential concepts on a Machine Learning curriculum, Machine Learning from scratch using NumPy.
The code is commented with the mathematical fundations needed to understand how the algoritms and models work.
The first version of this repository was used as teaching tool in the Bioinformatics master at Universidade do Minho in 2021.

> What I hear, I forget. What I see, I remember. What I do, I understand.

*Xunzi (340 - 245 BC)*

## Installation

`git clone https://github.com/vmspereira/si.git`

`cd si`

`pip install -e .`

## Folders organization

The _src_ folder contains the base source over which you will implement you code.

The _tests_ folder are python tests for continuous integration.

The _dataset_ folder contains some illustrative datasets.

The _script_ folder contains some notebooks to test your code.

## ML Algorithms

### Pre-processing

- Standard Scaler
- Variance Threshold
- Select K-best


### Unsupervised

- Principal Component Analysis
- K-means Clustering

### Supervised

- Linear regression
- Logistic regression
- Naive Bayesian
- Decision Tree
- Random Forest
- k-Nearest Neighbors
- SVM


- Neural Networks 
    - Dense
    - Flatten
    - Conv2D (using Img2Col)
    - MaxPooling2D
    - DropOut
    - BatchNormalization
    - RNN


- Grid Search
- Bagging Ensemble
- Cross Validation

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by]. You are free to use this work as long as you comply to the CC-BY-4 terms. For more information refer to [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/)
