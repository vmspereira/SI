![CI-CD](https://github.com/vmspereira/si/actions/workflows/main.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/415842359.svg)](https://zenodo.org/badge/latestdoi/415842359)
[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
# Intelligent Systems for Bioinformatics / Sistemas Inteligentes para Bioinformática

A library of algorithms to grasp essential concepts on a Machine Learning curriculum, Machine Learning from scratch using NumPy.
The code is commented with the mathematical foundations needed to understand how the algorithms and models work.
The first version of this repository was used as teaching tool in the Bioinformatics master at Universidade do Minho in 2021.

> What I hear, I forget. What I see, I remember. What I do, I understand.

*Xunzi (340 - 245 BC)*

Everything here is NumPy — no scikit-learn, no TensorFlow, no PyTorch. Every
gradient is written out, so each algorithm can be read end to end, right up to a
working (if tiny) language model.

## Installation

`git clone https://github.com/vmspereira/si.git`

`cd si`

`pip install -e .`

The SVM additionally needs [cvxopt](https://cvxopt.org/) to solve its quadratic
program. That is kept optional, so the rest of the library installs without a
solver toolchain:

`pip install -e .[svm]`

## Running the tests

```
PYTHONPATH=src python -m pytest tests/ -q
```

Run it from the repository root — `tests/test_1.py` loads a dataset by relative
path. Without cvxopt the SVM tests skip; with it, nothing skips. `tox` runs the
same suite on Python 3.10–3.12, and `tox -e lint` runs flake8.

The neural-network layers are checked against finite-difference gradients, which
is the practical way to be sure a hand-derived backward pass is correct.

## Folders organization

The _src_ folder contains the library source.

The _tests_ folder are python tests for continuous integration.

The _datasets_ folder contains some illustrative datasets.

The _scripts_ folder contains notebooks working through each topic.

## ML Algorithms

### Pre-processing

- Standard Scaler
- Variance Threshold
- Select K-best (ANOVA F for classification, correlation F for regression)
- Label Encoder
- One-Hot Encoder

### Unsupervised

- Principal Component Analysis
- K-means Clustering

### Supervised

- Linear regression (closed form and gradient descent)
- Logistic regression
- Naive Bayesian
- Decision Tree (gini or entropy criterion)
- Random Forest
- k-Nearest Neighbors (classification and regression)
- Linear Discriminant Analysis
- SVM

### Neural Networks

- Layers
    - Dense
    - Flatten, Reshape
    - Conv2D (using Img2Col)
    - MaxPooling2D, AveragePooling2D, ConstantPadding2D
    - DropOut
    - BatchNormalization, LayerNorm
    - RNN
    - SelfAttention, MultiHeadAttention
    - Embedding, PositionalEncoding, TransformerBlock
- Activations
    - Sigmoid, Tanh, HardSigmoid
    - ReLU, LeakyReLU, ELU, SELU, SoftPlus
    - Affine, Identity, Exponential
    - SoftMax
- Optimizers
    - SGD (with momentum)
    - Adam
    - Nesterov Accelerated Gradient
    - Adagrad, Adadelta, RMSprop
- Losses and metrics
    - MSE, RMSE, MAE
    - cross entropy, softmax cross entropy
    - accuracy, R², confusion matrix

### Model selection

- Grid Search
- Cross Validation (k-fold, or repeated random subsampling)
- Bagging Ensemble

## A toy language model

The attention layers assemble into a character-level transformer, trained on the
same next-token objective used at scale:

```python
from si.data import Dataset
from si.supervised.nn import (CharTokenizer, make_windows, load_text,
                              build_language_model, generate)

text = load_text('datasets/tiny-text.txt')
tokenizer = CharTokenizer(text)
X, y = make_windows(tokenizer.encode(text), seq_len=32, stride=3)

model = build_language_model(tokenizer.vocab_size, seq_len=32, n_blocks=2)
model.fit(Dataset(X, y))

print(generate(model, tokenizer, 'To be, or not to be', n_chars=200,
               seq_len=32, temperature=0.5))
```

`scripts/eval8.ipynb` walks through it. On the 5.5 KB corpus included here it
trains in about 25 seconds, taking the cross entropy from 3.55 to 0.59 against a
uniform-guess baseline of ln(54) = 3.99.

What it produces is the *statistics* of the text — real words, plausible letter
pairs, line breaks, capitalisation after newlines — and not fluent English:

```
To be, or not to be,
As dreath It ydeath. At lourss and worend and fury,
Steaing thou wrice places, comen, lend to and suk,
Whose worth's unktess to and himself;
```

That gap is compute and training data rather than a shortcoming of the code: the
mechanism is the one large models use, with 5 KB of Shakespeare behind it.

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by]. You are free to use this work as long as you comply to the CC-BY-4 terms. For more information refer to [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/)
