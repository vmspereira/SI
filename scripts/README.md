Examples
============

This folder contains examples to test and validate your code.

## Jupyter Notebooks:

- [eval1](eval1.ipynb): Dataset pre-processing, PCA and KNN;
- [eval2](eval2.ipynb): Linear and Logistic Regression, grid search and cross-validation;
- [eval3](eval3.ipynb): Decision Tree and bagging ensemble;
- [eval4](eval4.ipynb): A simple XNOR Neural Network;
- [eval5](eval5.ipynb): CNN for MNIST dataset;
- [eval6](eval6.ipynb): RNN, and what recurrence is and is not needed for;
- [eval7](eval7.ipynb): MNIST AutoEncoder;
- [eval8](eval8.ipynb): Attention, a transformer and a character-level language model;
- [eval9](eval9.ipynb): Naive Bayes, LDA, Random Forest and SVM compared.

Run them from this folder: they resolve `datasets/` relative to the repository
root via `os.path.dirname(os.path.realpath('.'))`.

`eval5` and `eval7` contain long training cells (1000 iterations each); reduce
those counts if you only want to check that they run. The rest complete in
seconds.