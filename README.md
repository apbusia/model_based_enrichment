# MBE: Model-based enrichment estimation and prediction for differential sequencing data

This repo provides a template for implementing the model-based enrichment (MBE) approach from the paper

A. Busia and J. Listgarten. MBE: Model-based enrichment estimation and prediction for differential sequencing data. *Genome Biology*, 2023.

using standard Python machine learning (ML) libraries.

Note that, because MBE is classifier-based, it can be implemented using any standard ML package that implements probabilistic classifiers. This package provides a standard ```MBEModel``` compatible with models implemented by several standard Python libraries, and its usage is demonstrated with [```scikit-learn```](https://scikit-learn.org/stable/index.html) in ```sklearn_example.py``` which run a simple MBE estimation analysis given a CSV of sequences and sequencing counts. This package also provides an adaptation of the base ```MBEModel``` compatible with any ML classifier implemented using TensorFlow, which (1) allows MBE to make use of modern-day neural network models (demonstrated in ```tf_example.py```) and (2) provides a template for adapting ```MBEModel``` to a specific ML library of your choice. For specific examples of some TensorFlow classifier architectures, see the source code from the paper: https://github.com/apbusia/selection_dre


## Training a model for MBE

MBE can be implemented using any probabilistic classifier, `classifier_model`, of choice. The base `MBEModel` can be used to train any underlying `classifier_model` with `fit` and `predict_log_proba` methods, such as scikit-learn models:

```python

    from model_based_enrichment import model_based_enrichment as mbe
    
    mbe_model = mbe.MBEModel(classifier_model)
    # sequences is a list of the N unique observed read sequences in the sequencing dataset.
    # counts is a (n_unique_sequences, n_conditions) matrix of observed read counts.
    mbe_model.fit(sequences, counts)
```

This workflow can easily be adapted to run MBE with different `classifier_model` models using `MBEModel` as a base class; see e.g. `MBETensorflowModel`.


## Predicting / estimating log-enrichment

Once the underlying probabilistic classifier has been trained, MBE can be used to predict the log-enrichment of unobserved sequences and estimate the log-enrichment of observed sequences:


```python

    predictions = mbe_model.log_enrichment(sequences) # sequences is a list of sequences to evaluate.

```
