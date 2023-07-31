import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.special import log_softmax


AA_ORDER = ['L', 'S', 'R', 'V', 'A', 'P', 'T', 'G', '*', 'I', 'Y', 'F', 'C', 'H', 'Q', 'N', 'K', 'D', 'E', 'W', 'M', '-']
AA_IDX = {AA_ORDER[i]: i for i in range(len(AA_ORDER))}


class MBEModel:
    '''Example/template for using a machine learning classifier for MBE.'''
    def __init__(self, model, seq_encoding_fn=None):
        '''MBE using a model with 'fit' and 'predict_log_proba' methods.'''
        self.encoding_fn = one_hot_encode if seq_encoding_fn is None else seq_encoding_fn
        self.model = model
    
    def fit(self, sequences, counts, normalize_classes=True):
        '''Fit underlying classifier using provided sequences (strings) and counts (integer array).
        Optionally normalize counts to balance classes.'''
        X, y, w = [], [], []
        n_classes = counts.shape[1]
        if normalize_classes:
            counts = np.amax(np.sum(counts, axis=0)) * counts / np.sum(counts, axis=0)
        for c in range(n_classes):
            observed = counts[:, c] > 0
            X.append(np.array(Parallel(n_jobs=-1, verbose=1)(delayed(self.encoding_fn)(seq) for seq in sequences[observed])))
            y.append([c] * int(np.sum(observed)))
            w.append(counts[:, c][observed])
        X = np.row_stack(X)
        y = np.concatenate(y)
        w = np.concatenate(w)
        self.model.fit(X, y, sample_weight=w)
    
    def log_enrichment(self, sequences, numerator_index=1, denominator_index=0):
        '''Predict log-enrichment (aka. log-density ratio) values for the
        provided sequences (strings) using trained model.'''
        X = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(self.encoding_fn)(seq) for seq in sequences))
        log_p = self.model.predict_log_proba(X)
        return log_p[:, numerator_index] - log_p[:, denominator_index]


class MBETensorflowModel(MBEModel):
    '''Example/template for adapting base MBEModel for use with a specific ML package.
    This example adapts data processing, fitting, and prediction for TensorFlow models.'''
    def fit(self, sequences, counts, ids=None, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4),
            seed=None, flatten=True, tile=True, callbacks=None, epochs=10, normalize_classes=True):
        '''Fit underlying classifier using provided sequences (strings) and counts (integer array).
        Optionally normalize counts to balance classes.'''
        if ids is None:
            ids = np.arange(len(sequences))
        if normalize_classes:
            counts = np.amax(np.sum(counts, axis=0)) * counts / np.sum(counts, axis=0)
        dataset = get_classification_dataset(sequences, counts, ids, self.encoding_fn, batch_size,
                                             shuffle, shuffle_buffer, seed, flatten, tile)
        return self.model.fit(dataset, epochs=epochs, callbacks=callbacks)
    
    def log_enrichment(self, sequences, numerator_index=1, denominator_index=0, batch_size=1024, flatten=True):
        '''Predict log-enrichment (aka. log-density ratio) values for the
        provided sequences (strings) using trained model.'''
        dataset = get_sequence_dataset(sequences, self.encoding_fn, batch_size=batch_size, shuffle=False, flatten=flatten)
        logits = self.model.predict(dataset)#[:n_samples]
        log_p = log_softmax(logits, axis=1)
        return log_p[:, numerator_index] - log_p[:, denominator_index]

        
def one_hot_encode(seq):
    """
    Returns a one-hot encoded vector representing
    the string input sequence.
    """
    seq_len = len(seq)
    alphabet_len = len(AA_ORDER)
    one_hot = np.zeros((seq_len, alphabet_len))
    for i in range(seq_len):
        one_hot[i, AA_IDX[seq[i]]] = 1
    return one_hot.flatten()


def index_encode(seq):
    """
    Returns an integer vector of indices representing
    the input sequence.
    """
    seq_len = len(seq)
    indices = [AA_IDX[s] for s in seq]
    return np.array(indices)


def get_classification_dataset(sequences, counts, ids, encoding_fn, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True, tile=True):
    """Returns a tf.data.Dataset that generates input/label/count data."""
    seq_len = len(sequences.iloc[0])
    ids = ids[np.sum(counts[ids], axis=1) > 0] # Filter unobserved sequences
    X = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(encoding_fn)(seq) for seq in sequences.iloc[ids]))
    counts = counts[ids]
    ds = tf.data.Dataset.from_tensor_slices((X, counts))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    flat_shape = [-1, seq_len * onehot_depth]
    n_classes = counts.shape[1]
    tile_dims = [n_classes, 1] if flatten else [n_classes, 1, 1]
    def tf_encoding_fn(x, c):
        x = tf.one_hot(x, onehot_depth)
        if flatten:
            x = tf.reshape(x, flat_shape)
        classes = tf.repeat(tf.range(n_classes), repeats=tf.shape(x)[0])
        if tile:
            x = tf.tile(x, tile_dims)
        counts = tf.reshape(tf.transpose(c), [-1])
        return x, classes, counts
    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_sequence_dataset(sequences, encoding_fn, ids=None, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
    """Returns a tf.data.Dataset that generates input sequences only."""
    if ids is None:
        ids = np.arange(len(sequences))
    seq_len = len(sequences.iloc[0])
    X = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(encoding_fn)(seq) for seq in sequences.iloc[ids]))
    ds = tf.data.Dataset.from_tensor_slices((X,))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    flat_shape = [-1, seq_len * onehot_depth]
    def tf_encoding_fn(x):
        x = tf.one_hot(x, onehot_depth)
        if flatten:
            x = tf.reshape(x, flat_shape)
        return x
    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds