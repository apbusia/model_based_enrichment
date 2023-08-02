import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats
from model_based_enrichment import model_based_enrichment as mbe

tfk = tf.keras
tfkl = tf.keras.layers


def get_regularizer(l1_reg=0., l2_reg=0.):
    """
    Returns a keras regularizer object given 
    the l1 and l2 regularization parameters
    """
    if l1_reg > 0 and l2_reg > 0:
        reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = tfk.regularizers.l1(l1_reg)
    elif l2_reg > 0:
        reg = tfk.regularizers.l2(l2_reg)
    else:
        reg = None
    return reg


def make_ann_classifier(input_shape, n_outputs=2, num_hid=2, hid_size=100, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Builds an artificial neural network model for classification.
    
    Copied from source code: https://github.com/apbusia/selection_dre/blob/main/modeling.py
    See source code for more TF model examples.
    """
    reg = get_regularizer(l1_reg, l2_reg)
    inp = tfkl.Input(shape=input_shape)
    z = inp
    for i in range(num_hid):
        z = tfkl.Dense(hid_size, activation='relu', kernel_regularizer=reg, bias_regularizer=reg)(z)
    out = tfkl.Dense(n_outputs, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(z)
    model = tfk.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipvalue=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model


def main(args):
    # Load sequences and counts from CSV file.
    print('\nLoading data:', args.data_file)
    seqs, counts, groundtruth = mbe.parse_csv(args.data_file,
                                              count_columns=args.count_columns,
                                              groundtruth_column=args.groundtruth_column)
    
    print('\nFitting model...')
    input_shape = (len(seqs.iloc[0]) * len(mbe.AA_ORDER))
    K.clear_session()
    tf_model = make_ann_classifier(input_shape, n_outputs=len(counts), num_hid=args.n_hidden, hid_size=args.hidden_size, lr=args.learning_rate)
    mbe_model = mbe.MBETensorflowModel(tf_model, mbe.index_encode)
    mbe_model.fit(seqs, counts, epochs=args.epochs, batch_size=args.batch_size, n_jobs=-1)
    
    print('\nMaking predictions...')
    preds = mbe_model.log_enrichment(seqs, batch_size=args.batch_size, n_jobs=-1)
    
    print('\nComparing to ground truth fitness...')
    spearman_r = stats.spearmanr(groundtruth, preds)[0]
    print('\nSpearman={:.3f}'.format(spearman_r))
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV containing a "seq" column of sequences and a counts column per condition', type=str)
    parser.add_argument('count_columns', help='column names in data_file', type=str, nargs='+')
    parser.add_argument('--groundtruth_column', default='true_enrichment', help='column name in data_file', type=str)
    parser.add_argument("--n_hidden", default=2, help="number of hidden layers in nn model", type=int)
    parser.add_argument("--hidden_size", default=100, help="size of hidden layers in nn model", type=int)
    parser.add_argument("--learning_rate", default=1e-3, help="learning rate for gradient descent", type=float)
    parser.add_argument("--epochs", default=10, help="number of epochs to run training", type=int)
    parser.add_argument("--batch_size", default=1000, help="number of samples per batch during training", type=int)
    args = parser.parse_args()
    main(args)