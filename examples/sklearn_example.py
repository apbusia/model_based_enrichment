import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats
from model_based_enrichment import model_based_enrichment as mbe


def main(args):
    # Load sequences and counts from CSV file.
    print('\nLoading data:', args.data_file)
    data_df = pd.read_csv(args.data_file)
    seqs = data_df['seq'] # List of read sequences.
    counts = np.column_stack([data_df[c] for c in args.count_columns]) # n_sequences x n_conditions array of read counts.
    groundtruth = data_df[args.groundtruth_column].values # Groundtruth property/enrichment values for evaluation.
    observed = np.sum(counts, axis=1) > 0
    seqs, groundtruth, counts = seqs[observed], groundtruth[observed], counts[observed]
    
    print('\nFitting model...')
    mbe_model = mbe.MBEModel(LogisticRegression())
    mbe_model.fit(seqs, counts)
    
    print('\nMaking predictions...')
    preds = mbe_model.log_enrichment(seqs)
    spearman_r = stats.spearmanr(groundtruth, preds)[0]
    print('\nSpearman={:.3f}'.format(spearman_r))
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV containing a "seq" column of sequences and a counts column per condition', type=str)
    parser.add_argument('count_columns', help='column names in data_file', type=str, nargs='+')
    parser.add_argument('--groundtruth_column', default='true_enrichment', help='column name in data_file', type=str)
    args = parser.parse_args()
    main(args)