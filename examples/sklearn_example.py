import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
from model_based_enrichment import model_based_enrichment as mbe


def main(args):
    # Load sequences and counts from CSV file.
    print('\nLoading data:', args.data_file)
    seqs, counts, groundtruth = mbe.parse_csv(args.data_file,
                                              count_columns=args.count_columns,
                                              groundtruth_column=args.groundtruth_column)
    
    print('\nFitting model...')
    mbe_model = mbe.MBEModel(LogisticRegression())
    mbe_model.fit(seqs, counts, n_jobs=-1)
    
    print('\nMaking predictions...')
    preds = mbe_model.log_enrichment(seqs, n_jobs=-1)
    
    print('\nComparing to ground truth fitness...')
    spearman_r = stats.spearmanr(groundtruth, preds)[0]
    print('\nSpearman={:.3f}'.format(spearman_r))
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV containing a "seq" column of sequences and a counts column per condition', type=str)
    parser.add_argument('count_columns', help='column names in data_file', type=str, nargs='+')
    parser.add_argument('--groundtruth_column', default='true_enrichment', help='column name in data_file', type=str)
    args = parser.parse_args()
    main(args)