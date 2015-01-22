#!/usr/bin/env python


from argparse import ArgumentParser

import sys
import os

import glob
import cPickle as pickle
import pandas as pd
import numpy as np

from joblib import Parallel, delayed


def process_arguments(args):

    parser = ArgumentParser(description='Multi-core MSD timbre quantizer')

    parser.add_argument('-n', '--num_cores',
                        dest='num_cores',
                        type=int,
                        default=2,
                        help='Number of cores to run in parallel')

    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        type=int,
                        default=0,
                        help='Verbosity level')

    parser.add_argument('-m', '--max-files', dest='max_files',
                        type=int,
                        default=None,
                        help='Maximum number of files to process.')

    parser.add_argument('vq_pickle', type=str,
                        help='Path to the vector quantization model')

    parser.add_argument('output_pickle', type=str,
                        help='Path to store the encoded data object')

    parser.add_argument('msd_path', type=str,
                        help='Path to the millionsong dataset')

    return vars(parser.parse_args(args))


def get_track_timbre(filename):
    '''Get the track id and timbre matrix from an MSD h5 file'''

    hdf = pd.HDFStore(filename, mode='r')

    timbres = pd.DataFrame([_ for _ in hdf.get_node('/analysis/segments_timbre')],
                           dtype='float32')

    track_id = hdf['/analysis/songs']['track_id'][0]

    hdf.close()
    return track_id, timbres


def msd_encoder(filename, VQ):
    '''Compute codeword histograms from MSD analysis files'''

    track_id, timbres = get_track_timbre(filename)

    # Compute the codeword histogram for this track
    codeword_hist = np.ravel(VQ.transform(timbres).mean(axis=0))

    # Wrap it as a dataframe
    newframe = pd.DataFrame.from_dict({track_id: codeword_hist},
                                      orient='index')

    # Sparsify and return
    return newframe.to_sparse(fill_value=0.0)


def run_encoding(num_cores=None, verbose=None,
                 vq_pickle=None, output_pickle=None,
                 msd_path=None, max_files=None):
    '''Do the big encoding job'''

    # Get the master file list
    msd_path = os.path.abspath(msd_path)

    files = sorted(glob.glob(os.path.join(msd_path,
                                          'data',
                                          '*', '*', '*',
                                          '*.h5')))

    if max_files is not None:
        files = files[:max_files]

    # Load the VQ object
    with open(vq_pickle, 'r') as f:
        VQ = pickle.load(f)['VQ']

    results = Parallel(n_jobs=num_cores,
                       verbose=verbose)(delayed(msd_encoder)(fn, VQ)
                                        for fn in files)

    results = pd.concat(results).to_sparse(fill_value=0.0)

    results.to_pickle(output_pickle)


if __name__ == '__main__':

    parameters = process_arguments(sys.argv[1:])

    run_encoding(**parameters)
