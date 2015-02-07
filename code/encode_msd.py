#!/usr/bin/env python
'''Vector quantization of MSD timbre features'''

from argparse import ArgumentParser

import sys
import os

import glob
import cPickle as pickle
import pandas as pd
import numpy as np

from joblib import Parallel, delayed


def process_arguments(args):
    '''Process arguments from the command line'''

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

    timbres = pd.DataFrame(list(hdf.get_node('/analysis/segments_timbre')),
                           dtype='float32')

    track_id = hdf['/analysis/songs']['track_id'][0]

    hdf.close()
    return track_id, timbres


def msd_encoder(filename, vector_quantizer):
    '''Compute codeword histograms from MSD analysis files'''

    print '\t{:s}'.format(os.path.basename(filename))

    track_id, timbres = get_track_timbre(filename)

    # Compute the codeword histogram for this track
    codeword_hist = np.ravel(vector_quantizer.transform(timbres).mean(axis=0))

    # Return the dict
    return {track_id: codeword_hist}


def run_encoding(num_cores=None, verbose=None,
                 vq_pickle=None, output_pickle=None,
                 msd_path=None, max_files=None,
                 batch_size=10000):
    '''Do the big encoding job'''

    # Get the master file list
    msd_path = os.path.abspath(msd_path)

    all_files = sorted(glob.glob(os.path.join(msd_path,
                                              'data',
                                              '*', '*', '*',
                                              '*.h5')))

    if max_files is not None:
        all_files = all_files[:max_files]

    # Load the vector_quantizer object
    with open(vq_pickle, 'r') as fdesc:
        vector_quantizer = pickle.load(fdesc)['VQ']

    print str(vector_quantizer)

    for start in range(0, len(all_files), batch_size):
        end = min(len(all_files), start + batch_size)

        print "Processing files {:d}--{:d}".format(start, end)

        files = all_files[start:end]

        results = {}
        for q in Parallel(n_jobs=num_cores,
                          verbose=verbose)(delayed(msd_encoder)(fn, vector_quantizer) for fn in files):
            results.update(q)

        with open('{:s}_{:d}-{:d}'.format(output_pickle, start, end),
                  mode='w') as fdesc:
            pickle.dump(results, fdesc, protocol=-1)


if __name__ == '__main__':

    run_encoding(**process_arguments(sys.argv[1:]))
