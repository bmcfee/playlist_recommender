#!/usr/bin/env python
'''Merge encoding dictionaries into a single dataframe.

Use this on the output of encode_msd.py
'''

import sys
import argparse
import cPickle as pickle
import pandas as pd


def get_args(args):
    '''Argument parser wrapper'''
    parser = argparse.ArgumentParser(description='Merge encoding dictionaries '
                                     'into a pandas pickle')

    parser.add_argument('-o', '--output',
                        dest='output_file',
                        required=True,
                        type=str,
                        help='Path to store pickle')

    parser.add_argument('input_files', type=str, nargs='+',
                        help='One or more input pickles')

    return vars(parser.parse_args(args))


def merge_encodings(output_file='', input_files=None):
    '''Does the work of merging encodings'''
    data = {}

    for inf in input_files:
        print "Processing {:s}".format(inf)
        with open(inf, 'r') as fdesc:
            data.update(pickle.load(fdesc))

    print 'Building dataframe'
    data = pd.DataFrame.from_dict(data, orient='index', dtype='float32')

    print 'Saving to {:s}'.format(output_file)
    data.to_pickle(output_file)


if __name__ == '__main__':
    merge_encodings(**get_args(sys.argv[1:]))
