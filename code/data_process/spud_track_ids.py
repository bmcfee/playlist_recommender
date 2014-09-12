#!/usr/bin/env python
# CREATED:2014-09-12 12:08:18 by Brian McFee <brian.mcfee@nyu.edu>
# extract the spotify track ids from SPUD.sqlite and save them to a json file on disk


import argparse
import sys

import sqlite3
import ujson as json

def get_ids(sqlitefile):
    ''' query the sqlite database for spotify ids '''

    results = []
    
    with sqlite3.connect(sqlitefile) as dbc:
        cur = dbc.cursor()
        cur.execute('select spotifyid from tracks ORDER BY trackid ASC')
        results = [res[0] for res in cur.fetchall()]
    
    return results

def save_ids(results, outfile):
    '''save results to disk'''

    with open(outfile, 'w') as f:
        json.dump(results, f)

def get_args(args):
    
    parser = argparse.ArgumentParser(description='SPUD database spotify track id extractor')

    parser.add_argument('SPUD.sqlite', action='store', help='Path to SPUD.sqlite')

    parser.add_argument('output.json', action='store', help='Path to store spotify ids as json')

    return vars(parser.parse_args(args))

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    results = get_ids(args['SPUD.sqlite'])
    save_ids(results, args['output.json'])

