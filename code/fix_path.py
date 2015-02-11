#!/usr/bin/env python

import sys
import os

new_path = sys.path[:1]
new_path.append(os.path.join(os.environ['HOME'],
                             '.local',
                             'python2.7',
                             'site-packages'))

new_path.extend(sys.path[1:])

sys.path = new_path
