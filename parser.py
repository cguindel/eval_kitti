#!/usr/bin/env python

import sys
import os
import numpy as np

# classes = ['car', 'pedestrian', 'cyclist', 'van', 'truck', 'person_sitting', 'tram']
classes = ['car', 'pedestrian', 'cyclist']
difficulties = ['easy', 'moderate', 'hard']
params = ['detection', 'orientation','iour','mppe']

eval_type = ''

if len(sys.argv)<2:
    print 'Usage: parser.py results_folder [evaluation_type]'

if len(sys.argv)==3:
    eval_type = sys.argv[2]

result_sha = sys.argv[1]
txt_dir = os.path.join('build','results', result_sha)

for class_name in classes:
    for param in params:
        if eval_type is '':
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '.txt')
        else:
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '_' + eval_type + '.txt')

        if not os.path.isfile(txt_name):
            print txt_name, 'not found'
            continue

        cont = np.loadtxt(txt_name)

        for idx, difficulty in enumerate(difficulties):
            sum = 0;
            if (param is 'detection') or (param is 'orientation'):
                for i in xrange(0, 40, 4):
                    sum += cont[idx][i]

                average = sum/11.0
            else:
                for i in xrange(0, cont.shape[1], 1):
                    sum += cont[idx][i]

                average = sum/cont.shape[1]
            print class_name, difficulty, param, average

        print '----------------'
        if eval_type is not '':
            break # No orientation for 3D or bird eye

    print '================='
