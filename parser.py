#!/usr/bin/env python

import sys
import os
import numpy as np

classes = ['car', 'pedestrian', 'cyclist']
difficulties = ['easy', 'moderate', 'hard']
params = ['detection', 'orientation']

if len(sys.argv)<2:
    print 'Usage: parser.py results_folder'

# print "This is the name of the script: ", sys.argv[0]
# print "Number of arguments: ", len(sys.argv)
# print "The arguments are: " , str(sys.argv)

result_sha = sys.argv[1]
txt_dir = os.path.join('build','results', result_sha)

for class_name in classes:
    for param in params:
        txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '.txt')

        cont = np.loadtxt(txt_name)

        for idx, difficulty in enumerate(difficulties):
            sum = 0;
            for i in xrange(0, 40, 4):
                sum += cont[idx][i]

            average = sum/11.0
            print class_name, difficulty, param, average

        print '----------------'

    print '================='
