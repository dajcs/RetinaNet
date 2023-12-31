# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:59:24 2023

@author: dajcs
"""

import glob

mAP = []
loss = []

for in_file in glob.glob('logs/exp3*.txt'):

    out_file = in_file.replace('_res', '_out')

    with open(in_file, 'r', encoding='utf-8') as sf, open(out_file, 'w', encoding='utf-8') as of:
            for line in sf:
                #  line = line.rstrip()
                if line.startswith('Epoch'):
                    of.write(line)
                    
             