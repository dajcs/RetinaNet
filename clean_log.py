# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:59:24 2023

@author: dajcs
"""

input_path = 'exp1gpu_res.txt'
output_path = 'exp1gpu_clean.txt'

with open(input_path, 'r', encoding='utf-8') as input_file, open(input_path, 'w', encoding='utf-8') as output_file:
         for line in input_file:
             line = line.rstrip()
             if line.endswith('s/it]'):
                 continue
             if line.endwwith('it/s]') and not '100%' in line:
                 continue
             output_file.write(line + '\n')
             