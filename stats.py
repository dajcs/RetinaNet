# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:58:15 2023

@author: dajcs
"""

import matplotlib.pyplot as plt
import pandas as pd
import ast

df = pd.read_csv('./data/Stream1/labels/val.csv')
# Function to convert string representation of list to actual list
def convert_to_list(row):
    try:
        return ast.literal_eval(row)
    except:
        return [None, None, None, None]

# Applying the function to the 'bbox' column
df['bbox_list'] = df['bbox'].apply(convert_to_list)

# Extracting x1, y1, x2, y2 into separate lists
x1_list = [bbox[1] for bbox in df['bbox_list'] if bbox[0] is not None]
y1_list = [bbox[0] for bbox in df['bbox_list'] if bbox[0] is not None]
x2_list = [bbox[3] for bbox in df['bbox_list'] if bbox[0] is not None]
y2_list = [bbox[2] for bbox in df['bbox_list'] if bbox[0] is not None]

# Displaying the first few elements of each list
x1_list[:5], y1_list[:5], x2_list[:5], y2_list[:5]



# Filtering the x1_list and y1_list for values between 0 and 10
x1_filtered = [x for x in x1_list if 0 <= x <= 15]
y1_filtered = [y for y in y1_list if 0 <= y <= 15]

# Plotting the histograms
plt.figure(figsize=(12, 6))

# Histogram for x1_filtered
plt.subplot(1, 2, 1)
plt.hist(x1_filtered, bins=range(16), edgecolor='black')
plt.title('Histogram of x1 values (0-15)')
plt.xlabel('x1 value')
plt.ylabel('Frequency')

# Histogram for y1_filtered
plt.subplot(1, 2, 2)
plt.hist(y1_filtered, bins=range(16), edgecolor='black')
plt.title('Histogram of y1 values (0-15)')
plt.xlabel('y1 value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Filtering the x1_list and y1_list for values between 0 and 10
x2_filtered = [x for x in x2_list if 1014 <= x <= 1024]
y2_filtered = [y for y in y2_list if 1014 <= y <= 1024]

# Plotting the histograms
plt.figure(figsize=(12, 6))

# Histogram for x1_filtered
plt.subplot(1, 2, 1)
plt.hist(x2_filtered, bins=range(1014, 1025), edgecolor='black')
plt.title('Histogram of x1 values (1014-1024)')
plt.xlabel('x2 value')
plt.ylabel('Frequency')

# Histogram for y1_filtered
plt.subplot(1, 2, 2)
plt.hist(y2_filtered, bins=range(1014, 1025), edgecolor='black')
plt.title('Histogram of y1 values (1014-1024)')
plt.xlabel('y2 value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
