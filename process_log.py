# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:59:24 2023

@author: dajcs
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt

mAP = []
loss = []

for file in glob.glob('logs/exp3*.txt'):

    with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                field = line.split()
                if field[2] == 'mAP:':
                    mAP.append(float(field[-1]))
                else:
                    loss.append(float(field[-1]))


df = pd.DataFrame({'loss': loss, 'mAP': mAP})

# Extracting the data for plotting
x = df.index  # Epochs starting from 0
y1 = df['loss']  # Loss values
y2 = df['mAP']  # mAP values

# Creating a plot with two different y-axis scales
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot 'loss' on the primary y-axis
ax1.plot(x, y1, 'b-', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.set_ylim([0, 0.5])  # Set y-axis range for 'loss'
ax1.tick_params('y', colors='b')
ax1.grid(True)

# Create a secondary y-axis for 'mAP'
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-', label='mAP')
ax2.set_ylabel('mAP', color='r')
ax2.set_ylim([0.6, 0.9])  # Set y-axis range for 'mAP'
ax2.tick_params('y', colors='r')

# Set x-axis ticks to every 20th epoch, starting from 0
x_ticks = range(0, len(x) + 1, 20)
ax1.set_xticks(x_ticks)

# Adding a title and adjusting the layout
plt.title('Loss on Training (left axis) and mAP on Validation (right axis) over Epochs')
fig.tight_layout()

plt.show()