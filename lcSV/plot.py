import io
import os
import sys
import csv
import gzip
import time
import json
import secrets
import copy
import pickle
import multiprocessing
import subprocess
import resource

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from .model import *
from .process import *

__all__ = [
    "plot_sv_coverage", "plot_training", "plot_sv_coverage_by_gt"
]

def plot_training(results, show_legends = True):
    L = len(results['model_ary'][0].haps[0])
    lls = results['ll_ary']
    n = len(lls)
    haps = {np.ones(L).tobytes(): 0}
    freqs = np.zeros((1, n))

    for i in range(n):
        m = results['model_ary'][i]
        for j, hap in enumerate(m.haps):
            h = hap.tobytes()
            if h not in haps.keys():
                haps[h] = freqs.shape[0]
                freqs = np.append(freqs, np.zeros((1,n)), axis=0)
            ridx = haps[h]
            freqs[ridx, i] = m.freqs[j]

    n_haps = freqs.shape[0]
    x = np.arange(1, n+1)
    fig, ax1 = plt.subplots()

    ax1.plot(x, lls, ls = '--', color='black')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Model score')
    ax1.tick_params(axis='y')
    y1_max = lls[0]
    y1_min = lls[-1]
    y1_ext = (y1_max - y1_min)/3
    ax1.set_ylim((y1_min - y1_ext, y1_max + y1_ext))

    colors = plt.get_cmap('tab20').colors[:n_haps]
    colors = [mcolors.to_hex(c) for c in colors]

    ax2 = ax1.twinx()
    for i in range(freqs.shape[0]):
        x = np.flatnonzero(freqs[i,:])
        ax2.plot(x, freqs[i,x]*100, color=colors[i])
    ax2.set_ylabel('Haplotype frequencies (%)')
    ax2.tick_params(axis='y')
    ax2.set_ylim((-5, 105))

    ax1.grid(True, alpha = 0.7)
    
    if show_legends:
        color_handles = []
        for i in range(n_haps):
            color_handles.append(Line2D([0], [0], color=colors[i], label=f'Haplotype {i}'))

        legend1 = plt.legend(handles=color_handles, title='Haplotypes', 
                             prop={'size': 10}, framealpha=1)
        legend1.get_title().set_fontsize(10)
        plt.gca().add_artist(legend1)

        linestyle_handles = [
            Line2D([0], [0], color='black', lw=2, linestyle='--', label='Model score')
        ]
        legend2 = plt.legend(handles=linestyle_handles, title='Modified BIC', 
                             prop={'size': 10}, framealpha=1, bbox_to_anchor = (1, 0.8))
        legend2.get_title().set_fontsize(10)
        plt.gca().add_artist(legend2)   
        legend2.get_frame().set_zorder(2)
        
    plt.show()
    return None

def plot_sv_coverage(means, coverage, samples, calling_dict):
    region = coverage/means[np.newaxis,:]
    
    colors = plt.get_cmap('tab20').colors[:10]
    colors = [mcolors.to_hex(c) for c in colors]
    
    xaxis = np.arange(coverage.shape[0])
    
    fig = plt.figure(figsize = (6,4))
    
    for i, k in enumerate(calling_dict.keys()):
        tmp_samples = calling_dict[k]
        for s in tmp_samples:
            index = np.where(samples == s)[0][0]
            
            if k == (0,0):
                plt.plot(xaxis, region[:, index], alpha = 1, color = '0.8')
            else:
                plt.plot(xaxis, region[:, index], alpha = 1, color = colors[i - 1])

    color_handles = []
    color_index = [0,0]
    for i, k in enumerate(calling_dict.keys()):
        if k != (0,0):
            color_handles.append(Line2D([0], [0], color=colors[i-1], label=k))

    legend1 = plt.legend(handles=color_handles, loc='upper left', prop={'size': 10}, framealpha=1)
    legend1.get_title().set_fontsize(9)
    plt.gca().add_artist(legend1)

    plt.ylabel('Coverage (X)')
    plt.show()
    return None

def plot_sv_coverage_by_gt(means, coverage, samples, calling_dict):
    region = coverage/means[np.newaxis,:]
    
    if len(calling_dict.keys()) > 10:
        print('Only region with less than 4 different haplotypes can be printed.')
        return None
    
    colors = plt.get_cmap('tab20').colors[:10]
    colors = [mcolors.to_hex(c) for c in colors]
    xaxis = np.arange(coverage.shape[0])
    
    for i, k in enumerate(calling_dict.keys()):
        tmp_samples = calling_dict[k]
        indices = np.where(np.isin(samples, tmp_samples))[0]
        plt.plot(xaxis, region[:,indices].mean(axis = 1), alpha = 1, color = colors[i])

    color_handles = []
    color_index = [0,0]
    for i, k in enumerate(calling_dict.keys()):
        color_handles.append(Line2D([0], [0], color=colors[i], label=k))

    legend1 = plt.legend(handles=color_handles, loc='upper left', prop={'size': 10}, framealpha=1)
    legend1.get_title().set_fontsize(9)
    plt.gca().add_artist(legend1)

    plt.ylabel('Coverage (X)')
    plt.show()
    return None