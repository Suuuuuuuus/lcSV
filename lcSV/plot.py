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
import itertools
from itertools import combinations_with_replacement
import collections
import sqlite3
import random

import pyranges as pr
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

from scipy.stats import nbinom
from scipy.stats import geom, beta
from scipy.special import logsumexp

from .model import *

__all__ = [
    "plot_sv_coverage", "plot_training", "plot_sv_coverage_by_gt"
]


def plot_sv_coverage(cov, chromosome, start, end, flank, calling_dict, side = 'both', tick_step = 0.1):
    means, _ = normalise_by_flank(cov, start, end, flank, side = side)
    
    region = cov[(cov['position'] >= start) & (cov['position'] <= end)]
    region['position'] = region['position']/1e6
    
    colors = plt.get_cmap(CATEGORY_CMAP_STR).colors[:10]
    colors = [mcolors.to_hex(c) for c in colors]
    
    all_samples = np.array([s.split(':')[0] for s in cov.columns[1:-1]])

    for i, k in enumerate(calling_dict.keys()):
        samples = calling_dict[k]
        for s in samples:
            index = np.where(all_samples == s)[0][0]
            
            if k == (0,0):
                plt.plot(region['position'], region[f'{s}:coverage']/means[index], alpha = 1, color = '0.8')
            else:
                plt.plot(region['position'], region[f'{s}:coverage']/means[index], alpha = 1, color = colors[i - 1])
   
    ticks = get_ticks(region, tick_step)
    if tick_step == 0.001:
        plt.xticks(ticks, [f"{int(tick*1000)}" for tick in ticks], rotation = 45)
        plt.xlabel(f'Chromosome {chromosome} (Kb)')
    else:
        plt.xticks(ticks, [f"{tick:.{int(np.log10(1/tick_step))}f}" for tick in ticks], rotation = 45)
        plt.xlabel(f'Chromosome {chromosome} (Mb)')

    color_handles = []
    color_index = [0,0]
    for i, k in enumerate(calling_dict.keys()):
        if k != (0,0):
            color_handles.append(Line2D([0], [0], color=colors[i-1], label=k))

    legend1 = plt.legend(handles=color_handles, loc='upper left', prop={'size': 10}, framealpha=1)
    legend1.get_title().set_fontsize(9)
    plt.gca().add_artist(legend1)

    plt.ylabel('Coverage (X)')
    return None

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

    colors = plt.get_cmap(CATEGORY_CMAP_STR).colors[:n_haps]
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
    return None

def plot_sv_coverage_by_gt(cov, chromosome, start, end, flank, calling_dict, side = 'both', tick_step = 0.1):
    means, _ = normalise_by_flank(cov, start, end, flank, side = side)
    
    region = cov[(cov['position'] >= start) & (cov['position'] <= end)]
    region['position'] = region['position']/1e6
    region.iloc[:,1:-1] = region.iloc[:,1:-1]/means[np.newaxis,:]
    
    if len(calling_dict.keys()) > 10:
        print('Only region with less than 3 different haplotypes can be printed.')
        return None
    
    colors = plt.get_cmap(CATEGORY_CMAP_STR).colors[:10]
    colors = [mcolors.to_hex(c) for c in colors]
    
    all_samples = np.array([s.split(':')[0] for s in region.columns[1:-1]])
    for i, k in enumerate(calling_dict.keys()):
        samples = calling_dict[k]
        indices = np.where(np.isin(all_samples, samples))[0] + 1
        
        tmp = pd.concat([region.iloc[:, 0], region.iloc[:,indices]], axis = 1)
        plt.plot(tmp['position'], tmp.iloc[:,1:].mean(axis = 1), alpha = 1, color = colors[i])
   
    ticks = get_ticks(region, tick_step)
    if tick_step == 0.001:
        plt.xticks(ticks, [f"{int(tick*1000)}" for tick in ticks], rotation = 45)
        plt.xlabel(f'Chromosome {chromosome} (Kb)')
    else:
        plt.xticks(ticks, [f"{tick:.{int(np.log10(1/tick_step))}f}" for tick in ticks], rotation = 45)
        plt.xlabel(f'Chromosome {chromosome} (Mb)')

    color_handles = []
    color_index = [0,0]
    for i, k in enumerate(calling_dict.keys()):
        color_handles.append(Line2D([0], [0], color=colors[i], label=k))

    legend1 = plt.legend(handles=color_handles, loc='upper left', prop={'size': 10}, framealpha=1)
    legend1.get_title().set_fontsize(9)
    plt.gca().add_artist(legend1)

    plt.ylabel('Coverage (X)')
    return None