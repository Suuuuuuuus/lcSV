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

import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.special import logsumexp

from .model import *

__all__ = ["deresolute_windows", "delineate_region", "assess_bins", "extract_target_cov", "normalise_by_flank_naive", "normalise_by_flank_with_map", "call_sv_samples", "generate_diploid_profiles", "run_inverse_length_penalty"]

def deresolute_windows(df, window_size, normalise = False):
    len_window = int(df.iloc[1,0] - df.iloc[0,0])
    len_region = int(len(df)*len_window)
    if window_size % len_window != 0 or (len_region % window_size) != 0:
        raise ValueError(f"Window size must be a multiple of {len_window} and divide {len_region}.")
    multiple = window_size/len_window
        
    df['window_start'] = (df.iloc[:, 0] // window_size) * window_size
    new_pos = df['window_start'].unique()
    agg_df = df.groupby('window_start').sum().reset_index()
    agg_df['position'] = new_pos
    agg_df = agg_df.drop(columns=['window_start'])
    
    if normalise:
        for s in agg_df.columns[1:]:
            agg_df[s] = agg_df[s]/window_size
    return agg_df

def delineate_region(start, length, svtype, binsize = 1000, extension = 3):
    if svtype == 'INS':
        e = start + length
        s = start - length
    else:
        e = start + length
        s = start
    
    s = int(s/binsize)*binsize - extension*binsize
    e = (int(e/binsize) + 1)*binsize + extension*binsize
    return s,e

def assess_bins(df, plausible_boundaries, map_t = 0.9, mq_t = 20, percent_windows_t = 0.25):
    start_idx, end_idx = plausible_boundaries
    metrics = df.copy()
    indices = np.where((metrics['mappability'] >= map_t) & (metrics['total:mean_mq'] >= mq_t))[0]
    metrics['valid'] = 0
    metrics.loc[indices, 'valid'] = 1
    
    tmp = metrics.loc[start_idx:end_idx,:]
    percent_windows = tmp['valid'].sum()/len(tmp)
    if percent_windows <= percent_windows_t:
        to_call = False
    else:
        to_call = True
    include_bins = np.where(metrics['valid'] == 1)[0]
    return to_call, include_bins

def extract_target_cov(df, start, end):
    df = df[(df['position'] >= start) & (df['position'] < end)].reset_index(drop = True)
    samples = [col for col in df.columns if col.endswith(":coverage") and col != "total:coverage"]
    coverage = df.loc[:,samples].replace("NA", 0).astype(float).to_numpy()
    samples = [s.split(':')[0] for s in samples]
    return samples, coverage

def normalise_by_flank_naive(df, start, end, flank, side = 'both'):
    fstart = max(start-flank, df.iloc[0,0])
    fend = min(end+flank, df.iloc[-1,0])
    
    left_flank = df[(df['position'] >= fstart) & (df['position'] < start)]
    right_flank = df[(df['position'] >= end) & (df['position'] < fend)]

    left_flank = left_flank[left_flank['total:coverage'] != 0] # Remove unaligned sites
    right_flank = right_flank[right_flank['total:coverage'] != 0]
    
    if side == 'both':
        cov = pd.concat([left_flank, right_flank], axis = 0)
        cov = cov.iloc[:,1:-1].to_numpy()
    elif side == 'left':
        cov = left_flank.iloc[:,1:-1].to_numpy()
    elif side == 'right':
        cov = right_flank.iloc[:,1:-1].to_numpy()
    else:
        raise ValueError("Unsupported side.")    
        
    means = np.mean(cov, axis = 0)
    variances = np.var(cov, axis = 0, ddof = 1)
    return means, variances

def normalise_by_flank_with_map(df, chromosome, start, end, flank, side = 'both', 
                        mapp = '/well/band/users/rbx225/recyclable_files/eichler_sv/mappability.bed.gz',
                        map_cutoff = 0.9):
    fstart = max(start-flank, df.iloc[0,0])
    fend = min(end+flank, df.iloc[-1,0])
    
    tsv1 = read_bed_tabix(mapp, chromosome, fstart, start)
    metrics1 = calculate_mappability_per_bin(tsv1, fstart, start)
    tsv2 = read_bed_tabix(mapp, chromosome, end, fend)
    metrics2 = calculate_mappability_per_bin(tsv2, end, fend)
    
    left_flank = df[(df['position'] >= fstart) & (df['position'] < start)]
    left_flank = pd.merge(left_flank, metrics1, on = ['position'])
    left_flank = left_flank[(left_flank['mappability'] >= map_cutoff) & (left_flank['total:coverage'] != 0)]
    right_flank = df[(df['position'] >= end) & (df['position'] < fend)]
    right_flank = pd.merge(right_flank, metrics2, on = ['position'])
    right_flank = right_flank[(right_flank['mappability'] >= map_cutoff) & (right_flank['total:coverage'] != 0)]
    
    if side == 'both':
        cov = pd.concat([left_flank, right_flank], axis = 0)
        cov = cov.iloc[:,1:-2].to_numpy()
    elif side == 'left':
        cov = left_flank.iloc[:,1:-2].to_numpy()
    elif side == 'right':
        cov = right_flank.iloc[:,1:-2].to_numpy()
    else:
        raise ValueError("Unsupported side.")    
        
    means = np.mean(cov, axis = 0)
    variances = np.var(cov, axis = 0, ddof = 1)
    return means, variances

def call_sv_samples(samples, genotypes):
    results = {}
    results[(0,0)] = []
    
    for i, g in enumerate(genotypes):
        if g in results.keys():
            results[g].append(samples[i])
        else:
            results[g] = [samples[i]]
    return dict(sorted(results.items()))

def generate_diploid_profiles(model):
    n_haps = len(model.haps)
    result = SVModel([], [])
    
    for i in range(n_haps):
        for j in range(i, n_haps):
            hap = model.haps[i] + model.haps[j]
            freq = model.freqs[i]*model.freqs[j]*(1+(i!=j))
            
            result.haps.append(hap)
            result.freqs.append(freq)
    result.normalise()
    return result

def run_inverse_length_penalty(hap):
    penalty = 0
    run_length = 1
    for i in range(1, len(hap)):
        if hap[i] == hap[i - 1]:
            run_length += 1
        else:
            penalty += 1 / run_length
            run_length = 1
    penalty += 1 / run_length
    return penalty