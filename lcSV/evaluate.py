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

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from .model import *
from .process import *

__all__ = ["multi_evaluate_model", "evaluate_per_hap", "normalise_ll", "get_best_haps", "evaluate_sim_model", "evaluate_real_model"]

def multi_evaluate_model(model, N, pre_computed_lls, L = None, bin_size = 1000):
    n_haps = len(model.haps)
    if L is None:
        L = len(model.haps[0])
    n_diploid = int(n_haps*(n_haps + 1)/2)
    
    dip_model = generate_diploid_profiles(model)
    final_lls = np.zeros((N, n_diploid))
    
    for i in range(n_diploid):
        ll_ary = evaluate_per_hap(dip_model.haps[i], pre_computed_lls)
        final_lls[:, i] = ll_ary
        
    trans_penalty = 0
    for h in model.haps:
        trans_penalty += run_inverse_length_penalty(h)

    final_lls = final_lls + np.log(np.array(dip_model.freqs))[np.newaxis, :] 
    model_ll = logsumexp(final_lls, axis=1).sum()
    model_ll = ((n_haps-1)*L + trans_penalty)*np.log(N*L) - 2*model_ll
    return model, final_lls, model_ll

def evaluate_per_hap(hap, pre_computed_lls):
    L, N = pre_computed_lls[0].shape
    
    result = np.zeros(N)
    for i in range(L):
        cn = int(hap[i])
        block = pre_computed_lls[cn]
        result = result + block[i,:]
    return result

def normalise_ll(lls, min_prob=1e-6):
    probs = np.exp(lls - logsumexp(lls, axis=1, keepdims=True))
    
    probs[probs < min_prob] = 0.0
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

def get_best_haps(model, probs):
    N, K = probs.shape
    n_haps = len(model.haps)
    keys = {}
    count = 0
    for i in range(n_haps):
        for j in range(i, n_haps):
            keys[count] = (i, j)
            count += 1
    
    indices = probs.argmax(axis = 1)
    results = [keys[i] for i in indices]
    return results


def evaluate_sim_model(result_dict, true_hap, true_gt):
    freq = 0
    concordance = 0
    info = 0
    
    probs = result_dict['probs']
    genotypes = result_dict['genotypes']
    best_model = result_dict['model_ary'][-1]
    N = len(genotypes)
    
    if (len(best_model.haps) != 2) or all(best_model.haps[1] != true_hap):
        pass

    else:
        freq = best_model.freqs[1]
        
        dosage_ary = []
        dosage2_ary = []

        for i in range(N):
            h1, h2 = true_gt[i]
            t1, t2 = genotypes[i]
            concordance += (max(((h1 == t1) + (h2 == t2)), ((h1 == t2) + (h2 == t1))))
            
            p = probs[i]
            dosage_ary.append(p[1] + 2*p[2])
            dosage2_ary.append(p[1] + 4*p[2])
        
        concordance = concordance/N
        dosage_ary = np.array(dosage_ary)
        dosage2_ary = np.array(dosage2_ary)
        maf = np.mean(dosage_ary)/2
        info = 1 - np.mean((dosage2_ary - dosage_ary**2)/(2*maf*(1-maf)))
    return concordance, info, freq

def evaluate_real_model(result_dict, plausible_boundaries, svtype, include_bins, n_bins):
    freqs = []
    infos = [1]
    concordance = 0
    
    probs = result_dict['probs']
    genotypes = result_dict['genotypes']
    best_model = result_dict['model_ary'][-1]
    N = len(genotypes)
    
    true_hap = None
    true_hap_idx = -9
        
    if len(best_model.haps) == 1:
        freqs = [1]
    else:
        for i, h in enumerate(best_model.haps):
            start_idx, end_idx = plausible_boundaries
            true_hap_ind = compare_plausible_sv(h, start_idx, end_idx, include_bins, n_bins)
            
            freqs.append(best_model.freqs[i])
            if i != 0:
                infos.append(compute_multiallelic_info(probs, i))
            
            if check_sv(true_hap_ind, h, svtype, plausible_boundaries, include_bins):
                true_hap = h
                true_hap_idx = i
                concordance = 1
    
    return infos, freqs, concordance, true_hap_idx

def check_sv(true_hap_ind, h, svtype, plausible_boundaries, include_bins):
    start_idx, end_idx = plausible_boundaries
    called_bins = np.where((include_bins >= start_idx) & (include_bins < end_idx))[0].size
    c1 = true_hap_ind
    c2 = (np.where(h != 1)[0].size <= called_bins + 1)
    if svtype == 'INS':
        c2p = ((h[h > 1] - 1).sum() <= called_bins + 1)
    else:
        c2p = False
    c3 = ((svtype == 'INS' and np.any(h >= 2)) or (svtype == 'DEL' and np.any(h == 0)))
    c4 = (not np.any(h == 9))
    return c1 and (c2 or c2p) and c3 and c4

def compare_plausible_sv(arr, start_idx, end_idx, include_bins, n_bins):
    full = np.full(n_bins, None)
    for i, b in enumerate(include_bins):
        full[b] = arr[i]

    sub_bins = np.arange(start_idx, end_idx)
    sub_vals = [full[b] for b in sub_bins if full[b] is not None]

    if not sub_vals:
        return False

    non1 = np.array(sub_vals) != 1
    padded = np.r_[False, non1, False]
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where(padded[:-1] & ~padded[1:])[0]

    return len(starts) == 1


def compute_multiallelic_info(GP, alt_index):
    N, K = GP.shape
    A = int((np.sqrt(8 * K + 1) - 1) / 2)
    if A * (A + 1) // 2 != K:
        raise ValueError("Invalid number of genotype columns in GP.")
    if alt_index < 1 or alt_index >= A:
        raise ValueError(f"alt_index must be in [1, {A-1}] for {A} alleles.")

    genotypes = list(combinations_with_replacement(range(A), 2))
    alt_copy_per_genotype = np.array([g.count(alt_index) for g in genotypes])

    dosage = GP @ alt_copy_per_genotype
    dosage2 = GP @ (alt_copy_per_genotype ** 2)

    var_g = np.mean(dosage2 - dosage**2)
    p = np.mean(dosage) / 2

    if p == 0 or p == 1:
        return 0.0

    info = 1 - var_g / (2 * p * (1 - p))
    return info