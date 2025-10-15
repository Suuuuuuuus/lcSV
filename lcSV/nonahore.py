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

from .model import *
from .compute import *
from .evaluate import *

__all__ = ["lcSV_core"]



def lcSV_core(means,  # An np.array of length N of non-SV mean coverage
             variances,  # An np.array of length N of non-SV variance coverage
             covs,  # An np.array of shape (LxN) of SV region coverage
             ploidy = 2, 
             n_iter = 100,  # Maximum number of iterations allowed
             n_sample_freq = 200,  # Number of new frequency sampled in each iter
             n_recomb = 1000,  # Number of new haplotype sampled in each iter
             bin_size = 1000,
             max_cnv = 10,
             xy = 0.01,
             verbose = False,
             adaptive = False,
             premature = True):
    pre_computed_lls = precompute_site_lls(means/ploidy, variances/ploidy, covs)

    L, N = covs.shape

    reference_model = SVModel([np.ones(L)], [1])
    best_model = copy.deepcopy(reference_model)

    best_model_ary = []
    best_ll_ary = []
    best_penalty_ary = []

    for iteration in range(n_iter):
        if adaptive:
            this_sample_freq = min(10*iteration, n_sample_freq)
            this_recomb = max(n_recomb - 50*iteration, int(n_recomb/2))
        else:
            this_sample_freq = n_sample_freq
            this_recomb = n_recomb
        
        reference_model = copy.deepcopy(best_model)
        old_haps_set = set()
        models = [reference_model]

        if len(reference_model.haps) > 1:
            for j in range(1, len(reference_model.haps)):
                model = copy.deepcopy(reference_model)
                model.haps = model.haps[:j] + model.haps[(j+1):]
                model.freqs = model.freqs[:j] + model.freqs[(j+1):]
                model.normalise()
                models.append(model)

        if len(reference_model.haps) > 1:
            for _ in range(this_sample_freq):
                model = copy.deepcopy(reference_model)
                model.freqs = dirichlet_sampling(model.freqs)
                model.normalise()
                if sum(np.array(model.freqs) >= xy) > 1:
                    models.append(model)

        for _ in range(this_recomb):
            new_hap = sample_recombinants(reference_model, L)
            if new_hap.tobytes() in old_haps_set:
                pass
            else:
                old_haps_set.add(new_hap.tobytes())
                for f in np.arange(1,20)*0.05:
                    model = copy.deepcopy(reference_model)
                    model.add(new_hap, f)
                    models.append(model)

                if len(reference_model.haps) > 1:
                    for ix in range(1, len(reference_model.haps)):
                        model = copy.deepcopy(reference_model)
                        model.replace(new_hap,ix)
                        models.append(model)

        _, _, ref_model_ll = multi_evaluate_model(reference_model, N, pre_computed_lls)

        best_model = copy.deepcopy(reference_model)
        best_model_ll = ref_model_ll
        
        if sys.platform.startswith("linux"):
            ncores = max(2*(len(os.sched_getaffinity(0))) - 1, 1)
        else:
            ncores = max(2*(multiprocessing.cpu_count()-1) - 1, 1)

        with multiprocessing.Pool(processes=ncores) as pool:
            results = pool.starmap(
                multi_evaluate_model,
                [(m, N, pre_computed_lls) for m in models]
            )

        for res in results:
            m, _, model_ll = res
            if model_ll < best_model_ll:
                best_model = sort_model(copy.deepcopy(m))
                best_model_ll = model_ll
                reference_model = copy.deepcopy(m)

        best_model_ary.append(reference_model)
        best_ll_ary.append(best_model_ll)
 
        if verbose and ((iteration + 1) % 5 == 0):
            print(f'------ Iteration {iteration + 1} ------')
            print(f'Best loglikelihood: {best_ll_ary[iteration]}')
            
        if len(best_ll_ary) >= 100 and len(set(best_ll_ary[-100:])) == 1 and premature:
            break
    
    _, lls, _ = multi_evaluate_model(best_model, N, pre_computed_lls)
    best_probs = normalise_ll(lls)
    best_genotypes = get_best_haps(best_model, best_probs)
    
    results = {}
    results['model_ary'] = best_model_ary
    results['ll_ary'] = best_ll_ary
    results['probs'] = best_probs
    results['genotypes'] = best_genotypes
    return results

def dirichlet_sampling(freqs, concentration = 10, limit = 1e-4):
    alpha = np.array(freqs) * concentration
    alpha[alpha <= limit] = limit
    rng = default_rng()
    res = rng.dirichlet(alpha)
    res[res <= limit] = limit
    return list(res)

def sample_from_freqs(freqs):
    freqs = np.array(freqs)
    cum_probs = np.cumsum(freqs)
    u = np.random.uniform(0, 1)
    return np.searchsorted(cum_probs, u)

def sample_recombinants(model, L, max_cnv = 10):
    hap1 = sample_from_freqs(model.freqs)
    hap2 = sample_from_freqs(model.freqs)
    breakpoints = np.random.choice(np.arange(1, L), size=2, replace=True)
    
    left = model.haps[hap1].copy()
    left[breakpoints[0]:] = 0
    right = model.haps[hap2].copy()
    right[:breakpoints[1]] = 0

    new_hap = left + right
    
    if np.any(new_hap >= max_cnv):
        new_hap = np.ones(L)
    return new_hap