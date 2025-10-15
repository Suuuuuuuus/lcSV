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

from scipy.stats import nbinom
from scipy.stats import geom, beta

__all__ = ["simulate_coverage", "simulate_coverotron"]

def simulate_coverage(L, Ecov, Dcov, binsize=1000, method="negative_binomial"):
    Ecov = np.where(Ecov * binsize == 0, 1e-3, Ecov * binsize)
    Dcov = np.where(Dcov * binsize == 0, 1e-2, Dcov * binsize)

    if method == "negative_binomial":
        size = (Ecov**2) / (Dcov - Ecov)
        prob = Ecov / Dcov
        return nbinom.rvs(size, prob, size=L)
    elif method == "normal":
        return np.maximum(norm.rvs(loc=Ecov, scale=np.sqrt(Dcov) + 0.05, size=S), 0)
    elif method == "poisson":
        return poisson.rvs(mu=Ecov, size=L)
    else:
        raise ValueError("Select from: 1.normal; 2.negative_binomial; 3.poisson.")

def simulate_coverotron(model, N, L, mean_coverage, sd_coverage, nb_var, binsize = 1000, model_sim = 'negative_binomial', ploidy = 2):
    cumulative = np.cumsum(np.concatenate([[0], model.freqs]))

    means = np.random.normal(loc=mean_coverage, scale=sd_coverage, size=N)
    variances = nb_var * means

    random_vals = np.random.rand(ploidy * N)
    hap_indices = [np.where(cumulative < val)[0][-1] for val in random_vals]
    haplotypes = np.array(hap_indices).reshape(N, ploidy)

    profiles = np.full((L, N), np.nan)
    coverage = np.full((L, N), np.nan)
    training = np.full((L, N), np.nan)

    for i in range(N):
        training[:, i] = simulate_coverage(L, means[i] * ploidy, variances[i] * ploidy, binsize, method=model_sim)

        profile_1 = model.haps[haplotypes[i, 0]]
        profile_2 = model.haps[haplotypes[i, 1]]
        profiles[:, i] = profile_1 + profile_2

        Ecov = means[i] * profiles[:, i]
        Dcov = variances[i] * profiles[:, i]
        coverage[:, i] = simulate_coverage(L, Ecov, Dcov, binsize, method=model_sim)

    haplotypes = haplotypes.astype(int)
    haplotypes = [tuple(sorted(i)) for i in haplotypes]
    return training, coverage, haplotypes