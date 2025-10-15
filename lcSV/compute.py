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

from scipy.stats import nbinom

from .model import *

__all__ = [
    "precompute_site_lls"
]

def precompute_site_lls(means, variances, coverage, max_cnv=20, mismap_proportion = 0.01, model='negative_binomial'):
    L, N = coverage.shape
    site_factor = np.ones(L)
    mismap_propn = np.full(L, mismap_proportion)

    LOG_TWO_PI = np.log(2 * np.pi)
    
    lls = []

    for cn in range(max_cnv):
        mean_term = (
            (means * cn)[np.newaxis, :] * site_factor[:, np.newaxis] +
            means[np.newaxis, :] * mismap_propn[:, np.newaxis]
        )

        if model == 'normal':
            var_term = (
                (variances * cn)[np.newaxis, :] * site_factor[:, np.newaxis] +
                variances[np.newaxis, :] * mismap_propn[:, np.newaxis]
            )
            var_term = np.maximum(var_term, 1e-10)
            residuals = coverage - mean_term
            log_likelihood = (
                -0.5 * LOG_TWO_PI
                - 0.5 * np.log(var_term)
                - 0.5 * (residuals ** 2) / var_term
            )

        elif model == 'negative_binomial':
            mu = mean_term
            var = (
                (variances * cn)[np.newaxis, :] * site_factor[:, np.newaxis] +
                variances[np.newaxis, :] * mismap_propn[:, np.newaxis]
            )
            var = np.maximum(var, mu + 1e-6)
            r = (mu ** 2) / (var - mu)
            p = r / (r + mu)
            x = np.clip(np.round(coverage), 0, None).astype(int)
            log_likelihood = nbinom.logpmf(x, r, p)

        else:
            raise ValueError("Unsupported model. Choose from: 1.normal; 2.negative_binomial.")
            
        lls.append(log_likelihood)
    return lls