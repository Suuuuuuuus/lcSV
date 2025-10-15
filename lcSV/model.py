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

__all__ = ["SVModel", "sort_model"]

class SVModel:
    def __init__(self, haps=None, freqs=None):
        self.haps = haps
        self.freqs = freqs
    
    def __repr__(self):
        return str(len(self.haps))
    
    def normalise(self):
        tmp = np.array(self.freqs)
        self.freqs = list(tmp/tmp.sum())
    
    def add(self, hap, freq):
        self.haps.append(hap)
        self.freqs = list((1-freq)*np.array(self.freqs))
        self.freqs.append(freq)
        self.normalise()
        
    def replace(self, hap, ix):
        self.haps[ix] = hap


def sort_model(model):
    if len(model.haps) == 1:
        return model
    else:
        haps = model.haps[1:]
        freqs = model.freqs[1:]

        sorted_haps = [hap for hap, _ in sorted(zip(haps, freqs), key=lambda x: x[1], reverse=True)]
        sorted_freqs = [freq for _, freq in sorted(zip(haps, freqs), key=lambda x: x[1], reverse=True)]
        
        new_haps = [model.haps[0]] + sorted_haps
        new_freqs = [model.freqs[0]] + sorted_freqs
    
    return SVModel(new_haps, new_freqs)