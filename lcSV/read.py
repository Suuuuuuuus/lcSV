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

import pandas as pd

from .model import *

__all__ = ["read_tabix", "read_bed_tabix", "read_bam", "read_pickle", "read_coverage_data", "load_region_files"]

def read_tabix(vcf, c, regstart, regend):
    command = f"tabix {vcf} chr{c}:{regstart}-{regend}"
    seqs = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    if seqs == ['']:
        return pd.DataFrame()

    seqs = [i.split('\t') for i in seqs if '##' not in i]
    seqs = pd.DataFrame(seqs).iloc[:, :5]
    seqs.columns = ['chr', 'pos', 'id', 'ref', 'alt']
    return seqs

def read_bed_tabix(vcf, c, regstart, regend):
    command = f"tabix {vcf} chr{c}:{regstart}-{regend}"
    seqs = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    if seqs == ['']:
        return pd.DataFrame()

    seqs = pd.DataFrame([i.split('\t') for i in seqs if '##' not in i], 
                        columns=["chr", "start", "end", "value"]).astype({"start": int, "end": int, "value": float})
    
    if seqs.loc[0, 'start'] < regstart:
        seqs.loc[0, 'start'] = regstart
    elif seqs.loc[0, 'start'] > regstart:
        new_row = {'chr': c, 'start': regstart, 'end': seqs.loc[0, 'start'], 'value': 0}
        seqs = pd.concat([pd.DataFrame([new_row]), seqs], ignore_index=True).reset_index(drop = True)
    else:
        pass

    if seqs.loc[len(seqs)-1, 'end'] > regend:
        seqs.loc[len(seqs)-1, 'end'] = regend
    elif seqs.loc[len(seqs)-1, 'end'] < regend:
        new_row = {'chr': c, 'start': seqs.loc[len(seqs)-1, 'end'], 'end': regend, 'value': 0}
        seqs = pd.concat([seqs, pd.DataFrame([new_row])], ignore_index=True).reset_index(drop = True)
    else:
        pass
    
    filled = []
    prev_end = regstart
    for _, row in seqs.iterrows():
        if row["start"] > prev_end:
            filled.append([prev_end, row["start"], 0.0])
        filled.append([row["start"], row["end"], row["value"]])
        prev_end = row["end"]

    if prev_end < regend:
        filled.append([prev_end, regend, 0.0])
        
    filled = pd.DataFrame(filled, columns = seqs.columns[1:])
    filled['chr'] = c
    filled = filled[['chr'] + seqs.columns[1:].tolist()]
    return filled

def read_bam(bam, c, regstart, regend):
    command = f"samtools view {bam} chr{c}:{regstart}-{regend}"
    reads = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    if reads == ['']:
        return pd.DataFrame()

    reads = [i.split('\t') for i in reads if '##' not in i]
    reads = pd.DataFrame(reads)
    reads.columns = [
        "ID", "flag", "chr", "pos", "map_quality", "CIGAR", "chr_alt", "pos_alt", "insert_size", "sequence", "base_quality"
    ] + [f'col{i}' for i in range(11, reads.shape[1])]

    reads['pos'] = reads['pos'].astype(int)
    reads['pos_alt'] = reads['pos_alt'].astype(int)
    return reads

def read_pickle(infile):
    with open(infile, 'rb') as f:
        data = pickle.load(f)
    return data

def read_coverage_data(file_path, sep = ','):    
    df = pd.read_csv(file_path, sep = sep)
    bins = list(zip(df['position'], df['position'] + df['size'], df['N']))
    samples = [col for col in df.columns if col.endswith(":coverage") and col != "total:coverage"]
    coverage = df.loc[:,samples].replace("NA", 0).astype(float).to_numpy()
    samples = [s.split(':')[0] for s in samples]
    return samples, bins, coverage

def load_region_files(chunks, chromosome, start, end, indir = 'results/coverage/coverotron/', flank = None):
    chromosome = str(chromosome)
    if flank is not None:
        start = start - flank
        end = end + flank
        
    starts = chunks[chromosome]['start']
    ends = chunks[chromosome]['end']
    
    dfs = []
    for s, e in zip(starts, ends):
        if not (e < start or s > end):  
            file_path = os.path.join(indir, f'chr{chromosome}.{s}.{e}.tsv.gz')
            if os.path.exists(file_path):
                dfs.append(pd.read_csv(file_path, sep='\t', compression = None))
            else:
                print(f"Warning: File {file_path} not found.")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No overlapping files found for the specified region.")