lcSV: low-coverage structural variant
=======================================

Low-coverage Structural Variant (lcSV) is a Python package designed to infer structural variants (SVs) (deletions and duplications) from low-coverage whole-genome sequencing (lcWGS) data. Instead of relying on high-depth read or split-read evidence, lcSV leverages genome-wide coverage patterns to detect variations in copy number. LcSV operates as a population-based method: it iteratively proposes, refines, and selects haplotypes with varying haplotype frequencies, building a reservoir that represents the genetic diversity of a studying cohort. Using these inferred haplotypes and their estimated frequencies, lcSV subsequently assigns individual genotypes for each sample. Because this inference depends on shared haplotype information across individuals, lcSV is not suitable for single-sample analysis and should be run on a cohort level, ideally with large populations where allele frequency estimates are solid.

Installation
-------------

Please install lcSV with the following command:

`pip install lcsv`

Running lcSV
-------------

Please see the example jupyter notebook provided under `examples`.

Dependencies
------------

LcSV supports Python 3.8+.

Installation requires [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [statsmodels](https://statsmodels.org/), and [scipy](https://scipy.org/).

Citation
------------

LcSV is not yet a published work, so citing the GitHub repository suffices. Please refer to the `CITATION.cff` file.

Development
-----------

See the main site of lcSV: https://github.com/Suuuuuuuus/lcSV.

Bugs shall be submitted to the [issue tracker](https://github.com/Suuuuuuuus/lcSV/issues). Please kindly provide a reproducible example demonstrating the problem.
