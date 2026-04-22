# DeepShape II  
## Source separation for radio weak-lensing measurements using deep learning

**DeepShape II** is a wide-field extension of the original [DeepShape](https://github.com/priyamvad-tripathi/DeepShape.git) framework. It provides tools for facet-based radio image reconstruction and deep learning-based optical deblending, enabling accurate weak-lensing shape measurements in radio surveys.

---

## Overview

This repository contains:
- Simulation pipelines for radio datasets  
- Facet-based image reconstruction methods  
- Deep learning models for deblending and shape measurement  
- Scripts for applying the pipeline to real observations  

The implementation builds on the methodology introduced in the following works:

1. **DeepShape II**: Wide-field radio shear measurement using deep learning *(in preparation; please search for latest version)*  
2. **DeepShape**: Radio Weak Lensing Shear Measurements using Deep Learning  
   Tripathi et al. (2025)  
   https://www.aanda.org/articles/aa/full_html/2025/04/aa54072-25/aa54072-25.html  
3. Shape measurement of radio galaxies using Equivariant CNNs  
   Tripathi et al. (2024)  
   https://ieeexplore.ieee.org/abstract/document/10715370  

---

## Installation

### 1. Clone the repository
```bash
conda update conda
conda install git
git clone https://github.com/priyamvad-tripathi/DeepShape2.git
cd DeepShape2
```

### 2. Create the environment
```bash
conda env create --name <env-name> --file environment.yml
conda activate <env-name>
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## Repository Structure

```
deepshape2/              Core library (models, configs, utilities)
scripts/
  simulations/           Dataset generation scripts
  implementation/        Application to real data
  extra/                 Utility scripts and examples
results/                 Reproduced results from the paper
```

---

## Data and Pretrained Models

- Pretrained model weights:  
  https://cloud.oca.eu/index.php/s/N7t7NyYCTMjK5XA  

- Datasets used in this work are available upon request  

After downloading, update paths in:
```
deepshape2/config/default.yaml
```

---

## Quick Start

A minimal end-to-end example is provided in:

```
scripts/extra/timing.py
```

This script demonstrates:
- Loading trained models  
- Working with visibility data  
- Faceting and reconstruction  
- Deblending  
- Shape measurement  

---

## Citation

If you use this repository, please cite the original DeepShape paper:

```bibtex
@ARTICLE{deepshape25,
  author = {{Tripathi}, P. and {Wang}, S. and {Prunet}, S. and {Ferrari}, A.},
  title = "{DeepShape: Radio weak-lensing shear measurements using deep learning}",
  journal = {\aap},
  year = 2025,
  month = apr,
  volume = {696},
  eid = {A216},
  pages = {A216},
  doi = {10.1051/0004-6361/202554072},
}
```

For **DeepShape II**, please search for the latest publication and cite the most recent version.

---

## Contact

For questions or dataset access, please contact:  
[priyamvad.tripathi@oca.eu](mailto:priyamvad.tripathi@oca.eu)
