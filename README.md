# DeepShape II  
## Source separation for radio weak-lensing measurements using deep learning

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

**DeepShape II** is a wide-field extension of the original [DeepShape](https://github.com/priyamvad-tripathi/DeepShape.git) framework, providing tools for facet-based source separation, plug-and-play radio image deconvolution, deep learning-based deblending, and shape measurement, with the goal of enabling accurate and scalable weak-lensing measurements in crowded radio fields.

---

## Method Overview

- End-to-end pipeline from visibilities to shape estimates  
- Parallelised source isolation in the visibility domain via faceting  
- Plug-and-play image deconvolution using a trained DRUNet denoiser  
- VAE-based network for source separation in the image domain  
- Equivariant CNN–based shape measurement  

For full methodological details, please refer to the original paper.

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
git clone https://github.com/priyamvad-tripathi/DeepShape2.git
cd DeepShape2
```

### 2. Create environment
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
  implementation/        Application to real/simulated data
  extra/                 Utility scripts and examples
results/                 Reproduced results from the paper
```

---

## Quick Start

Run the example pipeline:

```bash
python scripts/extra/timing.py
```

This demonstrates:
- Model loading  
- Visibility handling  
- Faceting and reconstruction  
- Deblending  
- Shape measurement  

---

## Configuration

Example configuration:

```yaml
data:
  dataset_path: /path/to/data

model:
  weights_path: /path/to/weights
```

Main config file:
```
deepshape2/config/default.yaml
```

---

## Data and Pretrained Models

- Pretrained weights:  
  https://cloud.oca.eu/index.php/s/N7t7NyYCTMjK5XA  

- Datasets available upon request  

---

## Citation

If you use this repository, please cite:

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

For questions or dataset access:  
[priyamvad.tripathi@oca.eu](mailto:priyamvad.tripathi@oca.eu)
