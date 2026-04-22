# DeepShape II  
## Source separation for radio weak-lensing measurements using deep learning

![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

**DeepShape II** is a wide-field extension of the original [DeepShape](https://github.com/priyamvad-tripathi/DeepShape.git) framework, providing tools for shape measurement of radio galaxies from noisy wide field visibility measurements.

## Overview

DeepShape II implements an end-to-end pipeline for shape measurement in wide-field radio observations, going from visibilities to final shape estimates. The framework combines:

- Parallelised source isolation in the visibility domain via faceting  
- Plug-and-play image deconvolution using a trained DRUNet denoiser  
- VAE-based network for source separation in the image domain  
- Equivariant CNN–based shape measurement  


In addition to the core methodology, this repository also provides scripts to simulate radio datasets based on the TRECS catalog.

The implementation builds on the methodology introduced in the following works:

1. **DeepShape II**: Wide-field radio shear measurement using deep learning *(in preparation; please search for latest version)*  

2. **DeepShape**: Radio Weak Lensing Shear Measurements using Deep Learning  
   [Tripathi et al (2024)](https://ieeexplore.ieee.org/abstract/document/10715370)

3. Shape measurement of radio galaxies using Equivariant CNNs  
  [Tripathi et al (2025)](https://www.aanda.org/articles/aa/full_html/2025/04/aa54072-25/aa54072-25.html)

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
- Timing each part
---

# Configuration

An example configuration is shown below:

```yaml

LOCAL_DIR: /path/to/data/
```
By default, the pipeline assumes the following directory structure:

* LOCAL_DIR/Data/ for datasets
* LOCAL_DIR/Model_weights/ for pretrained model weights

The main configuration file is located at:
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
