## DeepShape v2: Source separation for radio weak-lensing measurements using deep learning. 

Includes scripts for facet based image reconstruction and deep learning based optical image deblending.
DeepShape v2 is the wide-field version of [DeepShape](https://github.com/priyamvad-tripathi/DeepShape.git)

DeepShape is based on the findings presented in the following papers:
1. DeepShape2: Wide-field radio shear measurement using deep-learning: tbd
2. DeepShape: Radio Weak Lensing Shear Measurements using Deep Learning: [Tripathi et al (2025)](https://www.aanda.org/articles/aa/full_html/2025/04/aa54072-25/aa54072-25.html)
3. Shape measurement of radio galaxies using Equivariant CNNs: [Tripathi et al (2024)](https://ieeexplore.ieee.org/abstract/document/10715370)

## Installation
 
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)/[Miniconda](https://docs.anaconda.com/miniconda/install/)
2. Install git and clone this repository
  ````
  conda update conda
  conda install git
  git clone https://github.com/priyamvad-tripathi/DeepShape2.git
  ````
4. Create a conda environment and install all the required dependencies by running the following commands:
  ````
  cd DeepShape2/
  conda env create --name <env-name> --file environment.yml
  ````
5. Install the requirements and DeepShape2 files using pip:
   ````
   pip install -r requirements.txt
   pip install -e .
   
   ````

## Usage
### Dataset simulation
All the necessary scripts for simulating the training and testing datasets can be found in the [Simulation/](Simulation/) folder. Make sure to download all the FITS files containing the [T-RECS catalog](http://cdsarc.u-strasbg.fr/ftp/VII/282/fits/) and run the [make_catalog.py](Simulation/make_catalog.py) script to join all the FITS file into a single pandas dataframe containing only the required information. 
### Image Reconstruction
The [Reconstruction/](Reconstruction/) folder contains the scripts connected to image reconstruction using HQS-PnP algorithm. Make sure that the _DeepInverse_ library is correctly installed. By default, DRUNet is initialized using pre-trained weights from the library. This can be changed by setting the "pretrained" argument to a path containing the user-weights (see [PnP_tuning.py](Reconstruction/PnP_tuning.py) for details)
### Shape Measurement
The [Shape_Measurement/](Shape_Measurement/) folder contains the scripts connected to the shape measurement network. It also includes the scripts to perform shape measurements using [RadioLensfit](Shape_Measurement/RadioLensfit) and [SuperCALS](Shape_Measurement/SuperCALS) methods.
### Model Weights
DeepShape uses three networks: DRUNet denoiser, PSF autoencoder, and the shape measurement network. The trained weights for all three networks can be found at [OCA Cloud](https://cloud.oca.eu/index.php/s/KbMB8SbingdWibe).

## Cite
You can cite our work using the following $\BibTeX{}$ entry:
 ````
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
````

Feel free to [contact us](mailto:priyamvad.tripathi@oca.eu).

