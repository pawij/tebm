# Temporal Event-Based Model (TEBM)
The TEBM is a generative model that can estimate the timing and uncertainty of events from semi-longitudinal datasets with irregularly sampled and missing data.

If you use the TEBM, please cite this paper:

Wijeratne, P.A., Eshaghi, A., Scotton, W.J., et al. The temporal event-based model: learning event timelines in progressive diseases. Imaging Neuroscience 2023. doi: https://doi.org/10.1162/imag_a_00010

# TEBM install requirements
Linux OS (Ubuntu 16.04.1, or greater)
g++>=7.5.0  
c++>=3.8.0  
python>=3.7  
numpy>=1.19.5  
scipy>=1.7.3  
pandas  
pickle
pathos
sklearn  
matplotlib

Install and link "kde_ebm" package, available here:

https://github.com/ucl-pond/kde_ebm

Navigate to top directory and issue the following command:

CC=g++ CFLAGS=-lstdc++ python setup.py install

# Running the code
Navigate to examples/ and issue the following command:

python run_tebm_sim.py

# Worked example
To follow...