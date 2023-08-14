# TEBM install requirements
# Linux OS (Ubuntu 16.04.1, or greater)
# g++-7.5.0
# c++-3.8.0
# python-3.7
# numpy-1.19.5
# scipy-1.7.3
# pandas
# pickle
# sklearn
# matplotlib
# install and link "kde_ebm" package, available here: https://github.com/ucl-pond/kde_ebm
# navigate to top directory and issue the following command

CC=g++ CFLAGS=-lstdc++ python setup.py install

# after install, navigate to "examples/" directory and issue the following command

python run_tebm_trackhd_predicthd.py