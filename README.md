# ReMO_Training_Course
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/grgdll/ReMO_Training_Course/main?labpath=cmp_respiration_ReMO.ipynb)

This notebook demonstrates how to estimate mesopelagic respiration from oxygen data collected by BGC-Argo floats.

To work with this notebook, you first need to clone the current repository in a new directory (e.g., click [here](https://github.com/grgdll/ReMO_Training_Course/archive/refs/heads/main.zip)).

Unzipping this file should create on your computer the following directory structure:
<pre>
ReMO_Training_Course-main/
├── environment.yml
├── cmp_respiration_ReMO.ipynb
├── bgc_tools.py
├── README.md
└── LICENSE
</pre>

The file `environment.yml` allows you to recreate the conda environment needed for the notebook to work. If you do not have conda installed, follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

To create the `R_training` environment use the command `conda env create -f environment.yml`. See also [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more detailed instructions on how to create a conda environment from a yml file. This command will download and install the correct version of Python and the libraires needed to run the Notebook: please be patient, it may take a relatively long time. 

Then the newly created conda environment needs to be activated: from a terminal (for Windows users, please use the "Anaconda Powershell Prompt") type this command `conda activate R_training`.

Finally, to start the Notebook, type this command on your terminal `jupyter notebook cmp_respiration_ReMO.ipynb`, a windows on your browser should open with the Jupyter Notebook ready.
