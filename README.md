# ReMO_Training_Course
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/grgdll/ReMO_Training_Course/main?labpath=cmp_respiration_ReMO.ipynb)

This notebook demonstrates how to estimate mesopelagic respiration from oxygen data collected by BGC-Argo floats.

To work with these examples you first need to clone this repository. Clone this repository in a new directory (e.g., click [here](https://github.com/grgdll/ReMO_Training_Course/archive/refs/heads/main.zip)).

Unzipping this file should create on your computer the following directory structure:
<pre>
ReMO_Training_Course-main/
├── environment.yml
├── cmp_respiration_ReMO.ipynb
├── bgc_tools.py
├── README.md
└── LICENSE
</pre>

The file `environment.yml` allows you to recreate the conda environment needed for the notebooks to work. If you do not have conda installed, follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To create the `R_training` environment use the command `conda env create -f environment.yml`. See also [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more detailed instructions on how to create a conda environment from a yml file.

Then the newly created conda environment needs to be activated using `conda activate R_training`.
