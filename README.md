# ReMO_Training_Course

This notebook demonstrates how to estimate mesopelagic respiration from oxygen data collected by BGC-Argo floats.

To work with this notebook, you first need to clone the current repository in a new directory (e.g., click [here](https://github.com/grgdll/ReMO_Training_Course/archive/refs/heads/main.zip)).

This cloning should should create on your computer the following directory structure:
<pre>
ReMO_Training_Course-main/
├── environment.yml
├── cmp_respiration_ReMO.ipynb
├── bgc_tools.py
├── README.md
└── LICENSE
</pre>

The file `environment.yml` allows you to recreate the conda environment (i.e., Python and related libraries) needed for the notebook to work. If you do not have conda installed, first follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

To create the `R_training` environment open a terminal (for Windows users, please use the "Anaconda Powershell Prompt"), move to the `ReMO_Training_Course-main` directory (i.e., use command `cd` followed by the desired path) and type this command: `conda env create -f environment.yml`. See also [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more detailed instructions on how to create a conda environment from a yml file. This command will download and install the correct version of Python and the libraires needed to run the notebook: please be patient, it may take a relatively long time. 

Then, you need to activate the new conda environment. From the terminal type this command: `conda activate R_training`

Finally, to start the Notebook, type this command on your terminal: `jupyter notebook cmp_respiration_ReMO.ipynb`
A window should open on your browser with the Jupyter Notebook ready.
