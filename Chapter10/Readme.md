# Material for Chapter 10

The notebook for the exercise in Chapter 10 is available from here:
[Notebook for mini U-net](https://github.com/vedranaa/teaching-notebooks/blob/main/02506_week10_MiniUnet.ipynb)

Training the network may take long time, especially without GPU. We therefore sketch several ways of running the notebook with GPU support.

## Run the notebook on Google Colab

You can open the notebook in Google Colab from here, and enable GPU :
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vedranaa/teaching-notebooks/blob/main/02506_week10_MiniUnet.ipynb)


## Run the notebook on DTU Gbar

You can run the notebook on Gbar on an interactive GPU node. 

First get to a terminal on the cluster, either with ssh, ThinLinc client, or a browser-based ThinLinc [thinlinc.gbar.dtu.dk](thinlinc.gbar.dtu.dk). Open the terminal (command line).

Log onto an interactive GPU node such as `voltash` by running the following command on the command line:

`voltash -X`

To set up a Python environment, you can use the script `env02506.sh` which we provided for you. You should place the file `env02506.sh` in a folder on the Gbar and then run the command:

`source env02506.sh`

The first time you run this, the script will create a new python environment called `env02506` and activate it.

Then you should install the packages:

```
pip install torch torchvision
pip install Pillow
pip install notebook
```

Now you are good to go. You can navigate to the folder that you wish to store the code, download the notebook from the link above, and open a jupyter notebook by typing the following in the command line:

`jupyter-notebook`

## Use QIM platform to access GBar

We are working on establishing an easier way of using Gbar functionality which does the steps described above automatically. This is still experimental.

First go to QIM platform [https://platform.qim.dk/](https://platform.qim.dk/) and log on using your DTU credentials.

Under Tools menu, choose Jupyter launcher. In the new window yype in your DTU credentials (needs to be done twice). Under basic config QIM environment choose `qim3d` (it has the largest number of packages installed). Under advanced config choose HPC queue called `gpuqim`. Tick the checkbox Reset SSH tunnel. Click the button `Start jupyter server`. 

The platform shold launch your job and a button with `Open Jupyter` should appear, leading you to a jupyter server running in your home directory on the Gbar. 


