"""
constants.py

Description: Contains global constants used throughout project
"""

# Standard libraries
import os
from os.path import join, dirname


################################################################################
#                                  Debugging                                   #
################################################################################
DEBUG = True
SEED = 42

################################################################################
#                                  Data Paths                                  #
################################################################################
# Project directories to load data (images, metadata)
DIR_HOME = os.environ["HOME"]
DIR_PROJECT = dirname(dirname(__file__))

# Path to data directory, containing CXR data
DIR_DATA = join(DIR_PROJECT, "data")
DIR_DATA_MAP = {
    "save": join(DIR_DATA, "save_data"),
    "metadata": join(DIR_DATA, "metadata"),
    "vindr_cxr": join(DIR_DATA, "cxr_datasets", "vindr-cxr"),
    "vindr_pcxr": join(DIR_DATA, "cxr_datasets", "vindr-pcxr"),
}
DIR_METADATA_MAP = {
    "vindr_cxr": join(DIR_DATA_MAP["metadata"], "vindr_cxr_metadata.csv"),
    "vindr_pcxr": join(DIR_DATA_MAP["metadata"], "vindr_pcxr_metadata.csv"),
}

# Directory containing configurations
DIR_CONFIG = join(DIR_PROJECT, "config")
# Directory containing config specifications
DIR_CONFIG_SPECS = join(DIR_CONFIG, "configspecs")

# Local directory to save data (model weights, embeddings, figures)
DIR_SAVE_DATA = join(DIR_PROJECT, "data", "save_data")
DIR_TRAIN_RUNS = join(DIR_SAVE_DATA, "train_runs")
DIR_INFERENCE = join(DIR_SAVE_DATA, "inference")
DIR_EMBEDS = join(DIR_SAVE_DATA, "embeddings")
DIR_FIGURES = join(DIR_SAVE_DATA, "figures")

# Directory containing eda figures
DIR_FIGURES_EDA = join(DIR_FIGURES, "eda")
# Directory containing UMAP figures
DIR_FIGURES_UMAP = join(DIR_FIGURES, "umap")
# Directory containing GradCAM figures
DIR_FIGURES_CAM = join(DIR_FIGURES, "grad_cam")
# Directory containing prediction-related figures
DIR_FIGURES_PRED = join(DIR_FIGURES, "predictions")


################################################################################
#                                 Data Related                                 #
################################################################################
# Image size
IMG_SIZE = (512, 512)
