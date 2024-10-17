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
DIR_PROJECT = dirname(dirname(dirname(__file__)))

# Local directory to save data (model weights, embeddings, figures)
DIR_SAVE = join(DIR_PROJECT, "src", "data")
DIR_TRAIN_RUNS = join(DIR_SAVE, "train_runs")
DIR_INFERENCE = join(DIR_SAVE, "inference")
DIR_EMBEDS = join(DIR_SAVE, "embeddings")
DIR_FIGURES = join(DIR_SAVE, "figures")

# Directory containing eda figures
DIR_FIGURES_EDA = join(DIR_FIGURES, "eda")
# Directory containing UMAP figures
DIR_FIGURES_UMAP = join(DIR_FIGURES, "umap")
# Directory containing GradCAM figures
DIR_FIGURES_CAM = join(DIR_FIGURES, "grad_cam")
# Directory containing prediction-related figures
DIR_FIGURES_PRED = join(DIR_FIGURES, "predictions")

# Directory containing configurations
DIR_CONFIG = join(DIR_PROJECT "config")
# Directory containing config specifications
DIR_CONFIG_SPECS = join(DIR_CONFIG, "configspecs")

