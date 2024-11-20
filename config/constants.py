"""
constants.py

Description: Contains global constants used throughout project
"""

# Standard libraries
import os
from os.path import join, dirname


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
    "nih_cxr18": join(DIR_DATA, "cxr_datasets", "NIH"),
    "padchest": join(DIR_DATA, "cxr_datasets", "PC"),
    "chexbert": join(DIR_DATA, "cxr_datasets", "chexbert"),
}
DIR_METADATA_MAP = {
    # VinDr-CXR
    "vindr_cxr": {
        "dicom": join(DIR_DATA_MAP["metadata"], "raw", "vindr_cxr-dicom_metadata.csv"),
        "image": join(DIR_DATA_MAP["metadata"], "vindr_cxr_metadata.csv"),
    },
    # VinDr-PCXR
    "vindr_pcxr": {
        "dicom": join(DIR_DATA_MAP["metadata"], "raw", "vindr_pcxr-dicom_metadata.csv"),
        "image": join(DIR_DATA_MAP["metadata"], "vindr_pcxr_metadata.csv"),
    },
    # NIH - Chest Xray 18
    "nih_cxr18": {
        "image": join(DIR_DATA_MAP["metadata"], "nih_cxr18_metadata.csv")
    },
    # PadChest
    "padchest": {
        "image": join(DIR_DATA_MAP["metadata"], "padchest_metadata.csv")
    },
    # CheXBERT (originally CheXpert)
    "chexbert": {
        "image": join(DIR_DATA_MAP["metadata"], "chexbert_metadata.csv")
    },

    # Open Medical Imaging Datasets  NOTE: This is not used for the CXR datasets
    "open_data": join(DIR_DATA_MAP["metadata"], "open_data_metadata.xlsx")
}

# Directory containing configurations
DIR_CONFIG = join(DIR_PROJECT, "config")
# Directory containing config specifications
DIR_CONFIG_SPECS = join(DIR_CONFIG, "configspecs")

# Local directory to save data (model weights, embeddings, figures)
DIR_SAVE_DATA = DIR_DATA_MAP["save"]
DIR_TRAIN_RUNS = join(DIR_SAVE_DATA, "train_runs")
DIR_INFERENCE = join(DIR_SAVE_DATA, "inference")
DIR_EMBEDS = join(DIR_SAVE_DATA, "embeddings")
DIR_FIGURES = join(DIR_SAVE_DATA, "figures")
DIR_FINDINGS = join(DIR_SAVE_DATA, "findings")

# Directory containing eda figures
DIR_FIGURES_EDA = join(DIR_FIGURES, "eda")
# Directory containing UMAP figures
DIR_FIGURES_UMAP = join(DIR_FIGURES, "umap")
# Directory containing GradCAM figures
DIR_FIGURES_CAM = join(DIR_FIGURES, "grad_cam")
# Directory containing prediction-related figures
DIR_FIGURES_PRED = join(DIR_FIGURES, "predictions")

# Directory for EDA on Open Medical Data
DIR_FIGURES_EDA_CHALLENGES = join(DIR_FIGURES_EDA, "open_mi", "challenges")
DIR_FIGURES_EDA_BENCHMARKS = join(DIR_FIGURES_EDA, "open_mi", "benchmarks")
DIR_FIGURES_EDA_DATASETS = join(DIR_FIGURES_EDA, "open_mi", "datasets")
DIR_FIGURES_EDA_COLLECTIONS = join(DIR_FIGURES_EDA, "open_mi", "collections")
DIR_FIGURES_EDA_PAPERS = join(DIR_FIGURES_EDA, "open_mi", "papers")


################################################################################
#                                 Data Related                                 #
################################################################################
# Image size
IMG_SIZE = (224, 224)

# Mapping of label column to classes
COL_TO_CLASSES = {
    "Cardiomegaly": ["Negative", "Positive"],
    "Has Finding": ["Negative", "Positive"]
}

# Column containing image ID
ID_COL = "image_id"
