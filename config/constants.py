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
DIR_CXR_DATA = os.environ.get("DIR_CXR") or join(DIR_DATA, "cxr_datasets")
DIR_DATA_MAP = {
    "save": join(DIR_DATA, "save_data"),
    "metadata": join(DIR_DATA, "metadata"),
    "vindr_cxr": join(DIR_CXR_DATA, "vindr-cxr"),
    "vindr_pcxr": join(DIR_CXR_DATA, "vindr-pcxr"),
    "nih_cxr18": join(DIR_CXR_DATA, "NIH"),
    "padchest": join(DIR_CXR_DATA, "PC"),
    "chexbert": join(DIR_CXR_DATA, "chexbert"),
    # Dummy dataset
    "dummy": join(DIR_DATA, "dummy_data"),
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

    # Dummy dataset
    "dummy": {
        "image": join(DIR_DATA_MAP["metadata"], "dummy_metadata.csv")
    },

    # Open Medical Imaging Datasets  NOTE: This is not used for the CXR datasets
    "open_data": join(DIR_DATA_MAP["metadata"], "public_datasets_metadata.xlsx"),
    "openneuro": join(DIR_DATA_MAP["metadata"], "openneuro_metadata.csv"),
    "openneuro_parsed": join(DIR_DATA_MAP["metadata"], "openneuro_metadata_parsed.csv"),
    "midrc": join(DIR_DATA_MAP["metadata"], "midrc_metadata.csv"),
    "midrc_parsed": join(DIR_DATA_MAP["metadata"], "midrc_metadata_parsed.csv"),
}

# Metadata directory for individual OpenNeuro datasets
DIR_OPENNEURO_METADATA = join(DIR_DATA_MAP["metadata"], "openneuro")
# String formatter to resolve dataset metadata path
OPENNEURO_METADATA_FMT = "https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/refs/heads/{branch}/participants.tsv"

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
DIR_FIGURES_MI = join(DIR_FIGURES_EDA, "open_mi")
DIR_FIGURES_EDA_CHALLENGES = join(DIR_FIGURES_MI, "challenges")
DIR_FIGURES_EDA_BENCHMARKS = join(DIR_FIGURES_MI, "benchmarks")
DIR_FIGURES_EDA_DATASETS = join(DIR_FIGURES_MI, "datasets")
DIR_FIGURES_EDA_COLLECTIONS = join(DIR_FIGURES_MI, "collections")
DIR_FIGURES_EDA_PAPERS = join(DIR_FIGURES_MI, "papers")


################################################################################
#                                 Data Related                                 #
################################################################################
# Image size
IMG_SIZE = (224, 224)

# List of datasets
ALL_DATASETS = ["vindr_pcxr", "vindr_cxr", "nih_cxr18", "padchest", "chexbert"]

# Mapping of dataset to normalization constants
DSET_TO_NORM = {
    "vindr_cxr": {"mean": [0.5512], "std": [0.2411]
    },
    "vindr_pcxr": {"mean": [0.4676], "std": [0.1587]
    },
    "nih_cxr18": {"mean": [0.5124], "std": [0.2307]
    },
    "padchest": {"mean": [0.5053], "std": [0.2481]
    },
    "chexbert": {"mean": [0.5072], "std": [0.2889]
    },
}

# Mapping of label column to classes
COL_TO_CLASSES = {
    "Cardiomegaly": ["Negative", "Positive"],
    "Has Finding": ["Negative", "Positive"]
}

# Column containing image ID
ID_COL = "image_id"
