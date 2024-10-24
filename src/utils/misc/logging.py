"""
logging.py

Description: Wrapper over PyTorch Lightning's CSVLogger to output a simpler CSV
             file (history.csv) after training.
"""

# Standard libraries
import os

# Non-standard libraries
import pandas as pd
from comet_ml import ExistingExperiment


################################################################################
#                                  Constants                                   #
################################################################################
# Comet ML Experiment Cache
COMET_EXP_CACHE = {}


################################################################################
#                               Helper Functions                               #
################################################################################
def load_comet_logger(exp_key):
    """
    Load Comet ML logger for existing experiment.

    Parameters
    ----------
    exp_key : str
        Experiment key for existing experiment

    Returns
    -------
    comet_ml.ExistingExperiment
        Can be used for logging
    """
    # Check if in cache
    if exp_key in COMET_EXP_CACHE:
        return COMET_EXP_CACHE[exp_key]

    # Otherwise, load for the first time
    assert "COMET_API_KEY" in os.environ, "Please set `COMET_API_KEY` before running this script!"
    logger = ExistingExperiment(
        previous_experiment=exp_key,
    )
    # Store in cache
    COMET_EXP_CACHE[exp_key] = logger
    return logger
