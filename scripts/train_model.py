"""
train_model.py

Description: Used to train PyTorch models.
"""

# Standard libraries
import argparse
import logging
import os
import random
import shutil
import sys

# Non-standard libraries
import comet_ml             # NOTE: Recommended by Comet ML
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
)
from lightning.pytorch.loggers import CometLogger

# Custom libraries
from config import constants
from src.utils.model import load_model
from src.utils.misc import config as config_utils
from src.utils.data import load_data


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Default random seed
SEED = None

# Comet-ML project name
COMET_PROJECT = "ped_vs_adult-cxr"


################################################################################
#                                Initialization                                #
################################################################################
def set_seed(seed=SEED, include_algos=False):
    """
    Set random seed for all models.

    Parameters
    ----------
    seed : int, optional
        Random seed. If None, don't set seed, by default SEED
    include_algos : bool, optional
        If True, forces usage of deterministic algorithms at the cost of
        performance, by default False.
    """
    # If seed is None, don't set seed
    if seed is None or seed < 0:
        LOGGER.warning(f"Random seed is not set!")
        return

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Force deterministic algorithms
    if include_algos:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    LOGGER.info(f"Success! Set random seed: {seed}")


################################################################################
#                           Training/Inference Flow                            #
################################################################################
def run(hparams, dm, results_dir=constants.DIR_TRAIN_RUNS, fold=0):
    """
    Perform (1) model training, and/or (2) load model and perform testing.

    Parameters
    ----------
    hparams : dict
        Contains (data-related, model-related) setup parameters for training and
        testing
    dm : L.LightningDataModule
        Data module, which already called .setup()
    results_dir : str
        Path to directory containing trained model and/or test results
    fold : int, optional
        If performing cross-validation, supplies fold index. Index ranges
        between 0 to (num_folds - 1). If train-val-test split, remains 0, by
        default 0.
    """
    exp_name = hparams["exp_name"]

    # Create parent directory if not exists
    if not os.path.exists(f"{results_dir}/{exp_name}"):
        os.makedirs(f"{results_dir}/{exp_name}")

    # Directory for current experiment
    experiment_dir = f"{results_dir}/{exp_name}/{fold}"

    # Check if experiment run exists (i.e., resuming training / evaluation)
    run_exists = os.path.exists(experiment_dir)
    if run_exists and os.listdir(experiment_dir):
        LOGGER.info("Found pre-existing experiment directory! Resuming training/evaluation...")

    # Loggers
    loggers = []
    # If specified, use Comet ML for logging
    if hparams.get("use_comet_logger"):
        if hparams.get("debug"):
            LOGGER.info("Comet ML Logger disabled during debugging...")
            hparams["use_comet_logger"] = False
        elif not os.environ.get("COMET_API_KEY"):
            LOGGER.error(
                "Please set `COMET_API_KEY` environment variable before running! "
                "Or set `use_comet_logger` to false in config file..."
            )
        else:
            exp_key = None
            # If run exists, get stored experiment key to resume logging
            if run_exists:
                old_hparams = load_model.get_hyperparameters(
                    exp_name=hparams["exp_name"])
                exp_key = old_hparams.get("comet_exp_key")

            # Set up LOGGER
            comet_logger = CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                project_name=COMET_PROJECT,
                experiment_name=hparams["exp_name"],
                experiment_key=exp_key
            )

            # Store experiment key
            hparams["comet_exp_key"] = comet_logger.experiment.get_key()

            # Add tags
            tags = hparams.get("tags")
            if tags:
                comet_logger.experiment.add_tags(tags)
            loggers.append(comet_logger)

    # Flag for presence of validation set
    includes_val = dm.size("val") > 0

    # Callbacks
    callbacks = []
    # 1. Model checkpointing
    if hparams.get("checkpoint"):
        LOGGER.info("Storing model checkpoints...")
        callbacks.append(
            ModelCheckpoint(dirpath=experiment_dir, save_last=True,
                            monitor="val_loss" if includes_val else None))
    # 2. Stochastic Weight Averaging
    if hparams.get("swa"):
        LOGGER.info("Performing stochastic weight averaging (SWA)...")
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))
    # 3. Early stopping
    if hparams.get("early_stopping"):
        LOGGER.info("Performing early stopping on validation loss...")
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    # Initialize Trainer
    trainer = Trainer(default_root_dir=experiment_dir,
                      devices="auto", accelerator="auto",
                      num_sanity_val_steps=0,
                      log_every_n_steps=20,
                      accumulate_grad_batches=hparams.get("accum_batches", 1),
                      precision=hparams["precision"],
                      gradient_clip_val=hparams["grad_clip_norm"],
                      max_epochs=hparams["stop_epoch"],
                      enable_checkpointing=hparams["checkpoint"],
                      callbacks=callbacks,
                      logger=loggers,
                      fast_dev_run=hparams["debug"],
                      )

    # Show data stats
    LOGGER.info(f"[Training] Num Images: {dm.size('train')}")
    if includes_val:
        LOGGER.info(f"[Validation] Num Images: {dm.size('val')}")

    # Create model (from scratch) or load pretrained
    model = load_model.ModelWrapper(hparams)

    # 1. Perform training
    if hparams["train"]:
        # If resuming training
        ckpt_path = None
        if run_exists:
            ckpt_path = "last"

        # Create dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader() if includes_val else None

        # Perform training
        try:
            trainer.fit(model, train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                        ckpt_path=ckpt_path)
        except KeyboardInterrupt:
            # Delete experiment directory
            if os.path.exists(experiment_dir):
                LOGGER.error("Caught keyboard interrupt! Deleting experiment directory")
                shutil.rmtree(experiment_dir)
            exit(1)

    # 2. Perform testing
    if hparams["test"]:
        trainer.test(model=model, dataloaders=dm.test_dataloader())


def main(conf):
    """
    Main method to run experiments

    Parameters
    ----------
    conf : configobj.ConfigObj
        Contains configurations needed to run experiments
    """
    # Process configuration parameters
    # Flatten nesting in configuration file
    hparams = config_utils.flatten_nested_dict(conf)

    # Overwrite number of classes
    if "num_classes" not in hparams:
        LOGGER.info("`num_classes` not provided! Assuming it's a binary classification problem...")
        hparams["num_classes"] = 2

    # Add default image size, if not specified
    if "img_size" not in hparams:
        LOGGER.info(f"`img_size` not provided! Using default image size ({constants.IMG_SIZE})")
        hparams["img_size"] = constants.IMG_SIZE

    # Add default dataset-specific normalizing constants, if not specified
    if not hparams.get("norm_mean") and hparams["dset"] in constants.DSET_TO_NORM:
        norm_constants = constants.DSET_TO_NORM[hparams["dset"]]
        LOGGER.info(f"Normalization constants (mean/std) not provided! Using default image size ({norm_constants})")
        hparams["norm_mean"] = norm_constants["mean"]
        hparams["norm_std"] = norm_constants["std"]

    # 0. Set random seed
    set_seed(hparams.get("seed"))

    # 1. Set up data module
    dm = load_data.setup_data_module(hparams)

    # 2.1 Run experiment
    if hparams["cross_val_folds"] == 1:
        run(hparams, dm, constants.DIR_TRAIN_RUNS)
    # 2.2 Run experiment w/ kfold cross-validation)
    else:
        for fold_idx in range(hparams["cross_val_folds"]):
            dm.set_kfold_index(fold_idx)
            run(hparams, dm, constants.DIR_TRAIN_RUNS, fold=fold_idx)


if __name__ == "__main__":
    # 0. Initialize ArgumentParser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config",
        type=str,
        help=f"Name of configuration file under `{constants.DIR_CONFIG}`"
    )

    # 1. Get arguments
    ARGS = PARSER.parse_args()

    # 2. Load configurations
    CONF = config_utils.load_config(__file__, ARGS.config)
    LOGGER.debug("""
################################################################################
#                       Starting `model_training` Script                       #
################################################################################""")

    # 3. Run main
    main(CONF)
