"""
load_data.py

Description: Contains utility functions for instantiating DataModule objects.

Note: `hparams` is a direct dependence on arguments in `model_training.py`.
"""

# Standard libraries
import logging

# Custom libraries
from config import constants
from src.utils.data.dataset import CXRDataModule


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_HPARAMS = {
    "dsets": "vindr_cxr",
    "train_val_split": 1,
    "train_test_split": 1,
    "train": True,
    "test": True,

    "img_size": constants.IMG_SIZE,
    "label_col": "Cardiomegaly",

    "batch_size": 16,
    "full_seq": False,
    "shuffle": False,
}


################################################################################
#                                  Functions                                   #
################################################################################
def setup_data_module(hparams=None, use_defaults=False,
                      **overwrite_hparams):
    """
    Set up data module.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters
    use_defaults : bool, optional
        If True, start from default hyperparameters. Defaults to False.
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite `hparams`

    Returns
    -------
    lightning.pytorch.LightningDataModule
    """
    all_hparams = {}
    # 0. If specified, start from default hyperparameters
    if use_defaults:
        all_hparams.update(DEFAULT_HPARAMS)

    # INPUT: Ensure `hparams` is a dict
    hparams = hparams or {}

    # 0. Overwrite default hyperparameters
    all_hparams.update(hparams)
    all_hparams.update(overwrite_hparams)

    # 1. Instantiate data module
    dm = CXRDataModule(all_hparams)
    dm.setup()

    # Modify hyperparameters in-place to store training/val/test set sizes
    for split in ("train", "val", "test"):
        hparams[f"{split}_size"] = dm.size(split)

    return dm


def setup_default_data_module_for_dset(dset=None, **kwargs):
    """
    Get image dataloader for dataset split/name specified.

    Parameters
    ----------
    dset : str
        Name of dataset
    **kwargs : dict, optional
        Keyword arguments for `setup_data_module`
        
    Returns
    -------
    lightning.pytorch.DataModule
        Each batch returns images and a dict containing metadata
    """
    # Prepare arguments for data module
    dm_kwargs = create_eval_hparams(dset)
    # Update with kwargs
    dm_kwargs.update(kwargs)
    # Remove `use_defaults` kwargs
    dm_kwargs.pop("use_defaults", None)

    # Set up data module
    dm = setup_data_module(use_defaults=True, **dm_kwargs)

    return dm


def setup_default_dataloader_for_dset(dset, split=None, filters=None, **overwrite_hparams):
    """
    Create DataLoader for specific dataset and train/val/test split.

    Parameters
    ----------
    dset : str
        Name of dataset
    filters : dict, optional
        Mapping of column name to allowed value/s
    **overwrite_hparams : dict, optional
        Keyword arguments to overwrite hyperparameters
    """
    # Ensure filters is a dict
    filters = filters or {}

    # Create DataModule
    dm = setup_default_data_module_for_dset(
        dset=dset,
        **overwrite_hparams
    )

    # Get filtered dataloader
    dataloader = dm.get_filtered_dataloader(split=split, **filters)

    return dataloader


def create_eval_hparams(dset=None):
    """
    Create hyperparameters to evaluate on a data split (typically test)

    Parameters
    ----------
    dset : str
        If provided, filter by dataset name

    Returns
    -------
    dict
        Contains hyperparameters to overwrite, if necessary
    """
    # Accumulate hyperparameters to overwrite
    overwrite_hparams = {
        "shuffle": False,
        "augment_training": False,
        "imbalanced_sampler": False,
    }

    # Check that provided dataset or split is valid
    if dset:
        assert dset in constants.DIR_METADATA_MAP, (
            f"`{dset}` is not a valid dataset! Must be one of "
            f"{list(constants.DIR_METADATA_MAP.keys())}"
        )
        # Set dataset
        overwrite_hparams["dset"] = dset
        overwrite_hparams["dsets"] = [dset]

    return overwrite_hparams
