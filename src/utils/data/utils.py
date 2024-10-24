"""
utils.py

Description: Contains helper functions for a variety of data/label preprocessing
             functions.
"""

# Standard libraries
import json
import logging
import os

# Non-standard libraries
import numpy as np
import pandas as pd
import torchvision.transforms.v2 as T
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

# Custom libraries
from config import constants


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Random seed
SEED = 42


################################################################################
#                                Data Splitting                                #
################################################################################
def split_by_ids(patient_ids, train_split=0.8,
                 force_train_ids=None,
                 seed=SEED):
    """
    Splits list of patient IDs into training and val/test set.

    Note
    ----
    Split may not be exactly equal to desired train_split due to imbalance of
    patients.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    train_split : float, optional
        Proportion of total data to leave for training, by default 0.8
    force_train_ids : list, optional
        List of patient IDs to force into the training set
    seed : int, optional
        If provided, sets random seed to value, by default SEED

    Returns
    -------
    tuple of (np.array, np.array)
        Contains (train_indices, val_indices), which are arrays of indices into
        patient_ids to specify which are used for training or validation/test.
    """
    # Get expected # of items in training set
    n = len(patient_ids)
    n_train = int(n * train_split)

    # Add soft lower/upper bounds (5%) to expected number. 
    # NOTE: Assume it won't become negative
    n_train_min = int(n_train - (n * 0.05))
    n_train_max = int(n_train + (n * 0.05))

    # Create mapping of patient ID to number of occurrences
    id_to_len = {}
    for _id in patient_ids:
        if _id not in id_to_len:
            id_to_len[_id] = 0
        id_to_len[_id] += 1

    # Shuffle unique patient IDs
    unique_ids = list(id_to_len.keys())
    shuffled_unique_ids = shuffle(unique_ids, random_state=seed)

    # Randomly choose patients to add to training set until full
    train_ids = set()
    n_train_curr = 0

    # If provided, strictly move patients into training set
    if force_train_ids:
        for forced_id in force_train_ids:
            if forced_id in id_to_len:
                train_ids.add(forced_id)
                n_train_curr += 1
                shuffled_unique_ids.remove(forced_id)

    for _id in shuffled_unique_ids:
        # Add patient if number of training samples doesn't exceed upper bound
        if n_train_curr + id_to_len[_id] <= n_train_max:
            train_ids.add(_id)
            n_train_curr += id_to_len[_id]

        # Stop when there is roughly enough in the training set
        if n_train_curr >= n_train_min:
            break

    # Create indices
    train_idx = []
    val_idx = []
    for idx, _id in enumerate(patient_ids):
        if _id in train_ids:
            train_idx.append(idx)
        else:
            val_idx.append(idx)

    # Convert to arrays
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return train_idx, val_idx


def stratify_split_by_ids(
        image_ids, labels,
        train_split=0.8, force_train_ids=None,
        seed=SEED
    ):
    """
    Splits the given image_ids into train and val sets using
    StratifiedShuffleSplit.

    Note
    ----
    Assumes image_ids are unique (i.e., not duplicate patient IDs)

    Parameters
    ----------
    image_ids : list
        List of image IDs
    labels : list
        List of labels
    train_split : float, optional
        Fraction of samples to use for training, by default 0.8
    force_train_ids : list, optional
        List of image IDs to force into the training set, by default None
    seed : int, optional
        Seed for random shuffling, by default SEED

    Returns
    -------
    train_idx : np.array
        Indices of the training set
    val_idx : np.array
        Indices of the validation set
    """
    # Raise error, if image IDs are not unique
    if len(image_ids) != len(set(image_ids)):
        raise NotImplementedError(
            "Stratified split not implemented for case when image IDs are NOT "
            "unique (e.g., multiple images for 1 patient)!"
        )

    shuffler = StratifiedShuffleSplit(
        n_splits=int(1/(1-train_split)),
        train_size=train_split,
        random_state=seed,
    )

    # Create indices for stratified split
    train_idx, val_idx = next(shuffler.split(image_ids, labels))

    # If provided, strictly move patients into training set
    if force_train_ids:
        force_train_idx = [i for i, _id in enumerate(image_ids) if _id in force_train_ids]
        train_idx = np.concatenate([train_idx, force_train_idx])
        val_idx = np.array([i for i in range(len(image_ids)) if i not in train_idx])

    return train_idx, val_idx


def cross_validation_by_patient(patient_ids, num_folds=5):
    """
    Create train/val indices for Cross-Validation with exclusive patient ids
    betwen training and validation sets.

    Parameters
    ----------
    patient_ids : np.array or array-like
        List of patient IDs (IDs can repeat).
    num_folds : int
        Number of folds for cross-validation

    Returns
    -------
    list of <num_folds> tuples of (np.array, np.array)
        The k-th element is the k-th fold's (list of train_ids, list of val_ids)
    """
    folds = []

    training_folds = []
    remaining_ids = patient_ids

    # Get patient IDs for training set in each folds
    while num_folds > 1:
        proportion = 1 / num_folds
        train_idx, rest_idx = split_by_ids(remaining_ids, proportion)

        training_folds.append(np.unique(remaining_ids[train_idx]))
        remaining_ids = remaining_ids[rest_idx]
        
        num_folds -= 1

    # The last remaining IDs are the patient IDs of the last fold
    training_folds.append(np.unique(remaining_ids))

    # Create folds
    fold_idx = list(range(len(training_folds)))
    for i in fold_idx:
        # Get training set indices
        uniq_train_ids = set(training_folds[i])
        train_idx = np.where([_id in uniq_train_ids for _id in patient_ids])[0]

        # Get validation set indices
        val_indices = fold_idx.copy()
        val_indices.remove(i)
        val_patient_ids = np.concatenate(
            np.array(training_folds, dtype=object)[val_indices])
        uniq_val_ids = set(val_patient_ids)
        val_idx = np.where([_id in uniq_val_ids for _id in patient_ids])[0]

        folds.append((train_idx, val_idx))

    return folds


def assign_split_table(df_metadata,
                       other_split="test",
                       label_col="label",
                       id_col=constants.ID_COL,
                       stratify_split=False,
                       overwrite=False,
                       **split_kwargs):
    """
    Split table into train and test, and add "split" column to specify which
    split.

    Note
    ----
    If image has no label, it's not part of any split.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Each row represents an US image at least a patient ID and label
    other_split : str, optional
        Name of other split. For example, ("val", "test"), by default "test"
    label_col : str, optional
        Name of column in df_metadata that contains labels
    id_col : str, optional
        Name of column in df_metadata that contains patient/image IDs
    stratify_split : bool, optional
        If True, then stratify splits by label
    overwrite : bool, optional
        If val/test splits already exists, then don't overwrite
    **split_kwargs : Any
        Keyword arguments to pass into `split_by_ids` or `stratify_split_by_ids`
        if stratify_split is True.

    Returns
    -------
    pd.DataFrame
        Metadata table
    """
    # Reset index
    df_metadata = df_metadata.reset_index(drop=True)

    # If split already exists, then don't overwrite and return early
    if (not overwrite and "split" in df_metadata.columns.tolist()
            and other_split in df_metadata["split"].unique().tolist()):
        LOGGER.info(f"Split `{other_split}` already exists! Not overwriting...")
        return df_metadata

    # Add column for split
    if "split" in df_metadata.columns:
        LOGGER.warning(f"Overwriting existing data splits for (train/{other_split})!")
    df_metadata["split"] = None

    # Split into tables with labels and without
    df_labeled = df_metadata[~df_metadata[label_col].isna()]
    df_unlabeled = df_metadata[df_metadata[label_col].isna()]

    # Add labels to split keyword arguments, if stratifying
    if stratify_split:
        split_kwargs["labels"] = df_labeled[label_col].tolist()

    # If specified, only modify val/test if there are none that exist
    # Split labeled data into train and test
    patient_ids = df_labeled[id_col].tolist()
    split_func = stratify_split_by_ids if stratify_split else split_by_ids
    train_idx, test_idx = split_func(patient_ids, **split_kwargs)
    df_labeled.loc[train_idx, "split"] = "train"
    df_labeled.loc[test_idx, "split"] = other_split

    # Recombine data
    df_metadata = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    return df_metadata


def assign_unlabeled_split(df_metadata, split="train", label_col="label"):
    """
    Assign unlabeled data to data split (e.g., train)

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table where each row is an ultrasound image
    split : str, optional
        Data split. Can also be None, if not desired to be part of
        train/val/test
    label_col : str, optional
        Name of column in df_metadata that contains labels

    Returns
    -------
    pd.DataFrame
        Metadata table where unlabeled images are now part of specified split
    """
    # Don't perform in-place
    df_metadata = df_metadata.copy()

    # Add column for split, if not exists
    if "split" not in df_metadata.columns:
        df_metadata["split"] = None

    # Split into tables with labels and without
    df_labeled = df_metadata[~df_metadata[label_col].isna()]
    df_unlabeled = df_metadata[df_metadata[label_col].isna()]

    # Move unlabeled data to specified set
    df_unlabeled["split"] = split

    # Recombine data
    df_metadata = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    return df_metadata


def exclude_from_any_split(df_metadata, json_path):
    """
    Get list of filenames to exclude from any split. Unset their assigned split.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table
    json_path : str
        Path to json file containing list of filenames to exclude

    Returns
    -------
    pd.DataFrame
        Metadata table with rows whose data splits were could be unassigned
    """
    # Raise error, if file doesn't exist
    if not os.path.exists(json_path):
        raise RuntimeError("Exclude filename path doesn't exist!\n\t"
                            f"{json_path}")

    # Create copy to avoid in-place operations
    df_metadata = df_metadata.copy()

    # Load JSON file
    with open(json_path, "r") as handler:
        exclude_fnames = set(json.load(handler))

    # For each of train/val/test, ensure images are all filtered
    splits = [split for split in df_metadata["split"].unique().tolist() if split]
    for split in splits:
        # Check if each file should be included/excluded
        excluded_mask = df_metadata["filename"].map(
            lambda x: x in exclude_fnames or os.path.basename(x) in exclude_fnames
        )
        # Filter for split-specific data
        excluded_mask = excluded_mask & (df_metadata["split"] == split)
        # Unassign split
        df_metadata.loc[excluded_mask, "split"] = None
        LOGGER.info(f"Explicitly excluding {(excluded_mask).sum()} images from `{split}`!")

    return df_metadata


################################################################################
#                             Image Preprocessing                              #
###############################################################################
def prep_strong_augmentations(img_size=(256, 256), crop_scale=0.5):
    """
    Prepare strong training augmentations.

    Parameters
    ----------
    img_size : tuple, optional
        Expected image size post-augmentations
    crop_scale : float
        Minimum proportion/size of area to crop

    Returns
    -------
    dict
        Maps from texture/geometric to training transforms
    """
    transforms = {}
    transforms["texture"] = T.Compose([
        T.RandomEqualize(p=0.5),
        T.RandomAutocontrast(p=0.5),
        T.RandomAdjustSharpness(1.25, p=0.25),
        T.RandomApply([T.GaussianBlur(1, 0.1)], p=0.5),
    ])
    transforms["geometric"] = T.Compose([
        T.RandomRotation(15),
        T.RandomResizedCrop(img_size, scale=(crop_scale, 1)),
        T.RandomZoomOut(fill=0, side_range=(1.0,  2.0), p=0.25),
        T.Resize(img_size),
    ])
    return transforms


def prep_weak_augmentations(img_size=(256, 256)):
    """
    Prepare weak training augmentations.

    Parameters
    ----------
    img_size : tuple, optional
        Expected image size post-augmentations

    Returns
    -------
    dict
        Maps from texture/geometric to training transforms
    """
    transforms = {}
    transforms["texture"] = T.Compose([
        T.RandomEqualize(p=0.5),
        T.RandomAutocontrast(p=0.5),
        T.RandomAdjustSharpness(1.25, p=0.25),
        T.RandomApply([T.GaussianBlur(1, 0.1)], p=0.5),
    ])
    transforms["geometric"] = T.Compose([
        T.RandomRotation(7),
        T.RandomResizedCrop(img_size, scale=(0.8, 1)),
        T.RandomZoomOut(fill=0, side_range=(1.0,  1.1), p=0.5),
        T.Resize(img_size),
    ])
    return transforms


################################################################################
#                        Miscellaneous Helper Functions                        #
################################################################################
def argsort_unsort(arr):
    """
    Given an array to sort, return indices to sort and unsort the array.

    Note
    ----
    Given arr = [C, A, B, D] and its index array be [0, 1, 2, 3].
        Sort indices: [2, 0, 1, 3] result in [A, B, C, D] & [1, 2, 0, 3],
        respectively.

    To unsort, sort index array initially sorted by `arr`
        Initial index: [1, 2, 0, 3]
        Sorted indices: [2, 0, 1, 3] result in arr = [C, A, B, D]

    Parameters
    ----------
    arr : pd.Series or np.array
        Array of items to sort

    Returns
    -------
    tuple of (np.array, np.array)
        First array contains indices to sort the array.
        Second array contains indices to unsort the sorted array.
    """
    sort_idx = np.argsort(arr)
    unsort_idx = np.argsort(np.arange(len(arr))[sort_idx])

    return sort_idx, unsort_idx


def is_null(x):
    """
    Returns True, if x is null

    Parameters
    ----------
    x : Any
        Any object

    Returns
    -------
    bool
        True if x is null and False otherwise
    """
    if pd.isnull(x) or x is None:
        return True
    if x == "nan" or x == "None" or x == "N/A":
        return True
    return False
