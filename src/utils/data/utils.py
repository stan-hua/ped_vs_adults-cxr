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
    df_labeled = df_metadata[~df_metadata[label_col].isna()].reset_index(drop=True)
    df_unlabeled = df_metadata[df_metadata[label_col].isna()].reset_index(drop=True)

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


def sample_by_age_bins(df_metadata, age_bins, size=0, prop=0, age_col="age", seed=SEED):
    """
    Sample a number / proportion of patients from each age bin.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table with age annotations
    age_bins : list
        List of ages to create age bins for patients (e.g., [5, 10, 15] results
        in age bins of [5, 10) and [10, 15))
    size : int, optional
        Number of patients to sample from each age bin, by default 0
    prop : float, optional
        Proportion of patients from each age bin to sample, by default 0.2
    age_col : str, optional
        Name of age column in df_metadata, by default "age"
    seed : int, optional
        Random seed for reproducibility, by default SEED

    Returns
    -------
    tuple
        Two DataFrames: sampled and not sampled patients
    """
    assert size or prop, "Either 'size' or 'prop' must be greater than 0!"

    # Create a new column 'age_bin' to categorize ages into bins
    df_metadata["age_bin"] = pd.cut(df_metadata[age_col], bins=age_bins, right=False)

    # Initialize DataFrames to store the sampled and not sampled rows
    accum_sampled = []
    accum_not_sampled = []

    # Loop through each age bin and sample the specified proportion
    for age_bin in df_metadata["age_bin"].unique():
        # Skip null age bin
        if not age_bin or pd.isnull(age_bin):
            continue
        bin_df = df_metadata[df_metadata["age_bin"] == age_bin]
        # Skip empty bins
        if bin_df.empty:
            print(f"Empty age bin: {age_bin}")
            continue

        # Set current sample size
        curr_size = size if size else int(len(bin_df) * prop)

        # Sample
        sampled_bin_df = bin_df.sample(n=curr_size, random_state=seed)
        not_sampled_bin_df = bin_df.drop(sampled_bin_df.index)

        # Append the sampled and not sampled rows
        accum_sampled.append(sampled_bin_df)
        accum_not_sampled.append(not_sampled_bin_df)

    # Concatenate accumulated tables
    sampled_df = pd.concat(accum_sampled)
    not_sampled_df = pd.concat(accum_not_sampled)

    # Drop the "age_bin" column before returning the result
    sampled_df = sampled_df.drop(columns=["age_bin"])
    not_sampled_df = not_sampled_df.drop(columns=["age_bin"])

    return sampled_df, not_sampled_df


def get_age_bins(df_metadata, dset, split=None, age_col="age_years",
                 include_peds=False):
    """
    Determine age bins based on the dataset and split.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        Metadata table containing age column
    dset : str
        Name of the dataset.
    split : str, optional
        Name of the data split.
    age_col : str, optional
        Name of age column, by default "age_years"
    include_peds : bool, optional
        For adult datasets, include peds bin, if True

    Returns
    -------
    pd.Series
        List of string of age bins ages
    """
    split = split or ""

    # Determine age bin boundaries
    # CASE 1: Adult dataset
    age_splits = [18, 25, 40, 60, 80, 100]
    if include_peds:
        age_splits.insert(0, 0)
    # CASE 2: Pediatric dataset
    if dset in ["vindr_pcxr"] or "peds" in split:
        age_splits = list(range(19))

    # Assign each row to an age bin
    age_bins = pd.cut(df_metadata[age_col], bins=age_splits, right=False)

    # Map them back to integers, if it's only a single age per bin
    age_bins = age_bins.map(
        lambda x: str(x.left) if isinstance(x, pd.Interval) and (x.right - x.left == 1)
                  else str(x))
    age_bins = age_bins.astype(str)

    return age_bins


def stringify_dataset_split(dset, split=None):
    """
    Return a human-readable string for a given dataset-split combination.

    Parameters
    ----------
    dset : str
        Name of dataset
    split : str
        Name of data split

    Returns
    -------
    str
        Human-readable string
    """
    if split == "test":
        assert dset == "vindr_pcxr", "Only VinDr-PCXR has a valid used test split!"

    # Create mappings
    map_dset = {
        "vindr_pcxr": "VinDr-PCXR",
        "vindr_cxr": "VinDr-CXR",
        "padchest": "PadChest",
        "nih_cxr": "NIH",
        "nih_cxr18": "NIH",
        "chexbert": "CheXBERT",
    }
    map_split = {
        "test": "Healthy Children",
        "test_healthy_adult": "Healthy Adults",
        "test_peds": "Healthy Children",
        "test_adult_calib": "Healthy/Unhealthy Adults"
    }

    # CASE 1: Split provided
    if split is not None:
        return f"{map_split[split]} in {map_dset[dset]}"
    # CASE 2: Split not provided
    return map_dset[dset]


################################################################################
#                             Image Preprocessing                              #
###############################################################################
def prep_strong_augmentations(hparams):
    """
    Prepare strong training augmentations.

    Parameters
    ----------
    hparams : dict
        Contains hyperparameters for augmentations, such as:
        img_size : tuple, optional
            Expected image size post-augmentations
        crop_scale : float
            Minimum proportion/size of area to crop

    Returns
    -------
    dict
        Maps from texture/geometric to training transforms
    """
    transforms = {
        "texture": [],
        "geometric": [],
    }

    # 1. Texture Augmentations
    if hparams.get("aug_equalize"):
        transforms["texture"].append(T.RandomEqualize(p=0.5))
    if hparams.get("aug_sharpen"):
        transforms["aug_texture"].append(T.RandomAdjustSharpness(1.25, p=0.25))
    if hparams.get("aug_blur"):
        transforms["texture"].append(T.RandomApply([T.GaussianBlur(1, 0.1)], p=0.5))

    # 2. Geometric Augmentations
    if hparams.get("aug_rotate"):
        transforms["geometric"].append(T.RandomRotation(15))
    if hparams.get("aug_crop"):
        transforms["geometric"].append(T.RandomResizedCrop(hparams.get("img_size"), scale=(hparams.get("crop_scale"), 1)))
    if hparams.get("aug_zoomout"):
        transforms["geometric"].append(T.RandomZoomOut(fill=0, side_range=(1.0,  2.0), p=0.25))
    # Reshape back to image size
    if hparams.get("img_size"):
        transforms["geometric"].append(T.Resize(hparams.get("img_size")))

    # 3. Assemble transforms
    for transform_type in transforms.keys():
        if transforms[transform_type]:
            transforms[transform_type] = T.Compose(transforms[transform_type])
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
#                             Metadata Processing                              #
################################################################################
def extract_age(dset, df_metadata, age_col="age"):
    """
    Extracts age from a DataFrame of metadata, given the dataset name.
    
    Parameters
    ----------
    dset : str
        Name of dataset
    df_metadata : pd.DataFrame
        DataFrame of metadata
    age_col : str, optional
        Name of column containing age information
    
    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional column "age_years" containing the age in years.
    """
    # Temporarily fill nulls with empty string
    df_metadata[age_col] = df_metadata[age_col].fillna("")

    # CASE 1: Adult dataset (>18 years old)
    if dset == "vindr_cxr":
        df_metadata["age_years"] = df_metadata[age_col].map(lambda x: extract_age_from_str(x, "years"))
        df_metadata["age_years"] = df_metadata["age_years"].map(
            lambda x: x if isinstance(x, (int, float)) and not pd.isnull(x) and int(x) >= 18 else None)
        df_metadata["age_months"] = df_metadata["age_years"] * 12
    # CASE 2: If Pediatric dataset, parse age which is in months / years
    # NOTE: VinDr-PCXR should only have patients <= 10 years old
    elif dset == "vindr_pcxr":
        df_metadata["age_years"] = df_metadata[age_col].map(lambda x: extract_age_from_str(x, "years"))
        df_metadata["age_years"] = df_metadata["age_years"].map(
            lambda x: x if isinstance(x, (int, float)) and not pd.isnull(x) and int(x) <= 10 else None)
        df_metadata["age_months"] = df_metadata[age_col].map(lambda x: extract_age_from_str(x, "months"))
    # CASE 3: NIH Chest Xray 18 and PadChest
    elif dset in ["nih_cxr18", "padchest"]:
        df_metadata["age_years"] = df_metadata[age_col].map(lambda x: extract_age_from_str(x, "years"))
        df_metadata["age_months"] = df_metadata[age_col].map(lambda x: extract_age_from_str(x, "months"))
    else:
        raise ValueError(f"Invalid dataset: `{dset}`")
    
    return df_metadata


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


def extract_age_from_str(text, time="years"):
    """
    Extract age in months/years from string

    Parameters
    ----------
    text : str
        String to extract age from
    time : str, optional
        Time frame to extract. One of ("years", "months", "days")

    Returns
    -------
    int or None
        Approximate age in days/months/years
    """
    # Early return
    if not text:
        return None

    try:
        # CASE 1: If years provided
        if "Y" in text:
            years = int(text.replace("Y", ""))

            # If years is greater than 125, it is probably wrong
            if years > 125:
                LOGGER.warning(f"Found age ({years}) > 125 years! Assuming incorrect and returning null")
                return None

            # Perform date/time conversions
            if time == "years":
                return years
            if time == "months":
                return years * 12
            if time == "days":
                return years * 365
        # CASE 2: If months provided
        if "M" in text:
            months = int(text.replace("M", ""))
            # If years is greater than 125, it is probably wrong
            if months // 12 > 125:
                LOGGER.warning(f"Found age ({years}) > 125 years! Assuming incorrect and returning null")
                return None

            # Perform date/time conversions
            if time == "years":
                return months // 12
            if time == "months":
                return months
            # NOTE: This is an approximation
            if time == "days":
                LOGGER.warning("Approximating months to days using 30 days per month")
                return months * 30
        # CASE 3: If days provided
        if "D" in text:
            days = int(text.replace("D", ""))
            # If years is greater than 125, it is probably wrong
            if days // 365 > 125:
                LOGGER.warning(f"Found age ({years}) > 125 years! Assuming incorrect and returning null")
                return None

            # Perform date/time conversions
            if time == "years":
                return days / 365
            if time == "months":
                LOGGER.warning("Approximating days to months using 30 days per month")
                return days // 30
            if time == "days":
                return days
    except ValueError as error_msg:
        LOGGER.warning(error_msg)
        return None
