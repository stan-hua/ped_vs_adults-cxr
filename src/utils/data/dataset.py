"""
dataset.py

Description: Contains functions/classes to load dataset in PyTorch.

Datasets:

-----------------
VinDr-CXR Dataset
-----------------
    PhysioNet: https://physionet.org/content/vindr-cxr/1.0.0/
    Kaggle: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

    Note
    ----
    Data should be organized as in the PhysioNet dataset
    vindr-cxr / 1.0.0 /
        annotations /
            annotations_(train/test).csv
            image_labels_(train/test).csv
        train / *.dicom
        test  / *.dicom

------------------
VinDr-PCXR Dataset
------------------
    PhysioNet: https://physionet.org/content/vindr-pcxr/1.0.0/

    Note
    ----
    Data should be organized as in the PhysioNet dataset
    vindr-pcxr / 1.0.0 /
        annotations_(train/test).csv
        image_labels_(train/test).csv
        train / *.dicom
        test  / *.dicom
"""
# Standard libraries
import logging
import os

# Non-standard libraries
import pandas as pd
import lightning as L
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from functools import partial
from torch.utils.data import DataLoader

# Custom libraries
from config import constants
from . import utils
from .sampler import ImbalancedDatasetSampler


################################################################################
#                                  Constants                                   #
################################################################################
# Configure logging
LOGGER = logging.getLogger(__name__)

# Default parameters for data loader
DEFAULT_DATALOADER_PARAMS = {
    "batch_size": 16,
    "shuffle": False,
    "num_workers": os.cpu_count() - 1,
}

# Mapping of image mode
MAP_IMG_MODE = {
    1: "GRAY",
    3: "RGB",
}


################################################################################
#                             Data Module Classes                              #
################################################################################
class CXRDataModule(L.LightningDataModule):
    """
    CXRDataModule class.

    Note
    ----
    Used to load CXR data for train/val/test
    """

    def __init__(self, hparams,
                 default_dl_params=DEFAULT_DATALOADER_PARAMS):
        """
        Initialize CXRDataModule object.


        Parameters
        ----------
        default_dl_params : dict, optional
            Default dataloader parameters
        hparams : dict
            Can contain the following keyword arguments:
                mode : int, optional
                    Number of channels (mode) to read images into (1=grayscale, 3=RGB),
                    by default 3.
                augment_training : bool, optional
                    If True, add random augmentations during training, by default False.
                imbalanced_sampler : bool, optional
                    If True, perform imbalanced sampling during training to
                    account for label imbalances, by default False
                img_size : int or tuple of ints, optional
                    If int provided, resizes found images to
                    (img_size x img_size), by default None.
                train_test_split : float
                    Percentage of data to leave for training. The rest will be
                    used for testing
                train_val_split : float
                    Percentage of training set (test set removed) to leave for
                    validation
                cross_val_folds : int
                    Number of folds to use for cross-validation
                crop_scale : float
                    If augmenting training samples, lower bound on proportion of
                    area cropped relative to the full image.
                batch_size : int
                    Batch size
                shuffle : bool
                    If True, shuffle data during training
                num_workers : int
                    Number of CPU data gathering workers
        """
        super().__init__()

        # Store arguments
        # NOTE: Can't overwrite self.hparams in L.LightningDataModule
        self.my_hparams = hparams
        self.default_dl_params = default_dl_params

        # Load metadata for dataset
        self.df = load_metadata(
            dset=hparams["dset"],
            label_col=hparams.get("label_col", "label"),
            filter_negative=hparams.get("filter_negative", False),
        )
        # NOTE: Store static (original) version of dataframe
        self.df_orig = self.df.copy()

        ########################################################################
        #                        DataLoader Parameters                         #
        ########################################################################
        # Extract parameters for training/validation DataLoaders
        self.train_dataloader_params = {}
        for key, default_val in default_dl_params.items():
            self.train_dataloader_params[key] = hparams.get(key, default_val)

        # NOTE: Shuffle is turned off for validation/test set
        self.val_dataloader_params = self.train_dataloader_params.copy()
        self.val_dataloader_params["shuffle"] = False

        ########################################################################
        #                          Dataset Splitting                           #
        ########################################################################
        # (1) To split dataset into training and test sets
        if "train_test_split" in hparams:
            self.train_test_split = hparams.get("train_test_split")

        # (2) To further split training set into train-val set
        if "train_val_split" in hparams and hparams["train_val_split"] != 1.0:
            self.train_val_split = hparams.get("train_val_split")

        # (3) To further split training set into cross-validation sets
        if "cross_val_folds" in hparams and hparams["cross_val_folds"] > 1:
            self.fold = 0
            self.cross_val_folds = hparams["cross_val_folds"]
            self.cross_fold_indices = None

        ########################################################################
        #                            Augmentations                             #
        ########################################################################
        # Standard augmentations used for all training
        self.augmentations = utils.prep_strong_augmentations(hparams)

        # Assign augmentations
        self.transforms = None
        if self.my_hparams.get("augment_training"):
            self.transforms = self.augmentations


    def setup(self, stage="fit"):
        """
        Prepares data for model training/validation/testing
        """
        # (1) Split into training and test sets
        if hasattr(self, "train_test_split") and self.train_test_split < 1:
            # Split data into training/test split and rest
            # NOTE: This is to avoid overwriting other splits
            train_test_mask = self.df["split"].isin(["train", "test"])
            df_train_test = self.df[train_test_mask]
            df_rest = self.df[~train_test_mask]

            # Split into train/test by each dataset
            # NOTE: Do not overwrite train/test split if they already exist
            df_train_test = utils.assign_split_table(
                df_train_test, other_split="test",
                label_col=self.my_hparams.get("label_col", "label"),
                train_split=self.train_test_split,
                force_train_ids=self.my_hparams.get("force_train_ids"),
                overwrite=False,
            )

            # Recombine
            self.df = pd.concat([df_train_test, df_rest], ignore_index=True)

        # (2) Further split training set into train-val or cross-val sets
        # (2.1) Train-Val Split
        if hasattr(self, "train_val_split") and self.train_val_split < 1:
            # Split data into training split and rest
            # NOTE: This is to avoid overwriting other splits
            train_val_mask = self.df["split"].isin(["train", "val"])
            df_train_val = self.df[train_val_mask]
            df_rest = self.df[~train_val_mask]

            # Split training set into train/val by each dataset
            # NOTE: Do not overwrite train/val split if they already exist
            df_train_val = utils.assign_split_table(
                df_train_val, other_split="val",
                train_split=self.train_val_split,
                label_col=self.my_hparams.get("label_col", "label"),
                stratify_split=self.my_hparams.get("stratify_train_val_split", False),
                force_train_ids=self.my_hparams.get("force_train_ids"),
                overwrite=False,
            )

            # Recombine
            self.df = pd.concat([df_train_val, df_rest], ignore_index=True)

        # (2.2) K-Fold Cross Validation
        elif hasattr(self, "cross_val_folds"):
            train_ids = self.df[self.df["split"] == "train"]["image_id"].tolist()
            self.cross_fold_indices = utils.cross_validation_by_patient(
                train_ids, self.cross_val_folds)
            # By default, set to first kfold
            self.set_kfold_index(0)


    def train_dataloader(self):
        """
        Returns DataLoader for training set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for training data
        """
        # Instantiate CXRDatasetDataFrame
        train_dataset = CXRDatasetDataFrame(
            self.df[self.df["split"] == "train"],
            hparams=self.my_hparams,
            transforms=self.transforms
        )

        # CASE 1: Imbalanced sampler
        if self.my_hparams.get("imbalanced_sampler"):
            LOGGER.info("Using imbalanced sampler for training!")
            sampler = ImbalancedDatasetSampler(train_dataset)
            self.train_dataloader_params["sampler"] = sampler
            self.train_dataloader_params["shuffle"] = False

        # Create DataLoader with parameters specified
        return DataLoader(train_dataset, **self.train_dataloader_params)


    def val_dataloader(self):
        """
        Returns DataLoader for validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for validation data
        """
        # Instantiate CXRDatasetDataFrame
        df_val = self.df[self.df["split"] == "val"]
        val_dataset = CXRDatasetDataFrame(df_val, self.my_hparams)

        # Create DataLoader with parameters specified
        return DataLoader(val_dataset, **self.val_dataloader_params)


    def test_dataloader(self):
        """
        Returns DataLoader for test set.

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for test data
        """
        # Instantiate CXRDatasetDataFrame
        df_test = self.df[self.df["split"] == "test"]
        test_dataset = CXRDatasetDataFrame(df_test, self.my_hparams)

        # Create DataLoader with parameters specified
        return DataLoader(test_dataset, **self.val_dataloader_params)


    def get_dataloader(self, split):
        """
        Get specific data loader

        Parameters
        ----------
        split : str
            Split must be one of train/val/test

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader
        """
        assert split in ("train", "val", "test")
        split_to_func = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader
        }
        return split_to_func[split]()


    def get_filtered_dataloader(self, split, **filters):
        """
        Get data loader for a specific split with option to filter for specific
        dataset.

        Parameters
        ----------
        split : str
            Name of data split to load. One of (train/val/test)
        filters : Any
            Keyword arguments, containing row filters.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader with filtered data
        """
        # Create copy of data to restore later
        df_orig = self.df.copy()

        # Split data into specific split and rest
        split_mask = self.df["split"] == split
        df_split = self.df[split_mask].copy()
        df_rest = self.df[~split_mask].copy()

        # If provided, filter out rows in the specific split
        df_split = self.filter_metadata(split=split, **filters)

        # Modify stored metadata table
        # NOTE: So that created dataloader would use applied filters
        self.df = pd.concat([df_split, df_rest], ignore_index=True)
        # Construct dataloader
        dataloader = self.get_dataloader(split)

        # Restore original table
        self.df = df_orig

        return dataloader


    ############################################################################
    #                           Helper Functions                               #
    ############################################################################
    def set_kfold_index(self, fold):
        """
        If performing cross-validation, sets fold index.

        Note
        ----
        Fold index can range between [0, num_folds - 1]

        Parameters
        ----------
        fold : int
            Fold index
        """
        assert hasattr(self, "cross_val_folds")
        assert fold in list(range(self.cross_val_folds))

        self.fold = fold

        # Set training and validation data for fold
        train_ids, val_ids = self.cross_fold_indices[self.fold]

        # Split dataset into train/val and test
        df_train_val = self.df_orig[self.df_orig["split"].isin(["train", "val"])]
        df_test = self.df_orig[self.df_orig["split"] == "test"]

        # Assign training/validation splits
        train_ids = set(train_ids)
        df_train_val["split"] = df_train_val["image_id"].map(
            lambda x: "train" if x in train_ids else "val"
        )

        # Check that folds were assigned correctly
        empirical_val_ids = set(df_train_val[df_train_val["split"] == "val"]["image_id"].unique())
        assert val_ids == empirical_val_ids, f"[K-Fold Validation #{fold}] Validation set was not assigned correctly!"

        # Concatenate new splits
        self.df = pd.concat([df_train_val, df_test], ignore_index=True)

        # Log number of data points
        num_train = (self.df["split"] == "train").sum()
        num_val = (self.df["split"] == "val").sum()
        num_test = (self.df["split"] == "test").sum()
        LOGGER.info(f"[K-Fold Validation #{fold}]")
        LOGGER.info(f"\tNum. Training Images: {num_train}")
        LOGGER.info(f"\tNum. Validation Images: {num_val}")
        LOGGER.info(f"\tNum. Test Images: {num_test}")


    def filter_metadata(self, dset=None, split=None, **filters):
        """
        Get metadata filtered for dataset or split.

        Parameters
        ----------
        dset : str, optional
            Dataset to filter for
        split : str, optional
            Data split to filter for, by default None
        **filters : Any
            Column to value keyword arguments to filter
        """
        df = self.df.copy()

        # Filter on dataset and data split
        if dset is not None:
            df = df[df["dset"] == dset]
        if split is not None:
            df = df[df["split"] == split]

        # If provided, perform other filters
        if filters:
            for col, val in filters.items():
                # Raise errors, if column not present
                if col not in df:
                    raise RuntimeError(f"Column {col} not in table provided!")

                # CASE 1: Value is a list/tuple
                if isinstance(val, (list, tuple, set)):
                    mask = df[col].isin(val)
                    df = df[mask]
                # CASE 2: Value is a single item
                else:
                    mask = (df[col] == val)
                    df = df[mask]

        return df


    def size(self, split=None):
        """
        Get size of dataset or specific data split

        Parameters
        ----------
        split : str, optional
            Data split (train/val/test)

        Returns
        -------
        list
            List of patient IDs
        """
        return len(self.filter_metadata(split=split))


################################################################################
#                               Dataset Classes                                #
################################################################################
class CXRDatasetDataFrame(torch.utils.data.Dataset):
    """
    Dataset to load images and labels from a DataFrame.
    """
    def __init__(self, df, hparams, transforms=None):
        """
        Initialize CXRDatasetDataFrame object.

        Parameters
        ----------
        df : pd.DataFrame
            Contains path to images and labels.
        hparams : dict
            Contains hyperparameters
        transforms : dict, optional
            Contains image transforms, by default None
        """
        self.df = df
        self.hparams = hparams

        ########################################################################
        #                           Image Transforms                           #
        ########################################################################
        # Add Resize and Normalize transforms
        self.transforms = insert_standardizing_transforms(hparams, transforms)

        # If specified, add HistogramMatching transform
        if hparams.get("transform_hm") and hparams.get("transform_hm_src_dset"):
            LOGGER.info("[CXRDatasetDataFrame] Adding HistogramMatching transform!")
            self.transforms["domain_adaptation"] = create_histogram_matching_transform(hparams)


    def __getitem__(self, index):
        """
        Loads an image with metadata, or a group of images from the same US
        sequence.

        Parameters
        ----------
        index : int
            Integer index to paths.

        Returns
        -------
        tuple
            Contains torch.Tensor and dict (containing metadata). Metadata may
            include path to image and label.
        """
        label_col = self.hparams.get("label_col", "label")
        # Get metadata for row
        row = self.df.iloc[index].to_dict()
        label = row[label_col]
        image_id = row["image_id"]
        assert label is not None, f"Image `{image_id}` has an invalid `{label_col}` label!"

        # Load image
        img_path = os.path.join(row["dirname"], row["filename"])
        X = load_image(
            img_path,
            type_to_transforms=self.transforms,
            img_mode=self.hparams.get("img_mode", 3),
        )

        # Store metadata
        # NOTE: Assumes label is integer 0 or 1
        metadata = {
            "label": int(label),
            "image_id": image_id,
            "dset": row["dset"],
            "split": row["split"],
        }
        return X, metadata


    def __len__(self):
        """
        Return number of items in the dataset. 

        Returns
        -------
        int
            Number of items in dataset
        """
        return len(self.df)


    def get_labels(self):
        """
        Get labels for dataset.

        Returns
        -------
        np.array
            List of labels
        """
        return self.df[self.hparams.get("label_col", "label")].to_numpy()


################################################################################
#                               Helper Functions                               #
################################################################################
def load_metadata(dset="vindr_cxr", label_col="label", filter_negative=True):
    """
    Load a metadata table for a specific dataset.

    Note
    ----
    `filter_negative` is useful when label column is False (e.g., no cardiomegaly
    but there is another disease present, and we'd want no cardiomegaly ==
    no finding)

    Parameters
    ----------
    dset : str, optional
        Name of the dataset. Can be one of the following: ("vindr_cxr",
        "vindr_pcxr"). By default "vindr_cxr".
    label_col : str, optional
        Name of the column containing labels. By default "label"
    filter_negative : bool, optional
        If True, filter out images that are negative for the label to only be
        the ones with no finding. By default True

    Returns
    -------
    pd.DataFrame
        Metadata table for the dataset
    """
    assert dset in constants.DIR_METADATA_MAP, \
        f"Unknown dataset: {dset}. List of valid datasets: {constants.DIR_METADATA_MAP.keys()}"
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["image"])

    # If label column isn't present, then simply return here
    if label_col not in df_metadata.columns:
        LOGGER.warning(f"Missing label column `{label_col}` in metadata for dset: `{dset}`! Adding label column with -1 label...")
        df_metadata[label_col] = -1
        return df_metadata

    # If specified, filter negatives for those with no finding
    if filter_negative:
        LOGGER.info(f"[Setup] Ensuring that negatives in  `{dset}` have strictly 'no finding'...")

        # Filter for those with no findings
        df_negatives = df_metadata[~df_metadata["Has Finding"].fillna(False).astype(bool)]

        # Get those with findings in column
        mask = df_metadata[label_col].fillna(False).astype(bool)
        df_positives = df_metadata[mask]
        df_metadata = pd.concat([df_positives, df_negatives], ignore_index=True)

    # Drop rows where label column is uncertain (-1)
    df_metadata = df_metadata[df_metadata[label_col] != -1]

    # SPECIAL CASE: CheXBERT
    # NOTE: Assume that any missing value in column is "No Finding"
    if dset == "chexbert":
        LOGGER.info(f"[CheXBERT] Filling missing in label column `{label_col}` with 0")
        df_metadata[label_col] = df_metadata[label_col].fillna(0)

    return df_metadata


def load_dataset(df, hparams, **kwargs):
    """
    Load Dataset object

    Parameters
    ----------
    df : pd.DataFrame
        Metadata table
    hparams : dict
        Experiment hyperparameters

    Returns
    -------
    torch.utils.data.Dataset
        Dataset object
    """
    assert hparams["dset"] in constants.DIR_METADATA_MAP, (
        f"Unknown dataset: {hparams['dset']}. " 
        f"List of valid datasets: {constants.DIR_METADATA_MAP.keys()}"
    )
    return CXRDatasetDataFrame(df, hparams, **kwargs)


def load_dataset_from_paths(img_paths, hparams, labels=None, **kwargs):
    """
    Create a CXRDatasetDataFrame from a list of paths to images.

    Parameters
    ----------
    img_paths : list of str
        List of paths to images
    hparams : dict
        Experiment hyperparameters
    labels : list of int, optional
        List of corresponding labels for images, by default None
    **kwargs : Any
        Additional keyword arguments to pass into CXRDatasetDataFrame

    Returns
    -------
    CXRDatasetDataFrame
        Dataset object
    """
    # Prepre metadata dataframe
    df_metadata = pd.DataFrame({"path": img_paths})
    df_metadata["dset"] = ""
    df_metadata["split"] = ""
    df_metadata[hparams.get("label_col", "label")] = labels if labels is not None else -1
    df_metadata["image_id"] = df_metadata.index
    df_metadata["dirname"] = df_metadata["path"].apply(os.path.dirname)
    df_metadata["filename"] = df_metadata["path"].apply(os.path.basename)

    # Create dataset
    return CXRDatasetDataFrame(df_metadata, hparams, **kwargs)


################################################################################
#                          Image Loading / Transforms                          #
################################################################################
def insert_standardizing_transforms(
        hparams,
        type_to_transforms=None,
        normalize=True,
    ):
    """
    Insert resize and normalize transforms into the transforms dictionary.
    These transforms are important for standardizing images.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters
    type_to_transforms : dict, optional
        Dictionary of transforms to apply to the image. Keys must be one of the
        following: ("texture", "geometric", "post-processing"). Values must be
        callable transforms from torchvision.transforms, by default None
    normalize : bool, optional
        If True, normalize images based on mean/std specified in `hparams`.

    Returns
    -------
    dict
        Updated dictionary of transforms
    """
    type_to_transforms = type_to_transforms if type_to_transforms is not None else {}
    # If image size specified, at Resize transform
    if hparams.get("img_size"):
        transform_type = "geometric"
        type_to_transforms[transform_type] = [type_to_transforms[transform_type]] if transform_type in type_to_transforms else []
        type_to_transforms[transform_type].insert(0, T.Resize(hparams.get("img_size")))
        type_to_transforms[transform_type] = T.Compose(type_to_transforms[transform_type])

    # Add ToTensor transform at the end, if there's no other post-processing
    transform_type = "post-processing"
    if normalize and type_to_transforms.get(transform_type) is None:
        type_to_transforms[transform_type] = [T.ToDtype(torch.float32, scale=True)]

        # Add normalization, only if there are parameters for it
        mean, std = hparams.get("norm_mean"), hparams.get("norm_std")
        if mean and std:
            LOGGER.info(f"[CXR Transforms] Adding Normalize(mean={mean}, sd={std}) transformation!")
            type_to_transforms[transform_type].append(
                create_normalizer(mean, std, hparams.get("img_mode", 3)))

        type_to_transforms[transform_type] = T.Compose(type_to_transforms[transform_type])
    else:
        LOGGER.warning("[CXR Transforms] Skipping Normalize transform!")
    return type_to_transforms


def load_image(img_path, type_to_transforms=None, img_mode=3, as_numpy=False):
    """
    Load image from path.

    Parameters
    ----------
    img_path : str
        Path to image
    type_to_transforms : dict, optional
        Dictionary of transforms to apply to the image. Keys must be one of the
        following: ("texture", "geometric", "post-processing"). Values must be
        callable transforms from torchvision.transforms, by default None
    img_mode : int, optional
        Number of channels (mode) to read images into (1=grayscale, 3=RGB),
        by default 3
    as_numpy : bool, optional
        If True, return image as numpy array, by default False

    Returns
    -------
    torch.Tensor
        Loaded image as a tensor
    """
    # Load image
    X = torchvision.io.read_image(img_path, mode=MAP_IMG_MODE[img_mode])

    # If image is UINT16, convert to UINT8
    if X.dtype == torch.uint16:
        # If the maximum value
        X = X.to(torch.float32)
        assert X.max() > 255, f"Image `{img_path}` has pixel value > 255! Max: {X.max()}"
        X = (X.float() / 256).clamp(0, 255).to(torch.uint8)

    # Assertion to ensure loaded images are between 0 and 255
    assert X.max() <= 255.0, f"Image `{img_path}` has pixel value > 255! Max: {X.max()}"

    # Apply transforms sequentially
    if type_to_transforms is not None:
        for transform_type in ["domain_adaptation", "texture", "geometric", "post-processing"]:
            if type_to_transforms.get(transform_type) is not None:
                # CASE 1: Domain adaptation (assumed albumentations interface)
                if transform_type == "domain_adaptation":
                    X = X.numpy()
                    X = np.transpose(X, (1, 2, 0)) if img_mode == 3 else X
                    X = type_to_transforms[transform_type](image=X)["image"]
                # CASE 2: Other transform (assumed torchvision interface)
                else:
                    X = type_to_transforms[transform_type](X)

    # If specified, convert to numpy array
    if as_numpy:
        X = X.numpy()
        X = np.transpose(X, (1, 2, 0)) if img_mode == 3 else X

    # TODO: Uncomment if normalizing between 0 and 1
    # assert X.max() <= 1.0, f"Image `{img_path}` has pixel value > 1! Max: {X.max()}"

    return X


def create_histogram_matching_transform(hparams):
    """
    Create a histogram matching transform.

    Parameters
    ----------
    hparams : dict
        Experiment hyperparameters, which contains the following keys:
        transform_hm_src_dset : str
            Name of source dataset for histogram matching
        transform_hm_blend_ratio : float
            Blend ratio for histogram matching

    Returns
    -------
    A.Compose
        Composed transform with histogram matching and conversion back to a PyTorch tensor
    """
    # Lazy import, since only used here
    try:
        import albumentations as A
        from albumentations.pytorch.transforms import ToTensorV2
    except ImportError:
        raise ImportError(
            "Histogram matching requires albumentations package. "
            "Install with `pip install albumentations`."
        )

    # Retrieve parameters
    seed = hparams.get("seed", 42)
    src_dset = hparams.get("transform_hm_src_dset")
    blend_ratio = hparams.get("transform_hm_blend_ratio", 1.0)

    # Early return, if not using histogram matching
    if not (hparams.get("transform_hm") and src_dset and blend_ratio):
        LOGGER.info("[CXR Transforms] Skipping HistogramMatching transform!")
        return None

    # Load metadata from source dataset
    df_src_metadata = load_metadata(dset=src_dset)

    # Filter for training set images
    if "split" in df_src_metadata.columns.tolist():
        df_src_metadata = df_src_metadata[df_src_metadata["split"] == "train"]

    # Sample 64 images
    df_src_metadata = df_src_metadata.sample(n=64, random_state=seed)

    # Create image paths
    img_paths = df_src_metadata.apply(
        lambda row: os.path.join(row["dirname"], row["filename"]), axis=1
    ).tolist()

    # Only resize reference images
    # NOTE: Normalization is only useful post histogram transform
    type_to_transforms = insert_standardizing_transforms(hparams, normalize=False)

    # Create histogram matching transform
    read_func_kwargs = {
        "type_to_transforms": type_to_transforms,
        "img_mode": hparams.get("img_mode", 3),
        "as_numpy": True,
    }
    hm_transform = A.HistogramMatching(
        reference_images=img_paths,
        blend_ratio=(blend_ratio, blend_ratio),
        p=1.0,
        read_fn=partial(load_image, **read_func_kwargs),
    )

    # Create transform with conversion back to a PyTorch tensor
    composed_transform = A.Compose([hm_transform, ToTensorV2()])
    LOGGER.info(
        "[CXR Transforms] Adding HistogramMatching transform with "
        f"{len(img_paths)} reference images from `{src_dset}`!"
    )

    return composed_transform


def create_normalizer(mean, std, img_mode=3):
    """
    Create normalizing transform for dataset.

    Parameters
    ----------
    mean : list or float
        Mean pixel value(s) of the dataset. If a list, then assume it's RGB.
    std : list or float
        Standard deviation(s) of the dataset. If a list, then assume it's RGB.
    img_mode : int, optional
        Number of color channels in images. If 3, then RGB. If 1, then grayscale.
        By default 3

    Returns
    -------
    torch.nn.Module
        Normalizing transform
    """
    # Ensure they're the right size
    # CASE 1: Only grayscale mean/std provided, but RGB mean needed
    if img_mode == 3 and len(mean) == 1:
        mean = mean * 3
        std = std * 3
    # CASE 2: Only RGB mean/std provided, but grayscale mean needed
    elif img_mode == 1 and len(mean) == 3:
        mean = [sum(mean) / 3]
        std = [sum(std) / 3]

    # Add normalizing transform
    return T.Normalize(mean=mean, std=std)


def compute_image_statistics(dataset, mode=1):
    """
    Compute mean and std of dataset

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset class used to load images
    mode : int
        If mode == 3, then images are RGB and function returns mean for each
        channel. Otherwise, return mean across 1+ channels
    """
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, os.cpu_count()-1)
    )

    # Compute mean and standard deviation
    mean = 0.0 if mode == 1 else torch.zeros(3)
    std = 0.0 if mode == 1 else torch.zeros(3)
    total_images_count = 0
    for images, _ in dataloader:
        curr_batch_size = images.size(0)

        # Flatten images to (N, C, H, W) if RGB, or if grayscale (N, H, W)
        images = images.view(curr_batch_size, images.size(1), -1)

        # Average across (H, W)
        # CASE 1: RGB
        if images.size(1) == 3 and mode == 3:
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += curr_batch_size
        # CASE 2: Grayscale
        else:
            mean += images.mean((1, 2)).sum(0)
            std += images.std((1, 2)).sum(0)
            total_images_count += curr_batch_size

    # Compute average mean/std of channels
    mean /= total_images_count
    std /= total_images_count
    print(f"[Dataset Statistics] Mean={mean} \t|\tStd={std}")
    return mean, std


def visualize_dataset(dset):
    """
    Plot example images for dataset

    Parameters
    ----------
    dset : str
        Name of dataset
    """
    # Late import, since it's not needed elsewhere
    from src.utils.data import viz_data

    # Default hyperparameters
    hparams = {
        "dset": dset,
        "img_size": constants.IMG_SIZE,
        "img_mode": 1,
        "label_col": "Cardiomegaly",
    }

    # Load metadata and create dataset
    df_metadata = load_metadata(dset, "Cardiomegaly", True)
    dataset = load_dataset(df_metadata, hparams)

    # Visualize 25 randomly sampled images
    viz_data.plot_dataset_samples(dataset, dset=dset, num_samples=25)

    # Compute mean and std
    print(f"Dataset: {dset}")
    compute_image_statistics(dataset, mode=hparams["img_mode"])


def visualize_dataset_with_hm(dset, src_dset, blend_ratio=1.0):
    """
    Plot example images for dataset with histogram matching with a source dataset

    Parameters
    ----------
    dset : str
        Name of dataset
    src_dset : str
        Name of reference dataset
    blend_ratio : float, optional
        Histogram matching blending ratio
    """
    # Late import, since it's not needed elsewhere
    from src.utils.data import viz_data

    # Default hyperparameters
    hparams = {
        "dset": dset,
        "transform_hm": True,
        "transform_hm_src_dset": src_dset,
        "transform_hm_blend_ratio": blend_ratio,
        "img_size": constants.IMG_SIZE,
        "img_mode": 3,
        "label_col": "Cardiomegaly",
    }

    # Load metadata and create dataset
    df_metadata = load_metadata(dset, "Cardiomegaly", True)
    dataset = load_dataset(df_metadata, hparams)

    # Visualize 25 randomly sampled images
    viz_data.plot_dataset_samples(
        dataset,
        dset=f"{src_dset} (hm-{blend_ratio})",
        num_samples=9,
        shuffle=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "hm", dset),
        ext="png",
    )


if __name__ == "__main__":
    # from fire import Fire
    # Fire()

    dset = "vindr_pcxr"
    src_dsets = ["vindr_cxr"]
    blend_ratios = [0.25, 0.5, 0.75, 1.0]
    for blend_ratio in blend_ratios:
        for src_dset in src_dsets:
            visualize_dataset_with_hm(dset, src_dset, blend_ratio)
