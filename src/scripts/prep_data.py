"""
prep_metadata.py

Description: Prepare metadata files for VinDr-CXR and VinDr-PCXR datasets, using
             original annotation files and metadata in DICOM files.

Note: VinDr-PCXR is used primarily as an evaluation set.
"""

# Standard libraries
import gc
import logging
import os
import sys

# Non-standard libraries
import numpy as np
import pandas as pd
import pydicom
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.io import imsave
from skimage.transform import resize

# Custom libraries
from config import constants


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

# Flag to check if DICOM file exists before trying to read DICOM
FLAG_CHECK_FILE_EXISTS = False

# Flag to extract metadata from DICOM file
FLAG_EXTRACT_METADATA = True


################################################################################
#                                Main Functions                                #
################################################################################
def prep_vindr_cxr_metadata(data_dir=constants.DIR_DATA_MAP["vindr_cxr"]):
    """
    Load VinDr-CXR dataset and merge with metadata from DICOM files.

    Parameters
    ----------
    data_dir : str
        Path to VinDr-CXR dataset directory

    Returns
    -------
    pd.DataFrame
        Metadata table with metadata from DICOM files
    """
    LOGGER.info("Preparing VinDr-CXR metadata...")
    # Ensure that image directory exists
    assert os.path.isdir(data_dir), \
        f"VinDr-CXR directory doesn't exist! `{data_dir}` "

    # Ensure that metadata files exist
    metadata_dir = os.path.join(data_dir, "1.0.0", "annotations")
    assert os.path.isdir(metadata_dir), \
        f"VinDr-CXR directory `vindr-cxr/1.0.0/annotations` doesn't exist!"

    # Load train/test metadata
    df_train = pd.read_csv(os.path.join(metadata_dir, "image_labels_train.csv"))
    df_test = pd.read_csv(os.path.join(metadata_dir, "image_labels_test.csv"), index_col="image_id")
    df_test = df_test.rename(columns={"Other disease": "Other diseases"})

    # Drop radiologist column
    df_train = df_train.drop(columns=["rad_id"])

    # Do majority vote to handle disagreement between radiologists
    df_train = df_train.groupby(by=["image_id"]).apply(lambda df: (df.sum() / len(df)))

    # Print percentage of cases with disagreement
    in_disagreement = lambda df: round(100*(~df.iloc[:, 0].isin([0, 1])).mean(), 2)
    LOGGER.info(
        "==Percent of Data in Disagreement=="
        f"\n\tTrain: {in_disagreement(df_train)}%"
    )

    # Finish majority vote by rounding
    df_train = df_train.round()

    # Add dset and split columns
    df_train["dset"] = "vindr_cxr"
    df_train["split"] = "train"
    df_test["dset"] = "vindr_cxr"
    df_test["split"] = "test"

    # Combine tables
    df_metadata = pd.concat([df_train, df_test]).reset_index()

    # Extract metadata from each DICOM path
    df_metadata["dirname"] = df_metadata["split"].map(lambda x: os.path.join(data_dir, "1.0.0", x))
    df_metadata["filename"] = df_metadata["image_id"] + ".dicom"

    # Check that every DICOM exists
    dicom_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()
    if FLAG_CHECK_FILE_EXISTS:
        for dicom_path in tqdm(dicom_paths, desc="Checking DICOM file exists"):
            assert os.path.exists(dicom_path), f"File {dicom_path} does not exist!"

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64")

    # Extract metadata for every file
    if FLAG_EXTRACT_METADATA:
        accum_metadata = []
        for dicom_path in tqdm(dicom_paths, desc="Extracting DICOM metadata"):
            metadata = get_metadata_from_dicom(dicom_path)
            accum_metadata.append(metadata)
        df_extra_metadata = pd.DataFrame.from_dict(accum_metadata)
        df_metadata = pd.concat([df_metadata, df_extra_metadata], axis=1)

    # Create a "Has Finding" column
    df_metadata["Has Finding"] = (1 - df_metadata["No finding"])

    # Save metadata
    df_metadata.to_csv(constants.DIR_METADATA_MAP["vindr_cxr"], index=False)

    LOGGER.info("Preparing VinDr-CXR metadata...DONE")


def prep_vindr_pcxr_metadata(data_dir=constants.DIR_DATA_MAP["vindr_pcxr"]):
    """
    Load VinDr-PCXR dataset and merge with metadata from DICOM files.

    Parameters
    ----------
    data_dir : str
        Path to VinDr-PCXR dataset directory

    Returns
    -------
    pd.DataFrame
        Metadata table with metadata from DICOM files
    """
    LOGGER.info("Preparing VinDr-PCXR metadata...")

    # Ensure that image directory exists
    assert os.path.isdir(data_dir), \
        f"VinDr-PCXR directory doesn't exist! `{data_dir}` "

    # Ensure that metadata files exist
    metadata_dir = os.path.join(data_dir, "1.0.0")
    assert os.path.isdir(metadata_dir), \
        "VinDr-PCXR directory `vindr-pcxr/1.0.0/` doesn't exist!"

    # Load train/test metadata
    df_train = pd.read_csv(os.path.join(metadata_dir, "image_labels_train.csv"))
    df_test = pd.read_csv(os.path.join(metadata_dir, "image_labels_test.csv"))
    df_test = df_test.rename(columns={"Other disease": "Other diseases"})

    # Drop radiologist column
    df_train = df_train.drop(columns=["rad_ID"])
    df_test = df_test.drop(columns=["rad_ID"])

    # Add dset and split columns
    # NOTE: Traing split will be changed to test after specifying data directory
    df_train["dset"] = "vindr_pcxr"
    df_train["split"] = "train"
    df_test["dset"] = "vindr_pcxr"
    df_test["split"] = "test"

    # Combine tables
    df_metadata = pd.concat([df_train, df_test]).reset_index()

    # Extract metadata from each DICOM path
    df_metadata["dirname"] = df_metadata["split"].map(lambda x: os.path.join(data_dir, "1.0.0", x))
    df_metadata["filename"] = df_metadata["image_id"] + ".dicom"

    # Set all to test
    df_metadata["split"] = "test"

    # Check that every DICOM exists
    dicom_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()
    if FLAG_CHECK_FILE_EXISTS:
        for dicom_path in tqdm(dicom_paths, desc="Checking DICOM file exists"):
            assert os.path.exists(dicom_path), f"File {dicom_path} does not exist!"

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64", errors="ignore")

    # Extract metadata for every file
    if FLAG_EXTRACT_METADATA:
        accum_metadata = []
        for dicom_path in tqdm(dicom_paths, desc="Extracting DICOM metadata"):
            metadata = get_metadata_from_dicom(dicom_path)
            accum_metadata.append(metadata)
        df_extra_metadata = pd.DataFrame.from_dict(accum_metadata)
        df_metadata = pd.concat([df_metadata, df_extra_metadata], axis=1)

    # Create a "Has Finding" column
    df_metadata["Has Finding"] = (1 - df_metadata["No finding"])

    # Drop index column
    df_metadata = df_metadata.drop(columns=["index"])

    # Save metadata
    df_metadata.to_csv(constants.DIR_METADATA_MAP["vindr_pcxr"], index=False)

    LOGGER.info("Preparing VinDr-PCXR metadata...DONE")


def process_all_vindr_dicom_images(dset="vindr_cxr"):
    """
    Processes DICOM images from the VinDr dataset, saving them as PNGs.

    Parameters
    ----------
    dset : str, optional
        The dataset to process, either "vindr_cxr" or "vindr_pcxr". Default is "vindr_cxr".

    Notes
    -----
    - Creates a directory for processed images if not already present.
    - Skips processing if images are already processed.
    - Uses parallel processing to convert DICOM images to PNG format.
    """
    # Create a directory to store processed images
    dir_processed = os.path.join(constants.DIR_DATA_MAP[dset], "1.0.0", "train_test_processed")
    os.makedirs(dir_processed, exist_ok=True)

    # Load metadata
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset])

    # Skip, if already in processed directory
    if list(df_metadata["dirname"].unique()) == [dir_processed]:
        LOGGER.info(f"[Processing CXR Images] Skipping `{dset}`! Already processed...")
        return

    # Get DICOM path
    dicom_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()

    # Create new path as PNG in processed directory
    save_paths = [
        os.path.join(dir_processed, os.path.basename(path).replace(".dicom", ".png"))
        for path in dicom_paths
    ]

    # Process images in parallel
    num_cpus = os.cpu_count()
    LOGGER.info(f"Processing {len(dicom_paths)} images with {num_cpus} CPUs...")
    Parallel(n_jobs=num_cpus)(
        delayed(process_vindr_dicom_image)(
            dicom_path=dicom_path,
            save_path=save_path,
        )
        for dicom_path, save_path in zip(dicom_paths, save_paths)
    )

    LOGGER.info(f"Processing {len(dicom_paths)} images with {num_cpus} CPUs...DONE")

    # Modify dataframe to point to new paths
    df_metadata["dirname"] = dir_processed
    df_metadata["filename"] = df_metadata["filename"].apply(lambda x: x.replace(".dicom", ".png"))
    df_metadata.to_csv(constants.DIR_METADATA_MAP[dset], index=False)


################################################################################
#                               Helper Functions                               #
################################################################################
def get_metadata_from_dicom(dicom_path):
    """
    Extracts metadata from a DICOM file.

    Parameters
    ----------
    dicom_path : str
        Path to the DICOM file.

    Returns
    -------
    dict
        Contains the extracted metadata:
            - "sex": Sex of the patient
            - "age": Age of the patient
            - "patient_size": Size of the patient
            - "patient_weight": Weight of the patient
    """
    dicom_obj = pydicom.filereader.dcmread(dicom_path)

    # Get metadata
    metadata = {}
    metadata["sex"] = getattr(dicom_obj.get((0x10, 0x40)), "value", None)
    metadata["age"] = getattr(dicom_obj.get((0x10, 0x1010)), "value", None)
    metadata["patient_size"] = getattr(dicom_obj.get((0x10, 0x1020)), "value", None)
    metadata["patient_weight"] = getattr(dicom_obj.get((0x10, 0x1030)), "value", None)

    # Post-process to replace null placeholders
    for key, val in metadata.items():
        if val == "O":
            metadata[key] = None

    return metadata


def process_vindr_dicom_image(dicom_path, save_path, **kwargs):
    """
    Processes a DICOM image from the VinDr dataset and saves it as a PNG file.

    Parameters
    ----------
    dicom_path : str
        Path to the DICOM image file.
    save_path : str
        Path where the processed PNG image will be saved.
    **kwargs : Any
        Additional keyword arguments to pass into `load_vindr_dicom_image`.
    """
    # Ignore, if already exists
    if os.path.exists(save_path):
        return

    # Load image
    img_arr = (255 * load_vindr_dicom_image(dicom_path, **kwargs))

    # Resize image
    resized_img = resize(img_arr, (224, 224), anti_aliasing=True)

    # Convert image to int8
    resized_img = resized_img.astype(np.uint8)

    # Save image
    imsave(save_path, resized_img)

    # Do garbage collection
    del img_arr, resized_img
    gc.collect()


# Following code has been adapted from torchxrayvision package, for the purpose of classification
# Link: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L1744
def load_vindr_dicom_image(dicom_path, **kwargs):
    """
    Loads a DICOM image from the VinDr dataset and normalizes to [0, 1].

    Parameters
    ----------
    dicom_path : str
        Path to the DICOM image file
    **kwargs : Any
        Keyword arguments to pass into `normalize`

    Returns
    -------
    np.array
        DICOM image normalized between 0 and 1
    """
    # Load DICOM object
    dicom_obj = pydicom.filereader.dcmread(dicom_path)
    img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
    img = pydicom.pixel_data_handlers.apply_windowing(img, dicom_obj)

    # Set image dtype as double to increase precision in operations
    img = img.astype(np.float64)

    # Photometric Interpretation to see if the image needs to be inverted
    mode = dicom_obj[0x28, 0x04].value
    bitdepth = dicom_obj[0x28, 0x101].value

    # HACK: Assumes 8-bits per pixel if max intensity is < 256
    if img.max() < 256:
        bitdepth = 8

    # Invert image, if necessary
    if mode == "MONOCHROME1":
        img = -1 * img + 2**float(bitdepth)
    elif mode == "MONOCHROME2":
        pass
    else:
        raise Exception("Unknown Photometric Interpretation mode")

    # Normalize image
    img = normalize(img, maxval=2**float(bitdepth), reshape=True, **kwargs)

    return img


def normalize(img, maxval, reshape=False, rgb=True):
    """
    Normalize image intensities to approximately [0, 1]

    Parameters
    ----------
    img : numpy.ndarray
        Image to be normalized
    maxval : int
        Maximum intensity value in the image
    reshape : bool, optional
        Reshape the image to have a color channel, by default False
    rgb : bool, optional
        If True, add color channel to the image, by default True

    Returns
    -------
    numpy.ndarray
        Normalized image
    """
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = img / maxval

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    # Convert grayscale image of shape (H, W, 1) to RGB image
    if rgb:
        # Add color channel if grayscale image of shape (H, W)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0))

        # Convert from grayscale to RGB
        img = np.tile(img, (1, 1, 3))

    return img


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Set up interface
    Fire({
        "vindr_cxr_metadata": prep_vindr_cxr_metadata,
        "vindr_pcxr_metadata": prep_vindr_pcxr_metadata,
        "vindr_images": process_all_vindr_dicom_images,
    })
