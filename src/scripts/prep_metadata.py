"""
prep_metadata.py

Description: Prepare metadata files for VinDr-CXR and VinDr-PCXR datasets, using
             original annotation files and metadata in DICOM files.

Note: VinDr-PCXR is used primarily as an evaluation set.
"""

# Standard libraries
import logging
import os
import sys

# Non-standard libraries
import pandas as pd
import pydicom
from fire import Fire
from tqdm import tqdm

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
    df_metadata.to_csv(
        os.path.join(constants.DIR_DATA_MAP["metadata"], "vindr_cxr_metadata.csv"),
        index=False
    )

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
    df_metadata.to_csv(
        os.path.join(constants.DIR_DATA_MAP["metadata"], "vindr_pcxr_metadata.csv"),
        index=False
    )

    LOGGER.info("Preparing VinDr-PCXR metadata...DONE")


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


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Set up interface
    Fire({
        "vindr_cxr": prep_vindr_cxr_metadata,
        "vindr_pcxr": prep_vindr_pcxr_metadata
    })