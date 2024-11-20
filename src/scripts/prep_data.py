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
import torchvision
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.io import imsave
from skimage.transform import resize

# Custom libraries
from config import constants
from src.utils.data import utils as data_utils


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
FLAG_EXTRACT_METADATA = False

# Random seed
SEED = 42


################################################################################
#                                Main Functions                                #
################################################################################
def prep_vindr_cxr_metadata_dicom(data_dir=constants.DIR_DATA_MAP["vindr_cxr"]):
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

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64")

    # Extract metadata from every DICOM file
    dicom_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()
    accum_metadata = []
    for dicom_path in tqdm(dicom_paths, desc="Extracting DICOM metadata"):
        metadata = get_metadata_from_dicom(dicom_path)
        accum_metadata.append(metadata)
    df_extra_metadata = pd.DataFrame.from_dict(accum_metadata)
    df_metadata = pd.concat([df_metadata, df_extra_metadata], axis=1)

    # Save metadata
    save_path = constants.DIR_METADATA_MAP["vindr_cxr"]["dicom"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_metadata.to_csv(save_path, index=False)

    LOGGER.info("Preparing VinDr-CXR metadata...DONE")


def prep_vindr_cxr_metadata_post_process():
    """
    Post-processing for VinDr-CXR metadata.

    1. Creates a "Has Finding" column.
    2. Parses age (string) in years and months.
    3. Samples 10% of healthy adults from each age bin
    4. Creates a calibration set from 10% of Cardiomegaly, Healthy and Has Finding patients
    5. Saves the updated metadata to file.
    """
    dset = "vindr_cxr"

    # Ensure metadata file exists
    metadata_path = constants.DIR_METADATA_MAP[dset]["image"]
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            "[VinDr-CXR] Metadata file (PNG) doesn't exist for VinDr-CXR! "
            f"Expected at: {metadata_path}\n"
            "\tPlease ensure you've run the following functions:\n"
            "\t\tprep_vindr_cxr_metadata_dicom()\n"
            "\t\tprep_vindr_cxr_metadata_post_process()"
        )

    # Load metadata
    df_metadata = pd.read_csv(metadata_path)

    # Patient ID is the same as the image ID
    df_metadata["patient_id"] = df_metadata["image_id"]
    # Create a "Has Finding" column
    df_metadata["Has Finding"] = (1 - df_metadata["No finding"])
    # Add view
    df_metadata["view"] = "PA"

    # Parse age (string) in years and months
    df_metadata = data_utils.extract_age("vindr_cxr", df_metadata)

    # 1. From healthy adult patients w/ age, sample 20% of patients from each age bin
    # NOTE: Unlike other datasets, VinDr-CXR has much fewer age-annotated adults
    #   a. Get all healthy patients with age annotations
    age_col = "age_years"
    df_healthy_with_age = df_metadata[~df_metadata["Has Finding"].astype(bool)].dropna(subset=age_col)
    #   b. Perform sampling
    age_bins = [18, 25, 40, 60, 80, 100]
    df_healthy_test, _ = data_utils.sample_by_age_bins(
        df_healthy_with_age, age_bins,
        age_col=age_col,
        prop=0.2
    )
    #   c. Create training set from remaining patients
    df_train = df_metadata.drop(df_healthy_test.index)
    print(f"[VinDr-CXR] Number of healthy adults for evaluation: {len(df_healthy_test)}")

    # 2. Create calibration set from 10% of Cardiomegaly, Healthy and Has Finding patients
    has_cardiomegaly = df_train["Cardiomegaly"].astype(bool)
    has_finding = df_train["Has Finding"].astype(bool)
    accum_calib = [
        df_train[has_cardiomegaly].sample(frac=0.1, random_state=SEED),
        df_train[~has_finding].sample(frac=0.1, random_state=SEED),
        # NOTE: Sample patients with findings but not Cardiomegaly
        df_train[has_finding & ~has_cardiomegaly].sample(frac=0.1, random_state=SEED),
    ]
    df_calib = pd.concat(accum_calib, axis=0)
    df_train = df_train.drop(df_calib.index)
    print(f"[VinDr-CXR] Number of patients for calibration: {len(df_calib)}")
    print(f"[VinDr-CXR] Number of patients for training: {len(df_train)}")

    # Add splits
    df_train["split"] = "train"
    df_calib["split"] = "test_adult_calib"
    df_healthy_test["split"] = "test_healthy_adult"

    # Concatenate datasets
    df_metadata = pd.concat([df_train, df_calib, df_healthy_test], axis=0)

    # Reorder the columns
    cols = [
        "dset", "split", "dirname", "filename", "patient_id", "image_id",
        "age_years", "age_months", "sex", "view",
        "Has Finding", "Cardiomegaly",
    ]
    cols += [col for col in df_metadata.columns if col not in set(cols)]
    df_metadata = df_metadata[cols]

    # Save metadata
    df_metadata.to_csv(metadata_path, index=False)


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

    # Parse age in years and months
    df_metadata = data_utils.extract_age("vindr_pcxr", df_metadata)

    # Drop index column
    df_metadata = df_metadata.drop(columns=["index"])

    # Save metadata
    save_path = constants.DIR_METADATA_MAP["vindr_pcxr"]["dicom"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_metadata.to_csv(save_path, index=False)

    LOGGER.info("Preparing VinDr-PCXR metadata...DONE")


def prep_nih_cxr18_metadata(data_dir=constants.DIR_DATA_MAP["nih_cxr18"]):
    """
    Prepare metadata for the NIH Chest X-ray 18 dataset.

    This function processes the NIH Chest X-ray 18 dataset by loading metadata,
    extracting relevant information, and splitting the data into training and
    test sets based on age and health findings.

    Parameters
    ----------
    data_dir : str
        Path to the NIH Chest X-ray 18 dataset directory.
    """
    LOGGER.info("Preparing NIH Chest X-ray 18 metadata...")

    # Ensure that image directory exists
    assert os.path.isdir(data_dir), \
        f"NIH Chest X-ray 18 directory doesn't exist! `{data_dir}` "

    # Load train/test metadata
    df_metadata = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"))

    # Remove the last three columns
    df_metadata = df_metadata.iloc[:, :-5]

    # Rename columns
    df_metadata = df_metadata.rename(columns={
        "Image Index": "filename",
        "Follow-up #": "visit",
        "View Position": "view",
        "Patient Gender": "gender",
        "Finding Labels": "Findings",
    })

    # Add dset and split columns
    # NOTE: Traing split will be changed to test after specifying data directory
    df_metadata["dset"] = "nih_cxr18"
    df_metadata["split"] = "train"

    # Process columns for paths and patient ID and visit number
    df_metadata["dirname"] = os.path.join(data_dir, "images-224")
    df_metadata["image_id"] = df_metadata["filename"]
    df_metadata["patient_id"] = df_metadata["filename"].map(lambda x: x.split("_")[0])

    # Create a "Has Finding" column
    df_metadata["Has Finding"] = (1 - (df_metadata["Findings"] == "No Finding").astype(int))

    # Create Cardiomegaly column
    df_metadata["Cardiomegaly"] = (df_metadata["Findings"].str.contains("Cardiomegaly")).astype(int)

    # Parse age in years and months
    df_metadata = data_utils.extract_age("nih_cxr18", df_metadata, age_col="Patient Age")
    print("[NIH] Number of patients without age annotations:", df_metadata["age_years"].isna().sum())

    # Sample 1 row from every patient
    # NOTE In order of priority: Has Age >> Has Cardiomegaly
    print(f"[NIH] Before sampling 1 image per patient: {len(df_metadata)}")
    df_metadata = df_metadata.groupby("patient_id").apply(sample_per_patient_multivisit_cxr)
    print(f"[NIH] After sampling 1 image per patient: {len(df_metadata)}")

    # Set aside pediatric data for evaluation
    mask = df_metadata["age_years"] < 18
    df_peds_test = df_metadata[mask].copy()
    df_metadata = df_metadata[~mask].copy()
    print(f"[NIH] Number of pediatric patients for evaluation: {len(df_peds_test)}")

    # 1. From healthy adult patients w/ age, sample 10% of patients from each age bin
    #   a. Get all healthy patients with age annotations
    df_healthy_with_age = df_metadata[~df_metadata["Has Finding"].astype(bool)].dropna(subset="age_years")
    #   b. Perform sampling
    age_bins = [18, 25, 40, 60, 80, 100]
    df_healthy_test, _ = data_utils.sample_by_age_bins(
        df_healthy_with_age, age_bins,
        age_col="age_years",
        prop=0.1
    )
    #   c. Create training set from remaining patients
    df_train = df_metadata.drop(df_healthy_test.index)
    print(f"[NIH] Number of healthy adults for evaluation: {len(df_healthy_test)}")

    # 2. Create calibration set from 10% of Cardiomegaly, Healthy and Has Finding patients
    has_cardiomegaly = df_train["Cardiomegaly"].astype(bool)
    has_finding = df_train["Has Finding"].astype(bool)
    accum_calib = [
        df_train[has_cardiomegaly].sample(frac=0.1, random_state=SEED),
        df_train[~has_finding].sample(frac=0.1, random_state=SEED),
        # NOTE: Sample patients with findings but not Cardiomegaly
        df_train[has_finding & ~has_cardiomegaly].sample(frac=0.1, random_state=SEED),
    ]
    df_calib = pd.concat(accum_calib, axis=0)
    df_train = df_train.drop(df_calib.index)
    print(f"[NIH] Number of patients for calibration: {len(df_calib)}")
    print(f"[NIH] Number of patients for training: {len(df_train)}")

    # Add splits
    df_train["split"] = "train"
    df_calib["split"] = "test_adult_calib"
    df_healthy_test["split"] = "test_healthy_adult"
    df_peds_test["split"] = "test_peds"

    # Concatenate datasets
    df_metadata = pd.concat([df_train, df_calib, df_peds_test, df_healthy_test], axis=0)

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64", errors="ignore")

    # Keep only the following columns
    cols = [
        "dset", "split", "dirname", "filename", "patient_id", "image_id", "visit",
        "age_years", "age_months", "gender", "view",
        "Has Finding", "Cardiomegaly", "Findings"
    ]
    df_metadata = df_metadata[cols]

    # Save metadata
    save_path = constants.DIR_METADATA_MAP["nih_cxr18"]["image"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_metadata.to_csv(save_path, index=False)

    LOGGER.info("Preparing NIH Chest X-ray 18 metadata...DONE")


def prep_padchest_metadata(data_dir=constants.DIR_DATA_MAP["padchest"]):
    """
    Prepare metadata for the PadChest CXR dataset.

    Parameters
    ----------
    data_dir : str
        Path to the PadChest CXR dataset directory.
    """
    LOGGER.info("Preparing PadChest CXR metadata...")

    # Ensure that image directory exists
    assert os.path.isdir(data_dir), \
        f"PadChest CXR directory doesn't exist! `{data_dir}` "

    # Load train/test metadata
    df_metadata = pd.read_csv(os.path.join(data_dir, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"))

    # Rename columns
    df_metadata = df_metadata.rename(columns={
        "StudyID": "patient_id",
        "ImageID": "filename",
        "PatientBirth": "birth_year",
        "ViewPosition_DICOM": "view",
        "PatientSex_DICOM": "sex", 
        "Labels": "Findings",
    })

    # Add dset, split and directory name
    # NOTE: Traing split will be changed to test after specifying data directory
    df_metadata["dset"] = "padchest"
    df_metadata["split"] = "train"
    df_metadata["dirname"] = os.path.join(data_dir, "images-224")
    df_metadata["image_id"] = df_metadata["filename"]

    # Remove images that don't exist
    img_paths = (df_metadata["dirname"] + "/" + df_metadata["filename"]).tolist()
    exists_masks = np.array([os.path.exists(path) for path in img_paths])
    df_metadata = df_metadata[exists_masks]
    print("[PC] Number of images (to remove) that don't exist:", (~exists_masks).sum())

    # Remove images with no findings/labels annotated
    print("[PC] Starting number of images:", len(df_metadata))
    mask_missing_labels = df_metadata["Findings"].isna()
    df_metadata = df_metadata[~mask_missing_labels]
    print("[PC] Number of images (to remove) with no findings annotated:", mask_missing_labels.sum())

    # Keep only PA view images
    map_view = {"POSTEROANTERIOR": "PA", "PA": "PA"}
    df_metadata["view"] = df_metadata["view"].map(map_view.get)
    pa_mask = ~df_metadata["view"].isna()
    df_metadata = df_metadata[pa_mask]
    print("[PC] Number of images (after filtering for PA views):", pa_mask.sum())

    # Create a "Has Finding" column
    df_metadata["Has Finding"] = (1 - (df_metadata["Findings"].str.lower().str.contains("'normal'")).astype(int))

    # Create Cardiomegaly column
    df_metadata["Cardiomegaly"] = (df_metadata["Findings"].str.lower().str.contains("'cardiomegaly'")).astype(int)

    # Create age column from birth year and year of study
    df_metadata["study_year"] = df_metadata["StudyDate_DICOM"].astype(str).map(
        lambda x: int(x[:4]) if x.isnumeric() else None
    )
    df_metadata["age_years"] = df_metadata["study_year"] - df_metadata["birth_year"]
    print("[PC] Number of patients without age annotations:", df_metadata["age_years"].isna().sum())

    # Sample 1 row from every patient
    # NOTE In order of priority: Has Age >> Has Cardiomegaly
    print(f"[PC] Before sampling 1 image per patient: {len(df_metadata)}")
    df_metadata = df_metadata.groupby("patient_id").apply(sample_per_patient_multivisit_cxr)
    print(f"[PC] After sampling 1 image per patient: {len(df_metadata)}")

    # Set aside pediatric data for evaluation
    mask = df_metadata["age_years"] < 18
    df_peds_test = df_metadata[mask].copy()
    df_metadata = df_metadata[~mask].copy()
    print(f"[PC] Number of pediatric patients for evaluation: {len(df_peds_test)}")

    # 1. From healthy adult patients w/ age, sample 10% of patients from each age bin
    #   a. Get all healthy patients with age annotations
    df_healthy_with_age = df_metadata[~df_metadata["Has Finding"].astype(bool)].dropna(subset="age_years")
    #   b. Perform sampling
    age_bins = [18, 25, 40, 60, 80, 100]
    df_healthy_test, _ = data_utils.sample_by_age_bins(
        df_healthy_with_age, age_bins,
        age_col="age_years",
        prop=0.1
    )
    #   c. Create training set from remaining patients
    df_train = df_metadata.drop(df_healthy_test.index)
    print(f"[PC] Number of healthy adults for evaluation: {len(df_healthy_test)}")

    # 2. Create calibration set from 10% of Cardiomegaly, Healthy and Has Finding patients
    has_cardiomegaly = df_train["Cardiomegaly"].astype(bool)
    has_finding = df_train["Has Finding"].astype(bool)
    accum_calib = [
        df_train[has_cardiomegaly].sample(frac=0.1, random_state=SEED),
        df_train[~has_finding].sample(frac=0.1, random_state=SEED),
        # NOTE: Sample patients with findings but not Cardiomegaly
        df_train[has_finding & ~has_cardiomegaly].sample(frac=0.1, random_state=SEED),
    ]
    df_calib = pd.concat(accum_calib, axis=0)
    df_train = df_train.drop(df_calib.index)
    print(f"[PC] Number of patients for calibration: {len(df_calib)}")
    print(f"[PC] Number of patients for training: {len(df_train)}")

    # Add splits
    df_train["split"] = "train"
    df_calib["split"] = "test_adult_calib"
    df_healthy_test["split"] = "test_healthy_adult"
    df_peds_test["split"] = "test_peds"

    # Concatenate datasets
    df_metadata = pd.concat([df_train, df_calib, df_peds_test, df_healthy_test], axis=0)

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64", errors="ignore")

    # Keep only the following columns
    cols = [
        "dset", "split", "dirname", "filename", "patient_id", "image_id",
        "age_years", "sex", "view",
        "Has Finding", "Cardiomegaly", "Findings"
    ]
    df_metadata = df_metadata[cols]

    # Save metadata
    save_path = constants.DIR_METADATA_MAP["padchest"]["image"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_metadata.to_csv(save_path, index=False)
    LOGGER.info("Preparing PadChest CXR metadata...DONE")


def prep_chexbert_metadata(data_dir=constants.DIR_DATA_MAP["chexbert"]):
    """
    Prepare metadata for the CheXBERT CXR dataset.

    Parameters
    ----------
    data_dir : str
        Path to the CheXBERT CXR dataset directory.
    """
    LOGGER.info("Preparing CheXBERT CXR metadata...")

    # Ensure that image directory exists
    assert os.path.isdir(data_dir), \
        f"CheXBERT CXR directory doesn't exist! `{data_dir}` "

    # Load train/validation metadata
    df_metadata = pd.concat([
        pd.read_csv(os.path.join(data_dir, fname))
        for fname in ["train_cheXbert.csv", "valid.csv"]
    ], ignore_index=True)

    # Rename columns
    df_metadata = df_metadata.rename(columns={
        "AP/PA": "view",
        "Sex": "sex", 
        "Age": "age_years",
    })

    # Extract patient, visit and image ID from path
    path_components = df_metadata["Path"].str.split("/")
    df_metadata["patient_id"] = path_components.str[2]
    df_metadata["visit"] = path_components.str[3]
    df_metadata["image_id"] = path_components.str[2:].str.join("-")
    df_metadata["dirname"] = path_components.str[1:-1].str.join("/").map(
        lambda x: os.path.join(data_dir, x)
    )
    df_metadata["filename"] = path_components.str[-1]

    # Add dset, split and directory name
    df_metadata["dset"] = "chexbert"
    df_metadata["split"] = "train"

    # Remove images with no findings/labels annotated
    finding_cols = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
        "Support Devices", "No Finding"
    ]
    print("[CheXBERT] Starting number of images:", len(df_metadata))
    mask_missing_labels = df_metadata[finding_cols].isna().all(axis=1)
    df_metadata = df_metadata[~mask_missing_labels]
    print("[CheXBERT] Number of images (to remove) with no findings annotated:", mask_missing_labels.sum())

    # Remove 3 patients with age == 0
    # NOTE: All other patients are aged 18+
    mask = (df_metadata["age_years"] == 0)
    df_metadata = df_metadata[~mask]
    print("[CheXBERT] Number of images (to remove) with age == 0:", (mask).sum())

    # Keep only PA view images
    map_view = {"POSTEROANTERIOR": "PA", "PA": "PA"}
    df_metadata["view"] = df_metadata["view"].map(map_view.get)
    pa_mask = ~df_metadata["view"].isna()
    df_metadata = df_metadata[pa_mask]
    print("[CheXBERT] Number of images (after filtering for PA views):", pa_mask.sum())

    # Create a "Has Finding" column
    # NOTE: Assume all patients without "No Finding" == 1 has a finding
    df_metadata["Has Finding"] = (1 - (df_metadata["No Finding"].fillna(0)).astype(int))

    # Log number of patients without age annotations
    print("[CheXBERT] Number of patients without age annotations:", df_metadata["age_years"].isna().sum())

    # Sample 1 row from every patient
    # NOTE In order of priority: Has Age >> Has Cardiomegaly
    print(f"[CheXBERT] Before sampling 1 image per patient: {len(df_metadata)}")
    df_metadata = df_metadata.groupby("patient_id").apply(sample_per_patient_multivisit_cxr)
    print(f"[CheXBERT] After sampling 1 image per patient: {len(df_metadata)}")

    # 1. From healthy adult patients w/ age, sample 10% of patients from each age bin
    #   a. Get all healthy patients (with age annotations) and no uncertain findings labels
    healthy_mask = (~df_metadata["Has Finding"].astype(bool)) & (df_metadata[finding_cols] != -1).all(axis=1)
    df_healthy_with_age = df_metadata[healthy_mask].dropna(subset="age_years")
    #   b. Perform sampling
    age_bins = [18, 25, 40, 60, 80, 100]
    df_healthy_test, _ = data_utils.sample_by_age_bins(
        df_healthy_with_age, age_bins,
        age_col="age_years",
        prop=0.1
    )
    #   c. Create training set from remaining patients
    df_train = df_metadata.drop(df_healthy_test.index)
    print(f"[CheXBERT] Number of healthy adults for evaluation: {len(df_healthy_test)}")

    # 2. Create calibration set from 10% of Cardiomegaly, Healthy and Has Finding patients
    has_cardiomegaly = df_train["Cardiomegaly"].fillna(0).astype(bool)
    has_finding = df_train["Has Finding"].fillna(0).astype(bool)
    accum_calib = [
        df_train[has_cardiomegaly].sample(frac=0.1, random_state=SEED),
        df_train[~has_finding].sample(frac=0.1, random_state=SEED),
        # NOTE: Sample patients with findings but not Cardiomegaly
        df_train[has_finding & ~has_cardiomegaly].sample(frac=0.1, random_state=SEED),
    ]
    df_calib = pd.concat(accum_calib, axis=0)
    df_train = df_train.drop(df_calib.index)
    print(f"[CheXBERT] Number of patients for calibration: {len(df_calib)}")
    print(f"[CheXBERT] Number of patients for training: {len(df_train)}")

    # Add splits
    df_train["split"] = "train"
    df_calib["split"] = "test_adult_calib"
    df_healthy_test["split"] = "test_healthy_adult"

    # Concatenate datasets
    df_metadata = pd.concat([df_train, df_calib, df_healthy_test], axis=0)

    # Cast all float columns to int
    for col in df_metadata.columns:
        if df_metadata[col].dtype == "float64":
            df_metadata[col] = df_metadata[col].astype("int64", errors="ignore")

    # Keep only the following columns
    cols = [
        "dset", "split", "dirname", "filename", "patient_id", "image_id",
        "age_years", "sex", "view",
        "Has Finding", "Cardiomegaly"
    ]
    cols += [col for col in df_metadata.columns if col not in set(cols)]
    df_metadata = df_metadata[cols]

    # Save metadata
    save_path = constants.DIR_METADATA_MAP["chexbert"]["image"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_metadata.to_csv(save_path, index=False)
    LOGGER.info("Preparing CheXBERT CXR metadata...DONE")


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
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP[dset]["dicom"])

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
    df_metadata.to_csv(constants.DIR_METADATA_MAP[dset]["image"], index=False)


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
        # Attempt to read image
        try:
            torchvision.io.read_image(save_path, mode="RGB")
            return
        except:
            LOGGER.info(f"[Processing CXR Image] Image exists, but failed to load! File: {save_path}")

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


def sample_per_patient_multivisit_cxr(df_patient, age_col="age_years", n=1, seed=SEED):
    """
    Sample one image per patient in the NIH dataset.

    First, filter for visits with valid age, if available.

    Then, sample a row based on the following order of priority:
    1. Visit with Cardiomegaly
    2. Visit with any finding
    3. Any row

    Parameters
    ----------
    df_patient : pd.DataFrame
        Patient-level data
    age_col : str, optional
        Name of column containing age information, by default "age_years"
    n : int, optional
        Number of samples to take, by default 1
    seed : int, optional
        Random seed for reproducibility, by default SEED

    Returns
    -------
    pd.DataFrame
        Sampled row
    """
    df_patient = df_patient.copy()

    # Early return, if already only 1 row left
    if len(df_patient) == 1:
        return df_patient

    # 1. Filter for visits with valid age
    has_age = ~df_patient[age_col].isna()
    # If has rows with age, then filter on those rows
    if has_age.any():
        df_patient = df_patient[has_age]

    # Early return, if already only 1 row left
    if len(df_patient) == 1:
        return df_patient

    # 2. If has visit with Cardiomegaly, then sample row
    if df_patient["Cardiomegaly"].sum():
        df_cardiomegaly = df_patient[df_patient["Cardiomegaly"].astype(bool)]
        if len(df_cardiomegaly) == 1:
            return df_cardiomegaly
        return df_cardiomegaly.sample(n=n, random_state=seed)

    # 3. If no Cardiomegaly, then sample freely
    return df_patient.sample(n=n, random_state=seed)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    # Set up interface
    Fire({
        # In order that it needs to be run to process VinDr datasets
        # 1. Extract metadata from DICOM
        "vindr_cxr_metadata_dicom": prep_vindr_cxr_metadata_dicom,
        "vindr_pcxr_metadata_dicom": prep_vindr_pcxr_metadata,
        # 2. Process DICOMs to PNGs
        "vindr_images": process_all_vindr_dicom_images,
        # 3. Post-process metadata to create samples
        "vindr_cxr_metadata_post_process": prep_vindr_cxr_metadata_post_process,

        # NIH Chest X-ray 18 dataset
        "nih_cxr18_metadata": prep_nih_cxr18_metadata,

        # PadChest dataset
        "padchest_metadata": prep_padchest_metadata,

        # CheXBERT dataset
        "chexbert_metadata": prep_chexbert_metadata,
    })
