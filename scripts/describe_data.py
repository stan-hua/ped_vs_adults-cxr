"""
describe_data.py

Description: Used to create figures from open medical imaging metadata
"""

# Standard libraries
import ast
import logging
import math
import json
import re
import os
import warnings
from collections import defaultdict
from functools import reduce

# Non-standard libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fire import Fire
from matplotlib.colors import ListedColormap
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

# Custom libraries
from config import constants
from src.utils.data import viz_data
from src.utils.misc.openneuro_utils import OpenNeuroExtractor


################################################################################
#                                    Config                                    #
################################################################################
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings("ignore")

# Configure plotting
viz_data.set_theme(figsize=(16, 8), tick_scale=3)


################################################################################
#                                  Constants                                   #
################################################################################
# Mapping of data category to sheet number in XLSX metadata file
SHEET_ORDER = [
    "papers",               # MIDL Papers
    "midl_datasets",        # MIDL Datasets
    "challenges",
    "benchmarks",
    "datasets",
    "collections",
    "stanford_aimi",        # Image Collection (Stanford AIMI)
    "tcia",                 # Image Collection (TCIA)
    "midrc_parsed",         # Image Collection (MIDRC)
    "openneuro_parsed",     # Image Collection (OpenNeuro)
    "repackaging_1st",      # Repackaging (Primary)
    "repackaging_2nd",      # Repackaging (Secondary)
    "vqa_rad",              # Repackaging (VQA-RAD)
    "medsam",               # Repackaging (MedSAM)
]
CATEGORY_TO_SHEET = {sheet: idx for idx, sheet in enumerate(SHEET_ORDER)}

# Mapping of data category to string
CATEGORY_TO_STRING = {
    "challenges": "Challenges",
    "benchmarks": "Benchmarks",
    "datasets": "Well-Cited Datasets",
    "collections": "Data Collections",
    "papers": "Conference Papers",
    "collections_datasets": "Data Collections",
    # Individual dataset collections
    "midrc_parsed": "Data Collection (MIDRC)",
    "openneuro_parsed": "Data Collection (OpenNeuro)",
    "stanford_aimi": "Data Collection (Stanford AIMI)",
    "tcia": "Data Collection (TCIA)",
}

# Mapping of data category to directory
CATEGORY_TO_DIRECTORY = {
    "challenges": constants.DIR_FIGURES_EDA_CHALLENGES,
    "benchmarks": constants.DIR_FIGURES_EDA_BENCHMARKS,
    "datasets": constants.DIR_FIGURES_EDA_DATASETS,
    "papers": constants.DIR_FIGURES_EDA_PAPERS,
    "collections": constants.DIR_FIGURES_EDA_COLLECTIONS,
    "midrc_parsed": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "midrc"),
    "openneuro_parsed": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "openneuro"),
    "stanford_aimi": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "stanford_aimi"),
    "tcia": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "tcia"),
}

# Order to display categories
CATEGORY_ORDER = ["Conference Papers", "Data Collections", "Challenges", "Benchmarks", "Well-Cited Datasets"]

# Metadata columns
DEMOGRAPHICS_COL = "Patient Demographics / Covariates / Metadata"
MODALITY_COL = "Imaging Modality(ies)"
ORGAN_COL = "Organ / Body Part"
TASK_COL = "Task Category"
TASK_PATH_COL = "Task / Pathology"
CONTAINS_CHILDREN_COL = "Contains Children"
HAS_FINDINGS_COL = "Patients With Findings"
SOURCE_COL = "Source / Institutions (Location)"
PEDS_VS_ADULT_COL = "Peds vs. Adult"
AGE_DOCUMENTED_HOW_COL = "Age Documented How"
SAMPLE_SIZE_COL = "Sample Size"


################################################################################
#                                   Classes                                    #
################################################################################
class OpenDataVisualizer:
    """
    OpenDataVisualizer class.

    Notes
    -----
    Used to create figures from annotated challenges metadata
    """

    def __init__(self, df_metadata, data_category="challenges", create_plots=True, save=True):
        self.df_metadata = df_metadata
        self.data_category = data_category
        self.create_plots = create_plots
        self.data_category_str = CATEGORY_TO_STRING[data_category]
        self.save_dir = CATEGORY_TO_DIRECTORY[data_category] if save else None

        # Store constants to be filled in
        self.descriptions = {}


    def describe(self):
        self.descriptions["Number of Datasets"] = len(self.df_metadata)
        self.descriptions["Number of Datasets (With Age)"] = int(self.df_metadata[PEDS_VS_ADULT_COL].notnull().sum())

        # Parse sample size and age columns
        self.parse_sample_size_column()
        self.parse_age_columns()
        self.parse_demographics_columns()

        # Call functions
        self.describe_data_provenance()
        self.describe_patients()
        if self.data_category != "collections":
            self.describe_tasks()

        return self.descriptions


    def describe_data_provenance(self):
        """
        Describe data provenance-related information.

        1. Where is the data hosted?
        2. What organization/conference hosted the challenge?
        3. What proportion of the challenges are missing data provenance?
        4. From what institutions contribute the most data
        5. From what countries contribute the most data
        6. What proportion of challenges contain secondary data?
        """
        # Create copy to avoid in-place modifications
        df_metadata = self.df_metadata.copy()
        save_dir = self.save_dir
        data_category_str = self.data_category_str

        # 1. Where is the data hosted?
        # NOTE: Ignore if it's on Kaggle (but unofficial)
        if self.create_plots and "Data Location" in df_metadata.columns:
            df_metadata["Data Location"] = df_metadata["Data Location"].str.split(",").str[0]
            viz_data.catplot(
                df_metadata, x="Data Location", hue="Data Location",
                xlabel="", ylabel=f"Number of {data_category_str}",
                order=df_metadata["Data Location"].value_counts().index,
                plot_type="count",
                title=f"What Website is the {data_category_str} Data Hosted On?",
                save_dir=save_dir,
                save_fname="data_location(bar).png",
            )

        # 2. What organization/conference hosted the challenge?
        if self.create_plots and self.data_category == "challenges":
            conferences = df_metadata["Conference"].str.split(", ").str.join(" & ")
            self.descriptions["Prop. of Challenges in Conference"] = sum(conferences.notnull()) / len(conferences)
            conferences = conferences.fillna("None")
            viz_data.catplot(
                conferences.to_frame(), y="Conference", hue="Conference",
                xlabel="Number of Challenges", ylabel="",
                plot_type="count",
                order=conferences.value_counts().index,
                title="What Conference is the Challenge Featured In?",
                save_dir=save_dir,
                save_fname="conference(bar).png",
            )

        # 3. What proportion of the challenges are missing data provenance?
        # 3.1. Split source column into list of locations
        df_metadata[SOURCE_COL] = df_metadata[SOURCE_COL].str.split("\n")

        # 3.2. Plot proportion of challenges with complete/partially complete/missing data source
        df_metadata["Is Data Source Known"] = df_metadata[SOURCE_COL].map(
            lambda x: "Missing" if not isinstance(x, list) else (
                "Partial" if any("N/A" in item for item in x) else "Complete"
            )
        )
        if self.create_plots:
            viz_data.catplot(
                df_metadata, x="Is Data Source Known", hue="Is Data Source Known",
                xlabel="", ylabel=f"Number of {data_category_str}",
                plot_type="count",
                order=["Complete", "Partial", "Missing"],
                title="Do We Know Where The Data Is From?",
                save_dir=save_dir,
                save_fname="data_source(bar).png",
            )

        # 4. From what institutions contribute the most data
        institute_country = df_metadata[SOURCE_COL].dropna().explode()
        institutions = institute_country[~institute_country.str.contains("N/A")]
        institutions = institutions.str.split(r" \(").str[0]
        institutions.name = "Institutions"
        self.descriptions["Number of Unique Institutions"] = int(institutions.nunique())
        if self.create_plots:
            viz_data.catplot(
                institutions.to_frame(), y="Institutions", hue="Institutions",
                xlabel=f"Number of {data_category_str}", ylabel="",
                plot_type="count",
                order=institutions.value_counts().index,
                title="What Institutions Contribute the Most Data?",
                save_dir=save_dir,
                save_fname="institutions(bar).png",
            )

        # 5. From what countries contribute the most data
        institute_country = df_metadata[SOURCE_COL].dropna().explode()
        countries = institute_country.str.split(r" \(").str[1].str.split(")").str[0]
        countries.name = "Country"
        countries.reset_index(drop=True, inplace=True)
        self.descriptions["Number of Unique Countries"] = int(countries.nunique())
        if self.create_plots:
            viz_data.catplot(
                countries.to_frame(), y="Country", hue="Country",
                xlabel=f"Number of {data_category_str}", ylabel="",
                plot_type="count",
                order=countries.value_counts().index,
                title="What Countries Contribute the Most Data?",
                save_dir=save_dir,
                save_fname="countries(bar).png",
            )

        # 6. What proportion of challenges contain secondary data?
        # TODO: Need to filter out those whose data provenance is missing
        contains_secondary = df_metadata["Considerations"].str.lower().str.contains("secondary")
        contains_secondary = contains_secondary.fillna(False)
        contains_secondary.name = "Contains Secondary"
        if self.create_plots:
            viz_data.catplot(
                contains_secondary.to_frame(), x="Contains Secondary", hue="Contains Secondary",
                xlabel="", ylabel=f"Number of {data_category_str}",
                plot_type="count",
                title="Datasets that Contain Secondary Data?",
                save_dir=save_dir,
                save_fname="secondary_data(bar).png",
            )


    def describe_patients(self):
        """
        Describe the patients across all datasets.
        """
        # Create copy to avoid in-place modifications
        df_metadata = self.df_metadata.copy()
        save_dir = self.save_dir
        data_category_str = self.data_category_str

        # Drop datasets without age annotation
        df_metadata = df_metadata.dropna(subset=PEDS_VS_ADULT_COL)

        # 1. Plot the kinds of youngest, central and oldest present
        age_ranges = df_metadata["Age Range"].map(convert_age_range_to_int).dropna()
        lower = age_ranges.map(lambda x: x[0]).dropna().astype(int).tolist()
        upper = age_ranges.map(lambda x: x[1]).dropna().astype(int).tolist()
        middle = df_metadata.apply(
            lambda row: row["Avg. Age"] or row["Median Age"] or None,
            axis=1).map(extract_years_from_str).dropna().astype(int).tolist()
        df_age_bounds = pd.DataFrame({
            "Age": lower + middle + upper,
            "Bound": (["Youngest Age"] * len(lower)) + (["Average/Median Age"] * len(middle)) + (["Oldest Age"] * len(upper)),
        })
        if self.create_plots:
            viz_data.numplot(
                df_age_bounds[df_age_bounds["Bound"] == "Youngest Age"], x="Age", hue="Age",
                xlabel="Age (in Years)", ylabel="",
                plot_type="strip", jitter=0.4,
                s=10, edgecolor="black", linewidth=1.5,
                vertical_lines=[18],
                palette="vlag",
                tick_params={"axis":"y", "left": False, "labelleft": False},
                title="What is The Youngest Age in Each Dataset?",
                legend=False,
                figsize=(8, 5),
                save_dir=save_dir,
                save_fname="age_youngest(strip).png",
            )
            viz_data.numplot(
                df_age_bounds[df_age_bounds["Bound"] == "Average/Median Age"], x="Age", hue="Age",
                xlabel="Age (in Years)", ylabel="",
                plot_type="strip", jitter=0.4,
                s=10, edgecolor="black", linewidth=1.5,
                vertical_lines=[18],
                palette="vlag",
                tick_params={"axis":"y", "left": False, "labelleft": False},
                title="What is the Average/Median Age in Each Dataset?",
                legend=False,
                figsize=(8, 5),
                save_dir=save_dir,
                save_fname="age_middle(strip).png",
            )
            viz_data.numplot(
                df_age_bounds[df_age_bounds["Bound"] == "Oldest Age"], x="Age", hue="Age",
                xlabel="Age (in Years)", ylabel="",
                plot_type="strip", jitter=0.4,
                s=10, edgecolor="black", linewidth=1.5,
                vertical_lines=[18],
                palette="vlag",
                tick_params={"axis":"y", "left": False, "labelleft": False},
                title="What is the Oldest Age in Each Dataset?",
                legend=False,
                figsize=(8, 5),
                save_dir=save_dir,
                save_fname="age_oldest(strip).png",
            )

        # Compute average age of all patients
        avg_age = df_metadata["Avg. Age"].map(extract_years_from_str)
        num_patients = df_metadata["num_patients"]
        self.descriptions["Avg. Age"] = round((avg_age * num_patients).sum() / num_patients.sum(), 2)
        self.descriptions["Min. Age"] = min(lower)
        self.descriptions["Max. Age"] = max(upper)

        # 2. Plot gender/sex
        # 2.1. Standardize sex to gender
        df_metadata["Gender"] = df_metadata.apply(lambda row: row["Sex"] or row["Gender"], axis=1)
        df_metadata["Prop. Female"] = df_metadata["Gender"].map(parse_female_prop)
        # Plot proportion of male/female patients
        mask = ~df_metadata["Prop. Female"].isna()
        num_female_patients = (df_metadata[mask]["num_patients"] * df_metadata[mask]["Prop. Female"]).sum()
        total_num_patients = df_metadata[mask]["num_patients"].sum()
        self.descriptions["Prop. Female (By Dataset)"] = round(df_metadata["Prop. Female"].mean(), 4)
        self.descriptions["Prop. Female (By Patient, Only Labeled)"] = round(num_female_patients / total_num_patients, 4)
        # Now, assume all unlabeled datasets have 50% male and 50% female
        df_metadata["Prop. Female"] = df_metadata["Prop. Female"].fillna(0.5)
        num_female_patients = (df_metadata["num_patients"] * df_metadata["Prop. Female"]).sum()
        total_num_patients = df_metadata["num_patients"].sum()
        self.descriptions["Prop. Female (By Patient, Assuming 0.5 for Unlabeled)"] = round(num_female_patients / total_num_patients, 4)


    def describe_tasks(self):
        """
        Describe the imaging modalities, organs / body parts, and tasks in each dataset
        """
        # Create copy to avoid in-place modifications
        df_metadata = self.df_metadata.copy()
        save_dir = self.save_dir
        data_category_str = self.data_category_str

        # 1. Task
        unique_tasks = sorted(df_metadata[TASK_COL].str.split(", ").explode().unique())
        # Task to count
        accum_task_count = {
            "task": [],
            "count": [],
        }
        for task in unique_tasks:
            task_mask = df_metadata[TASK_COL].str.contains(task)
            accum_task_count["task"].append(task)
            accum_task_count["count"].append(task_mask.sum())
        # Create bar plot for task type counts
        df_task_count = pd.DataFrame(accum_task_count)
        self.descriptions["Tasks"] = accum_task_count
        if self.create_plots:
            viz_data.catplot(
                df_task_count, x="count", y="task",
                xlabel=f"Number of {data_category_str}", ylabel="",
                plot_type="bar",
                order=unique_tasks,
                title="Most Common Task Types",
                legend=True,
                save_dir=save_dir,
                save_fname="tasks(bar).png",
            )

        # 1.1 TODO: Anatomy Segmentation Data. What proportion is adults? Do these adults have findings?
        df_metadata_seg = df_metadata[df_metadata[TASK_COL].str.contains("Anatomy/Organ Segmentation/Detection")]
        if self.create_plots:
            viz_data.catplot(
                df_metadata_seg, y=CONTAINS_CHILDREN_COL, hue=HAS_FINDINGS_COL,
                xlabel=f"Number of {data_category_str}", ylabel="",
                plot_type="count",
                order=unique_tasks,
                title="Is Available Segmentation Data Primarily Diseased Adults?",
                legend=True,
                save_dir=save_dir,
                save_fname="seg_data_has_findings(bar).png",
            )

        # 2. Imaging modalities
        unique_modalities = []
        # Modality to count
        accum_modality_count = {
            "modality": [],
            "count": [],
        }
        for modality in unique_modalities:
            modality_mask = df_metadata[MODALITY_COL].str.contains(modality)
            accum_modality_count["modality"].append(modality)
            accum_modality_count["count"].append(modality_mask.sum())
        # Create bar plot for modality type counts
        df_modality_count = pd.DataFrame(accum_modality_count)
        self.descriptions["Modalities"] = accum_modality_count
        if self.create_plots:
            viz_data.catplot(
                df_modality_count, x="count", y="modality",
                plot_type="bar",
                xlabel=f"Number of {data_category_str}", ylabel="",
                order=unique_modalities,
                title="Most Common Imaging Modalities",
                legend=True,
                save_dir=save_dir,
                save_fname="img_modalities(bar).png",
            )

        # X. Organ / Body Part
        # df_metadata[ORGAN_COL] = df_metadata[ORGAN_COL].str.split(", ")
        # df_organs = df_metadata.explode(ORGAN_COL).reset_index(drop=True)
        # self.descriptions["Organs"] = df_organs.groupby(CONTAINS_CHILDREN_COL)[ORGAN_COL].value_counts().round(2).to_dict()
        # viz_data.catplot(
        #     df_organs, y=ORGAN_COL, hue=CONTAINS_CHILDREN_COL,
        #     xlabel=f"Number of {data_category_str}", ylabel="",
        #     plot_type="count",
        #     order=df_organs[ORGAN_COL].value_counts().index,
        #     title="What Organ / Body Part Are Most Commonly Captured?",
        #     legend=True,
        #     save_dir=save_dir,
        #     save_fname="organs(bar).png",
        # )


    def parse_sample_size_column(self):
        """
        Parse sample size column to estimate the number of patients, sequences,
        and images in each dataset.
        """
        df_metadata, descriptions = parse_sample_size_column(self.df_metadata)

        # Update modified metadata table
        self.descriptions.update(descriptions)
        self.df_metadata = df_metadata


    def parse_age_columns(self):
        """
        Parse age-related columns in the metadata table.

        1. Compute the proportion of children in each dataset.
        2. Add a column for Age Documented
        3. Add a column for Contains Children
        4. Plot Age Documented
        5. Add a column for Proportion of Patients are Adults
        6. Plot Proportion of Patients are Children
        """
        data_category_str = self.data_category_str
        save_dir = self.save_dir

        # Parse age columns
        df_metadata, descriptions = parse_age_columns(self.df_metadata)

        # Update modified metadata table
        self.df_metadata = df_metadata
        self.descriptions.update(descriptions)

        # Plot Age Documented
        ylabel_str = data_category_str if data_category_str == "Challenges" else "Datasets"
        ylabel = f"Number of {ylabel_str}"
        viz_data.catplot(
            df_metadata, x="Age Documented", hue="Contains Children",
            xlabel="", ylabel=ylabel,
            plot_type="count",
            order=["Yes", "No"],
            title=f"Do {data_category_str} Describe The Patients Age?",
            legend=True,
            save_dir=save_dir,
            save_fname="age_documented(bar).png",
        )


    def parse_demographics_columns(self):
        """
        Parse and expand the demographics-related column into individual columns.

        This function processes the "Patient Demographics / Covariates / Metadata"
        column by parsing its text content into a dictionary and expanding it into
        separate columns.
        """
        self.df_metadata = parse_demographics_columns(self.df_metadata)


################################################################################
#                              Calling Functions                               #
################################################################################
def descibe_papers():
    df_midl_papers = load_annotations("papers")
    df_midl_datasets = load_annotations("midl_datasets")

    print("==MIDL Papers==")
    # Print number of papers with public data
    public_data_col = "Private/Public Data"
    public_mask = df_midl_papers[public_data_col].str.contains("Public")
    print(f"Number of Papers with Public Data: {public_mask.sum()} / {len(public_mask)} ({public_mask.mean().round(4)})")

    # Print number of papers that mention age
    is_age_explicit_col = "Is Age Explicitly Documented"
    age_documented_mask = df_midl_papers[is_age_explicit_col]
    print(f"Number of Papers that Mention Age: {age_documented_mask.sum()} / {len(age_documented_mask)} ({age_documented_mask.mean().round(4)})")

    # Print number of paper with peds data
    contains_peds_mask = df_midl_papers[PEDS_VS_ADULT_COL].str.contains("Peds")
    print(f"Number of Papers Known To Have Peds Data: {contains_peds_mask.sum()} / {len(contains_peds_mask)} ({contains_peds_mask.mean().round(4)})")

    print("")
    print("==MIDL Referenced Datasets==")
    # Number of MIDL-referenced datasets that mention age
    age_documented_mask = ~df_midl_datasets[AGE_DOCUMENTED_HOW_COL].isna()
    print(f"Number of MIDL Datasets that Mention Age: {age_documented_mask.sum()} / {len(age_documented_mask)} ({age_documented_mask.mean().round(4)})")

    # Specifically, how many provide patient ages?
    patient_age_provided_mask = (df_midl_datasets[AGE_DOCUMENTED_HOW_COL] == "Patient-Level")
    print(f"Number of MIDL Datasets that Provide Patient-Level Age: {patient_age_provided_mask.sum()} / {len(patient_age_provided_mask)} ({patient_age_provided_mask.mean().round(4)})")

    # Number of datasets where we can infer adult/peds
    contains_peds_mask = df_midl_datasets[PEDS_VS_ADULT_COL].str.contains("Peds")
    print(f"Number of MIDL Datasets Known To Have Peds Data: {contains_peds_mask.sum()} / {len(contains_peds_mask)} ({contains_peds_mask.mean().round(4)})")


def describe_peds_in_each_category(load_kwargs=None):
    """
    Print percentages related to how under-represented children are
    """
    load_kwargs = load_kwargs or {}

    accum_data = []

    ############################################################################
    #                           Data Collections                               #
    ############################################################################
    df_collections = load_all_dataset_annotations(filter_categories=["collections_datasets"], **load_kwargs)
    curr = {"index": "Dataset Collections"}
    curr.update(compute_summary_stats(df_collections))
    accum_data.append(curr)

    # 1. Stanford AIMI
    # 2. TCIA
    # 3. MIDRC
    # 4. OpenNeuro
    collection_keys = ["openneuro_parsed", "tcia", "midrc_parsed", "stanford_aimi"]
    for filter_key in collection_keys:
        df_curr_collection = load_all_dataset_annotations(filter_categories=[filter_key], **load_kwargs)
        curr = {"index": f"Collection {filter_key}"}
        curr.update(compute_summary_stats(df_curr_collection))
        accum_data.append(curr)

    ############################################################################
    #                              Challenges                                  #
    ############################################################################
    df_challenges = load_all_dataset_annotations(filter_categories=["challenges"], **load_kwargs)
    curr = {"index": "ML Challenges"}
    curr.update(compute_summary_stats(df_challenges))
    accum_data.append(curr)

    #############################################################################
    #                         Highly-Cited Datasets                             #
    #############################################################################
    df_datasets = load_all_dataset_annotations(filter_categories=["datasets"], **load_kwargs)
    curr = {"index": "Highly-Cited Datasets"}
    curr.update(compute_summary_stats(df_datasets))
    accum_data.append(curr)

    ############################################################################
    #                              Benchmarks                                  #
    ############################################################################
    df_benchmarks = load_all_dataset_annotations(filter_categories=["benchmarks"], **load_kwargs)
    curr = {"index": "ML Benchmarks"}
    curr.update(compute_summary_stats(df_benchmarks))
    accum_data.append(curr)

    # 1. MSD
    df_msd = df_benchmarks[df_benchmarks["Benchmark"] == "Medical Segmentation Decathlon"]
    curr = {"index": "Benchmarks (MSD)"}
    curr.update(compute_summary_stats(df_msd))
    accum_data.append(curr)

    # 2. MedFAIR
    df_medfair = df_benchmarks[df_benchmarks["Benchmark"] == "MedFAIR"]
    curr = {"index": "Benchmarks (MedFAIR)"}
    curr.update(compute_summary_stats(df_medfair))
    accum_data.append(curr)

    # 3. AMOS
    df_amos = df_benchmarks[df_benchmarks["Benchmark"] == "AMOS"]
    curr = {"index": "Benchmarks (AMOS)"}
    curr.update(compute_summary_stats(df_amos))
    accum_data.append(curr)

    ############################################################################
    #                             All Datasets                                 #
    ############################################################################
    # 0. All datasets
    df_annotations = load_all_dataset_annotations(**load_kwargs)
    curr = {"index": "Total"}
    curr.update(compute_summary_stats(df_annotations))
    accum_data.append(curr)
    df_filtered = df_annotations[df_annotations[CONTAINS_CHILDREN_COL] != "Unknown"]
    print(df_filtered[CONTAINS_CHILDREN_COL].value_counts())
    print(df_filtered[CONTAINS_CHILDREN_COL].value_counts(normalize=True))

    # 1. Excluding cancer datasets
    df_wo_cancer = load_all_dataset_annotations(exclude_cancer=True)
    curr = {"index": "Total w/o Cancer"}
    curr.update(compute_summary_stats(df_wo_cancer))
    accum_data.append(curr)

    # Combine all stats
    df_stats = pd.DataFrame(accum_data)
    save_path = os.path.join(constants.DIR_FIGURES_MI, "summary_stats.csv")
    df_stats.to_csv(save_path, index=False)
    print(df_stats)


def save_peds_dataset_in_each_modality(load_kwargs=None):
    """
    Create list of pediatric datasets by imaging modality

    Parameters
    ----------
    load_kwargs : **kwargs
        Keyword arguments, by default None
    """
    load_kwargs = load_kwargs or {}
    df_all = load_all_dataset_annotations(**load_kwargs)

    # Filter for pediatric data
    df_filtered = df_all[df_all["Prop. Adult"].notnull() & (df_all["Prop. Adult"] < 1)]

    # Store modality to all datasets
    modality_to_all_dsets = {}
    modalities = ["CT", "MRI", "X-ray", "US", "Fundus"]
    for modality in modalities:
        # At the dataset level, check how many datasets are peds only
        mask = df_filtered[MODALITY_COL].map(lambda x: modality in str(x).split(", "))
        df_curr = df_filtered[mask].copy()

        # Get proportion of data that this modality represents, if multi-modal dataset
        # NOTE: If proportion for each modality is not annotated, assume it's
        #       equally split across modalities
        modality_prop = df_curr.apply(
            lambda row: row["Modalities"][modality] if row["Modalities"] and modality in row["Modalities"]
                        else 1/len(row[MODALITY_COL].split(", ")),
            axis=1
        )
        # Get the correct estimate on the number of sequences for the modality
        col = "num_sequences" if modality in ["CT", "MRI", "US"] else "num_images"
        prop_peds = 1 - df_curr["Prop. Adult"]

        # Estimate the number of data points
        num_points = df_curr[col] * modality_prop * prop_peds
        df_curr["Peds Sample Size"] = num_points

        # Compute percentage of total data contributed by each dataset
        perc_contributed = (100 * num_points / num_points.sum()).astype(float)
        df_curr["Percentage Overall"] = perc_contributed.round(2)

        # Sort by percentage contributed
        df_curr = df_curr.sort_values(by="Percentage Overall", ascending=False).reset_index(drop=True)

        # Store all datasets for this modality
        store_cols = [
            "Category",
            "Image Collection", "Benchmark", 
            "Dataset Name",
            "Link", "Paper Link",
            "Considerations", "Inferred",
            "Is Age Explicitly Documented",
            AGE_DOCUMENTED_HOW_COL,
            "Is Age Complete",
            SOURCE_COL,
            DEMOGRAPHICS_COL,
            SAMPLE_SIZE_COL,
            "Peds Sample Size",
            "Percentage Overall",
            MODALITY_COL,
            PEDS_VS_ADULT_COL,
            ORGAN_COL,
            TASK_PATH_COL,
            TASK_COL,
        ]
        modality_to_all_dsets[modality] = df_curr[store_cols]

    # Store datasets per modality
    for modality, df_dsets in modality_to_all_dsets.items():
        save_dir = os.path.join(constants.DIR_FIGURES_MI, "peds_datasets")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{modality.lower()}.csv")
        df_dsets.to_csv(save_path, index=False)


def describe_data_repackaging():
    df_primary = load_all_dataset_annotations(filter_categories=["repackaging_1st"])
    df_secondary = load_all_dataset_annotations(filter_categories=["repackaging_2nd"])

    ############################################################################
    #             Does data repackaging worsen age reporting?                  #
    ############################################################################
    print("Number of Repackaged Datasets w/ Age Annotations:")
    print(df_secondary[AGE_DOCUMENTED_HOW_COL].value_counts(dropna=False))

    # Check how age is documented among datasets that reuse data with patient-level information
    dsets_with_age = set(df_secondary[df_secondary[AGE_DOCUMENTED_HOW_COL] == "Patient-Level"]["Dataset Name"].tolist())
    mask_has_dset_with_age = df_primary["Secondary Datasets"].map(lambda x: bool(set(x).intersection(dsets_with_age)))
    print("Number of Datasets that Repackage Datasets w/ Age:", mask_has_dset_with_age.sum())
    print("How Age is Referenced in Subsequent Datasets?")
    print(df_primary.loc[mask_has_dset_with_age, ["Is Age Explicitly Documented", AGE_DOCUMENTED_HOW_COL]].value_counts(dropna=False))

    # Average adult proportion per dataset
    print("Avg. Prop Peds (Repackaged Datasets):", 1 - df_secondary["Prop. Adult"].mean())

    # # of datasets with referenced datasets (with annotations)
    all_dsets = set(df_secondary["Dataset Name"].tolist())
    mask_has_dsets = df_primary["Secondary Datasets"].map(lambda x: bool(set(x).intersection(all_dsets)))
    print("Number of Datasets w/ Manual Annotations in Repackaged Dataset:", mask_has_dsets.sum())

    # Average adult proportion per dataset, weighted by the number of references
    dset_referenced = df_primary.loc[mask_has_dsets, "Secondary Datasets"].explode()
    dset_referenced.name = "Dataset Name"
    df_references = pd.merge(df_secondary, dset_referenced, on="Dataset Name")
    print("Avg. Prop Peds (Referenced Datasets, Weighted by Use):", 1 - df_references["Prop. Adult"].mean())

    ############################################################################
    #             Effort Needed to Resolve Pediatric Data Gap                  #
    ############################################################################
    func = lambda n, p: n*(0.22-p)/(1-0.22)
    print("Number of Datasets Needed to Resolve Pediatric Data Gap")
    print("(N=100, p=1.18%):", func(100, 0.018))
    print("(N=110, p=1.18%):", func(110, 0.018))

    ############################################################################
    #                                MedSAM                                    #
    ############################################################################
    df_medsam = load_all_dataset_annotations(filter_categories=["medsam"])

    # Count number of datasets missing age
    print(df_medsam.groupby("Validation")[CONTAINS_CHILDREN_COL].value_counts())

    # Estimate the number of pediatric patients
    num_peds = int((df_medsam["num_patients"] * (1 - df_medsam["Prop. Adult"])).sum())
    num_patients = df_medsam["num_patients"].sum()
    print(f"Num. Pediatric Patients in MedSAM Datasets: {num_peds} / {num_patients}")

    # Filter for training datasets
    df_internal = df_medsam[df_medsam["Validation"] == "Internal"]
    # Estimate the number of pediatric patients
    num_peds = int((df_internal["num_patients"] * (1 - df_internal["Prop. Adult"])).sum())
    # Get total number of patients the adult/pediatric and adult-only datasets
    print("Num. Pediatric Patients in Internal Datasets: ", num_peds)
    print("Total Number of Patients in Internal Datasets: ")
    print(df_internal.groupby(CONTAINS_CHILDREN_COL)["num_patients"].sum())

    # Proportion of Pediatric Patients Before and After Including Adult-Only Datasets
    num_patients_peds_and_adult = df_internal[df_internal[CONTAINS_CHILDREN_COL] == "Peds & Adult"]["num_patients"].sum()
    num_patients_adult_only = df_internal[df_internal[CONTAINS_CHILDREN_COL] == "Adult Only"]["num_patients"].sum()
    num_patients_total = num_patients_peds_and_adult + num_patients_adult_only
    print(f"Before Adding Adult-Only Datasets: {round(100*num_peds / num_patients_peds_and_adult, 2)}%")
    print(f"After Adding Adult-Only Datasets: {round(100*num_peds / num_patients_total, 2)}%")


# NOTE: This function is useful for a glance but not used in the study
def visualize_annotated_data():
    cat_to_descriptions = {}
    categories = ["collections", "challenges", "benchmarks", "datasets", "papers"]
    for category in categories:
        df_curr = load_annotations(category)
        visualizer = OpenDataVisualizer(df_curr, category, create_plots=False, save=False)
        try:
            cat_to_descriptions[category] = visualizer.describe()
        except:
            cat_to_descriptions[category] = visualizer.descriptions

    print(json.dumps(cat_to_descriptions, indent=4))


################################################################################
#                          Aggregating Plot Functions                          #
################################################################################
def plot_age_documented():
    """
    Create horizontal stacked bar plot of how age is documented across all data
    categories.
    """
    age_documented_col = "Age Documented How"

    def compute_age_documented_percentage(df):
        # Fill missing with "Not Documented"
        df[age_documented_col] = df[age_documented_col].fillna("Not Documented")
        # Summarize the percentages of age documented
        age_documented_percs = (100 * df[age_documented_col].value_counts(normalize=True)).round(1).reset_index()
        age_documented_percs.columns = [age_documented_col, "Percentage"]
        return age_documented_percs

    # What proportion of datasets providing summary statistics is enough to know
    # if there are children
    def summary_stats_contain_age_summary(df):
        mask = df[age_documented_col] == "Summary Statistics"
        if not mask.sum():
            print("Data provided does not have rows with summary statistics")
            return None
        if DEMOGRAPHICS_COL not in df.columns.tolist():
            print("Demographics column not found in the dataframe")
            return None
        df_summary = df[mask]
        stats = {
            "contain_age_range": prop_to_perc(df_summary[DEMOGRAPHICS_COL].str.contains("Age Range:").mean()),
            "contain_avg_age": prop_to_perc(df_summary[DEMOGRAPHICS_COL].str.contains("Avg. Age:").mean()),
            "contain_median_age": prop_to_perc(df_summary[DEMOGRAPHICS_COL].str.contains("Median Age:").mean()),
        }
        return stats

    # Summarize the percentages of age documented for each category
    accum_percentages = []
    accum_age_range_given = []
    for data_category in ["papers", "challenges", "benchmarks", "datasets", "collections_datasets"]:
        df_curr = load_annotations(data_category)
        age_documented_percs = compute_age_documented_percentage(df_curr)
        age_documented_percs["Category"] = CATEGORY_TO_STRING[data_category]
        accum_percentages.append(age_documented_percs)
        accum_age_range_given.append(summary_stats_contain_age_summary(df_curr))

    # Concatenate percentages
    df_percentages_all = pd.concat(accum_percentages, ignore_index=True, axis=0)

    # Sort by the following Age Documented
    how_order = ["Not Documented", "Task/Data Description", "Summary Statistics", "Binned Patient-Level", "Patient-Level"]
    how_colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]

    how_order_and_color = list(reversed(tuple(zip(how_order, how_colors))))
    df_percentages_all[age_documented_col] = pd.Categorical(df_percentages_all[age_documented_col], categories=how_order, ordered=True)
    df_percentages_all = df_percentages_all.sort_values(by=age_documented_col).reset_index(drop=True)
    df_cum_percentages = df_percentages_all.copy()
    # Add them together iteratively to create the heights needed in the plot
    for category in df_percentages_all["Category"].unique():
        mask = df_cum_percentages["Category"] == category
        df_cum_percentages.loc[mask, "Percentage"] = df_percentages_all.loc[mask, "Percentage"].cumsum()

    # Create a bar plot
    viz_data.set_theme(figsize=(16, 8), tick_scale=3)
    fig, ax = plt.subplots()
    new_colors = []
    for documented_how, how_color in how_order_and_color:
        df_curr_age_documented = df_cum_percentages[df_cum_percentages[age_documented_col] == documented_how]
        viz_data.catplot(
            df_curr_age_documented, x="Percentage", y="Category",
            plot_type="bar", saturation=0.6,
            color=how_color,
            order=CATEGORY_ORDER,
            hue_order=how_order,
            xlabel="Percentage (%)", ylabel="",
            x_lim=(0, 100),
            tick_params={"axis": "y", "left": False, "labelleft": False},
            title="How is Age Documented?",
            legend=False,
            ax=ax,
        )
        curr_plotted_colors = set(patch.get_facecolor() for patch in ax.patches)
        curr_plotted_colors = curr_plotted_colors.difference(set(new_colors))
        new_colors.append(list(curr_plotted_colors)[0])

    # Create custom legend at the bottom
    legend_handles = [
        mpatches.Patch(color=new_colors[idx], label=documented_how)
        for idx, documented_how in enumerate(reversed(how_order))
    ]
    fig.legend(
        handles=legend_handles, reverse=True,
        handlelength=1.5,
        columnspacing=1,
        ncol=len(how_order), loc='upper center', bbox_to_anchor=(0.5, 1.15),
        # ncol=1, loc='center left', bbox_to_anchor=(1, 0.5),
        # title="How",
    )
    plt.tight_layout()

    # Save figure
    save_dir = os.path.join(constants.DIR_FIGURES_EDA, "open_mi")
    save_fname = "age_documented_how (bar).svg"
    fig.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight")
    plt.close()


def plot_table_dataset_breakdown():
    """
    Plot table heatmap breaking adult vs. peds data by modality and task
    """
    df_peds = get_dataset_counts_broken_by_modality_task(True).T.reset_index()
    df_peds["Type"] = "Peds"
    df_adult = get_dataset_counts_broken_by_modality_task(False).T.reset_index()
    df_adult["Type"] = "Adult"
    df_all = pd.concat([df_peds, df_adult], axis=0)
    # Set ordering
    col_order = ["All", "Condition Classification", "Anatomy Segmentation", "Lesion Segmentation", "Image Enhancement"]
    df_all["Task"] = pd.Categorical(df_all["Task"], categories=col_order, ordered=True)
    df_all = df_all.set_index(["Task", "Type"])
    df_all = df_all.sort_index(level=[0, 1])
    df_all = df_all.T

    # Save to CSV
    save_path = os.path.join(constants.DIR_FIGURES_MI, "breakdown_by_modality_and_task.csv")
    df_all.to_csv(save_path)

    # Configure plotting
    viz_data.set_theme(figsize=(20, 10), tick_scale=3)
    fig, ax = plt.subplots()
    ax.axis('off')

    # Count number of digits of each cell
    digit_counts = df_all.map(lambda x: 0 if x == 0 else len(str(int(x))))
    # Create colormap
    # colors = [
    #     '#000000',  # Black for 0
    #     '#D05050',  # Red for 1 digit
    #     '#CC7979',  # Light red for 2 digits
    #     '#BE8E8E',  # Lighter red for 3 digits
    #     '#DDCE91',  # Light yelow for 4 digits
    #     '#BBDD91',  # Light green for 5 digits
    #     '#9DDB86'   # Dark green for 6 digits
    # ]
    # Get 7 sequential colors from the Rocket palette
    cmap = sns.color_palette("rocket", as_cmap=True)
    colors = [cmap(pos) for pos in np.linspace(0, 1, 7)]
    cmap = ListedColormap(colors)

    # Create table
    col_width = 0.1
    row_height = 0.1
    table = ax.table(
        cellText=df_all.values.astype(str),
        rowLabels=df_all.index,
        colLabels=df_all.columns,
        cellColours=digit_counts.map(cmap).values,
        loc='center',
        rowLoc="center",
        colLoc="center",
        cellLoc='center',
        colWidths=[col_width] * (df_all.shape[1] + 1),
    )

    # Adjust font size
    table.auto_set_font_size(False)
    for cell in table.get_celld().values():
        cell.set_fontsize(24)
        cell.set_height(row_height)  # Adjust row height if needed

    # Create legend
    legend_labels = ['0', '[10^0, 10^1)', '[10^1, 10^2)', '[10^2, 10^3)', '[10^3, 10^4)', '[10^4, 10^5)', '[10^5, 10^6)']
    patches = [mpatches.Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
    ax.legend(
        patches, legend_labels,
        title='Legend',
        loc='upper right',
        bbox_to_anchor=(1.3, 0.9),  # Adjust position outside the table
        ncol=1,
        frameon=True
    )

    plt.title("Data Breakdown by Task and Modality")
    save_path = os.path.join(constants.DIR_FIGURES_EDA, "open_mi", "peds_breakdown.svg")
    plt.savefig(save_path, bbox_inches="tight")


################################################################################
#                            DEPRECATED: Figure 2c                             #
################################################################################
def plot_countries():
    """
    Plot data (patient) contribution by country
    """
    def extract_patients_by_country(row):
        if pd.isnull(row[SOURCE_COL]) or pd.isnull(row["num_patients"]):
            return {}
        countries = [
            institution_country.split("(")[-1].split(")")[0]
            for institution_country in row[SOURCE_COL].split("\n")
        ]
        num_patients = row["num_patients"]
        curr_ret = defaultdict(int)
        for country in countries:
            curr_ret[country] += num_patients // len(countries)
        return curr_ret

    # Get patient-country distribution for all patients
    df_all = load_all_dataset_annotations()
    country_to_num_patients_lst = df_all.apply(extract_patients_by_country, axis=1)

    # Get patient-country distribution for pediatric patients
    df_peds = df_all.dropna(subset=[PEDS_VS_ADULT_COL]).copy()
    df_peds["num_patients"] = df_peds["num_patients"] * (1 - df_peds["Prop. Adult"])
    df_peds["num_patients"] = df_peds["num_patients"].map(lambda x: x if x else None)
    country_to_num_children_lst = df_peds.apply(extract_patients_by_country, axis=1)

    # Sum up number of patients per country
    total_num_patients = 0
    country_to_num_patients = defaultdict(int)
    for curr_dict in country_to_num_patients_lst:
        for country, count in curr_dict.items():
            country_to_num_patients[country] += count
            total_num_patients += count

    # Sum up number of children per country
    total_num_children = 0
    country_to_num_children = defaultdict(int)
    for curr_dict in country_to_num_children_lst:
        for country, count in curr_dict.items():
            country_to_num_children[country] += count
            total_num_children += count

    # Get country to percentage of patients contributed
    country_to_perc = {
        country: round(100 * count / total_num_patients, 2)
        for country, count in country_to_num_patients.items()
    }

    # Get country to percentage of patients contributed
    country_to_perc_peds = {
        country: round(100 * count / total_num_children, 2)
        for country, count in country_to_num_children.items()
    }

    # Get top-5 contributing countries
    df_perc = pd.DataFrame(country_to_perc.items(), columns=["Country", "Percentage"])
    df_perc = df_perc.sort_values(by="Percentage", ascending=False)
    df_perc = df_perc.iloc[:5]
    order = df_perc["Country"].tolist()

    # Get top-5 contributing countries (for peds)
    df_perc_peds = pd.DataFrame(country_to_perc_peds.items(), columns=["Country", "Percentage"])
    df_perc_peds = df_perc_peds.sort_values(by="Percentage", ascending=False)
    df_perc_peds = df_perc_peds.iloc[:5]
    order = df_perc_peds["Country"].tolist()

    # Create bar plot
    viz_data.set_theme(figsize=(12, 8), tick_scale=3)
    viz_data.catplot(
        df_perc, x="Percentage", y="Country",
        plot_type="bar", color="#512200", saturation=0.75,
        tick_params={"axis":"y", "left": False},
        xlabel="Percentage of Patients (%)", ylabel="",
        order=order,
        title="Top-5 Data Contributing Countries",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="countries(bar).svg",
    )


def plot_task_types(task_types=None):
    """
    Plot task types of datasets
    """
    task_types = task_types or [
        "Anatomy/Organ Segmentation/Detection",
        "Lesion/Tumor Segmentation/Detection",
        "Disease (Risk) Segmentation/Detection",
        "Condition/Disease Classification",
        "Image Reconstruction/Generation (Enhancement/Registration)",
    ]

    # Count the number of times each specified task type has appeared in a dataset
    df_all = load_all_dataset_annotations()
    task_type_to_count = {task_type: 0 for task_type in task_types}
    num_datasets = 0

    # For each row, check if each of the task_type is included
    task_rows = df_all[TASK_COL].tolist()
    for task_str in task_rows:
        if not isinstance(task_str, str) or not task_str or pd.isnull(task_str):
            continue

        num_datasets += 1
        for task_type in task_types:
            if task_type in task_str:
                task_type_to_count[task_type] += 1

    # Divide by the number of datasets to get the percentage
    task_type_to_perc = {task_type: 100 * count / num_datasets for task_type, count in task_type_to_count.items()}
    # Convert to dataframe
    df_perc = pd.DataFrame(task_type_to_perc.items(), columns=["Task Type", "Percentage"])
    df_perc = df_perc.sort_values(by="Percentage", ascending=False)
    order = df_perc["Task Type"].tolist()

    # Create bar plot
    viz_data.set_theme(figsize=(12, 8), tick_scale=3)
    viz_data.catplot(
        df_perc, x="Percentage", y="Task Type",
        plot_type="bar", color="#512200", saturation=0.75,
        xlabel="Percentage of Datasets (%)", ylabel="",
        tick_params={"axis":"y", "left": False},
        order=order,
        title="Most Common Tasks",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="task_categories(bar).svg",
    )


def plot_modalities(modalities=None):
    """
    For each modality, plot what percentage of datasets have it
    """
    modalities = modalities or ["CT", "MRI", "X-ray", "US", "Fundus"]

    # Count the number of times each specified modality has appeared in a dataset
    df_all = load_all_dataset_annotations()
    modality_to_count = {modality: 0 for modality in modalities}

    # Get proportion of data that this modality represents, if multi-modal dataset
    # NOTE: If proportion for each modality is not annotated, assume it's
    #       equally split across modalities
    for modality in modalities:
        df_curr = df_all[df_all[MODALITY_COL].str.contains(modality, na=False)]
        modality_prop = df_curr.apply(
            lambda row: row["Modalities"][modality] if row["Modalities"] and modality in row["Modalities"]
                        else 1/len(row[MODALITY_COL].split(", ")),
            axis=1
        )
        # Get the correct estimate on the number of sequences/images for the modality
        col = "num_sequences" if modality in ["CT", "MRI", "US"] else "num_images"
        modality_to_count[modality] = (df_curr[col] * modality_prop).sum()

    # Divide by the number of datasets to get the percentage
    sum_points = sum(modality_to_count.values())
    modality_to_perc = {modality: 100 * count / sum_points for modality, count in modality_to_count.items()}
    # Convert to dataframe
    df_perc = pd.DataFrame(modality_to_perc.items(), columns=["Modality", "Percentage"])
    df_perc = df_perc.sort_values(by="Percentage", ascending=False)
    order = df_perc["Modality"].tolist()

    # Create bar plot
    viz_data.set_theme(figsize=(12, 8), tick_scale=3)
    viz_data.catplot(
        df_perc, x="Percentage", y="Modality",
        plot_type="bar", color="#512200", saturation=0.75,
        xlabel="Percentage (%)", ylabel="",
        tick_params={"axis":"y", "left": False},
        order=order,
        title="Most Common Imaging Modalities",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="modalities(bar).svg",
    )


################################################################################
#                              OpenNeuro Specific                              #
################################################################################
def download_openneuro_datasets(resume=False):
    """
    Download metadata for all OpenNeuro datasets

    Parameters
    ----------
    resume : bool
        If True, resume from latest dataset ID instead of from the start
    """
    # Create save directory
    save_dir = constants.DIR_OPENNEURO_METADATA
    os.makedirs(save_dir, exist_ok=True)

    # Load OpenNeuro dataset IDs
    df_openneuro = load_annotations("collections_openneuro")
    dataset_ids = df_openneuro["accession_number"].tolist()

    # Get latest dataset id
    exist_files = os.listdir(save_dir)
    if resume and exist_files:
        exist_dset_ids = sorted([f.split("-")[0] for f in exist_files])
        # Filter dataset IDs for everything after the last
        last_idx = dataset_ids.index(exist_dset_ids[-1])
        dataset_ids = dataset_ids[last_idx+1:]
        print(f"Removed {last_idx} from the list to download!")

    # Download metadata for each dataset
    for dset_id in tqdm(dataset_ids):
        curr_save_path = os.path.join(save_dir, f"{dset_id}-metadata.csv")
        # Skip, if already exists
        if os.path.exists(curr_save_path):
            continue

        # Attempt to load metadata from GitHub directly
        df_curr = None
        for main_branch in ["master", "main"]:
            git_path = constants.OPENNEURO_METADATA_FMT.format(
                dataset_id=dset_id,
                branch="master",
            )
            try:
                # SPECIAL CASE: Dataset `ds002717` has too many columns
                if dset_id == "ds002717":
                    cols = ["codes", "bids_code", "age", "gender", "group"]
                    df_curr = pd.read_csv(git_path, sep=",", header=None, skiprows=1, names=cols)
                # CATCH-ALL CASE: Load directly
                else:
                    df_curr = pd.read_csv(git_path, sep="\t")
                break
            except:
                continue

        # Skip, if not found
        if df_curr is None:
            print(f"\t[{dset_id}] Failed to download metadata for dataset! Skipping...")
            continue

        # Strip whitespace from column names
        df_curr.columns = [col.strip() for col in df_curr.columns]

        # Process participant ID
        subject_col = "participant_id"
        if subject_col in df_curr.columns:
            # Drop phantoms
            if df_curr[subject_col].dtype == "object":
                df_curr = df_curr.dropna(subset=[subject_col])
                df_curr[subject_col] = df_curr[subject_col].astype(str)
                df_curr = df_curr[~df_curr[subject_col].str.contains("phantom|emptyroom")]

            # Replace whitespace with null
            df_curr[subject_col] = df_curr[subject_col].astype(str).str.strip().replace("", None)

            # Drop rows with no participant ID
            df_curr = df_curr.dropna(subset=[subject_col])

        # Get age column
        age_col = "age"
        if age_col not in df_curr.columns:
            age_cols = [
                col for col in df_curr.columns
                if "age" in col.lower()
                    and "weight" not in col.lower()
                    and "height" not in col.lower()
            ]
            # CASE 1: No age column. Add null column
            if len(age_cols) == 0:
                print(f"\t[{dset_id}] Failed to find an age column! Skipping...")
                continue
            # CASE 2: 1+ age columns. Rename the first
            if len(age_cols) > 1:
                # Heuristic: Choose one that says MRI/scan/T1/T2
                heuristics = ["mri", "scan", "t1", "t2", "ses"]
                filter_cols = [col for col in age_cols if any(h in col for h in heuristics)]
                if filter_cols:
                    print(f"\t[{dset_id}] Filtering based on heuristics: {age_cols} -> {filter_cols}")
                    age_cols = filter_cols
                # Print taking first
                print(f"\t[{dset_id}] Found 2+ age columns! Choosing the first among: {age_cols}")
            chosen_col = age_cols[0]
            df_curr = df_curr.rename(columns={chosen_col: "age"})

            # If it contains "months", divide by 12
            if "months" in chosen_col.lower():
                df_curr["age"] = df_curr["age"] / 12
            elif "days" in chosen_col.lower():
                df_curr["age"] = df_curr["age"] / 365

            # SPECIAL CASE: Skip dataset, if only contains "young/older"
            unique_vals = df_curr["age"].astype(str).str.lower().unique().tolist()
            exclude_terms = ["ya", "young", "oa", "older"]
            if any(t in unique_vals for t in exclude_terms):
                print(f"\t[{dset_id}] Contains coarse young/older in age column! Skipping...")
                continue

        # Remove whitespace
        if df_curr["age"].dtype == "object":
            df_curr["age"] = df_curr["age"].astype(str).str.strip()

        # Ensure no dataset has gender/sex in age column
        sex_terms = ["M", "m", "male", "F", "f", "female"]
        if any(term in set(df_curr["age"].tolist()) for term in sex_terms):
            exist_cols = [col for col in ["sex", "gender"] if col in df_curr.columns.tolist()]
            if len(exist_cols) == 0:
                raise RuntimeError(f"[{dset_id}] Missing sex or gender column, despite misplacing it in age!")
            if len(exist_cols) > 1:
                raise RuntimeError(f"[{dset_id}] Both sex/gender columns exist, misplacing one in age! Create special case...")

            # Get single column
            sex_col = exist_cols[0]
            print(f"\t[{dset_id}] Gender/sex column ({sex_col}) has been swapped with age column! Swapping...")
            df_curr["age"], df_curr[sex_col] = df_curr[sex_col], df_curr["age"]

        # Skip, if entire column is empty
        if df_curr[age_col].isna().all():
            print(f"\t[{dset_id}] Entire age column is empty! Skipping...")
            continue

        # Attempt to parse values in age column
        try:
            df_curr["age"] = df_curr["age"].map(openneuro_parse_age)
        except Exception as error_msg:
            raise RuntimeError(f"[{dset_id}] Failed to parse age! Trace: \n{error_msg}")

        # If negative ages exist, then dataset is invalid
        # NOTE: This handles special case `ds001365`
        if df_curr["age"].min() < 0 or df_curr["age"].max() >= 120:
            print(f"\t[{dset_id}] Invalid ages found! Dataset ages must be invalid. Skipping...")
            continue

        # Save only a few columns
        save_cols = ["participant_id", "subject_id", "id", "age", "gender", "sex", "race", "ethnicity"]
        save_cols = [col for col in save_cols if col in df_curr.columns.tolist()]
        df_curr = df_curr[save_cols]

        # Save to directory
        df_curr.to_csv(curr_save_path, index=False)


def summarize_openneuro_datasets():
    """
    Parse age-related information from all structural MRI datasets on OpenNeuro.

    Note
    ----
    This function creates the data/metadata/openneuro_metadata_parsed.csv
    """
    assert os.path.exists(constants.DIR_OPENNEURO_METADATA), "Missing metadata directory!"

    # Instantiate class that extracts info from dataset's associated GitHub page
    github_extractor = OpenNeuroExtractor()

    # Load OpenNeuro dataset IDs
    df_neuro = load_annotations("collections_openneuro")
    indices = df_neuro.index.tolist()

    # Load metadata for each dataset
    for idx in indices:
        dset_id = df_neuro.loc[idx, "accession_number"]
        metadata_path = os.path.join(constants.DIR_OPENNEURO_METADATA, f"{dset_id}-metadata.csv")

        # Column: Modality
        df_neuro.loc[idx, MODALITY_COL] = "MRI"

        # Column: Number of samples
        # NOTE: This is adjusted later, if patient-level metadata exists
        col = SAMPLE_SIZE_COL
        num_subjects = df_neuro.loc[idx, "num_patients"]
        df_neuro.loc[idx, col] = f"Patients: {num_subjects}\nSequences: {num_subjects}"

        # Column: Age Documented How
        # Early return, if metadata doesn't exist
        if not os.path.exists(metadata_path):
            df_neuro.loc[idx, "Is Age Explicitly Documented"] = False
            df_neuro.loc[idx, AGE_DOCUMENTED_HOW_COL] = None

            # Column: Peds vs. Adult
            age_ranges = df_neuro.loc[idx, "ages"]
            if pd.isnull(age_ranges) or not isinstance(age_ranges, str):
                continue

            # CASE: Lower/upper bounds are provided but not in the metadata
            age_ranges = age_ranges.replace("+", "")
            age_lower = ast.literal_eval(age_ranges.split(",")[0].split("-")[0].strip())
            age_upper = ast.literal_eval(age_ranges.split(",")[-1].split("-")[-1].strip())

            # Filter invalids
            if age_lower < 0 or max(age_lower, age_upper) >= 120:
                print(f"[{dset_id}] Invalid age ranges!")
                continue

            df_neuro.loc[idx, "Is Age Explicitly Documented"] = True
            df_neuro.loc[idx, AGE_DOCUMENTED_HOW_COL] = "Task/Data Description"

            # Column: Peds vs. Adult
            if age_lower >= 18:
                val = "Adult"
            elif age_upper < 18:
                val = "Peds"
            else:
                val = f"Peds, Adult"
            df_neuro.loc[idx, PEDS_VS_ADULT_COL] = val

            # Column: Demographics
            df_neuro.loc[idx, DEMOGRAPHICS_COL] = f"Age Range: {age_lower:.2f} to {age_upper:.2f} years"
            continue

        # Load metadata file
        df_curr = pd.read_csv(metadata_path)

        # NOTE: We only save metadata files for datasets with age annotations,
        #       so this dataset explicitly documents age.
        col = "Is Age Explicitly Documented"
        df_neuro.loc[idx, col] = True

        # Drop empty rows
        if "participant_id" in df_curr.columns:
            df_curr["participant_id"] = df_curr["participant_id"].astype(str).str.strip()
            df_curr = df_curr.dropna(subset=["participant_id"])

        # Binned patient level
        col = AGE_DOCUMENTED_HOW_COL
        is_binned = df_curr["age"].astype(str).str.contains("-").any()
        if is_binned:
            df_neuro.loc[idx, col] = "Binned Patient-Level"
        else:
            df_neuro.loc[idx, col] = "Patient-Level"

        # Column: Is Age Complete
        col = "Is Age Complete"
        df_neuro.loc[idx, col] = not df_curr["age"].isna().any()

        # Ensure sex/gender is not in dataset age
        assert "M" not in df_curr["age"].tolist(), f"Dataset `{dset_id}` contains sex/gender in age"
        assert "F" not in df_curr["age"].tolist(), f"Dataset `{dset_id}` contains sex/gender in age"

        # Parse ages
        parsed_ages = df_curr["parsed_age"] = df_curr["age"].map(openneuro_parse_age)
        parsed_ages = parsed_ages.dropna()

        github_metadata = github_extractor.process_dataset(dset_id)
        # Column: Paper Link
        paper_cols = ["doi_of_papers_from_source_data_lab", "doi_of_paper_published_using_openneuro_dataset"]
        paper_link = None
        for col in paper_cols:
            curr_paper_link = df_neuro.loc[idx, col]
            if curr_paper_link:
                paper_link = curr_paper_link
                break
        df_neuro.loc[idx, "Paper Link"] = paper_link
        # Column: Institution
        df_neuro.loc[idx, SOURCE_COL] = github_metadata.get("institutions")
        # Column: Senior Author
        df_neuro.loc[idx, "Ethics Approval"] = github_metadata.get("ethics_approval")

        # Column: Demographics
        descs = []
        descs.append(f"Avg. Age: {parsed_ages.mean():.2f} years")
        descs.append(f"Age Range: {parsed_ages.min():.2f} to {parsed_ages.max():.2f} years")
        df_neuro.loc[idx, DEMOGRAPHICS_COL] = "\n".join(descs)

        # Column: Number of samples
        col = SAMPLE_SIZE_COL
        num_subjects = len(df_curr)
        df_neuro.loc[idx, col] = f"Patients: {num_subjects}\nSequences: {num_subjects}"

        # Column: Peds vs. Adult
        col = PEDS_VS_ADULT_COL
        contains_peds = (parsed_ages.min() < 18).any()
        contains_adults = (parsed_ages.max() >= 18).any()
        val = "Adult"
        if (parsed_ages.max() < 18).all():
            val = "Peds"
        elif contains_peds and contains_adults:
            prop_adult = floor_to_decimal(100*(parsed_ages >= 18).mean(), 2) 
            val = f"Peds, Adult (>{prop_adult}%)"
        df_neuro.loc[idx, col] = val

    # Rename columns
    df_neuro = df_neuro.rename(
        columns={
            "accession_number": "Dataset Name",
            "dataset_url": "Link",
        }
    )
    df_neuro[ORGAN_COL] = "Brain"

    # Store only select number of columns
    cols = [
        "Dataset Name",
        "Link",
        "Paper Link",
        SOURCE_COL,
        "Ethics Approval",    # NOTE: This is used to get the institution if not available
        "Is Age Explicitly Documented",
        AGE_DOCUMENTED_HOW_COL,
        "Is Age Complete",
        DEMOGRAPHICS_COL,
        SAMPLE_SIZE_COL,
        MODALITY_COL,
        PEDS_VS_ADULT_COL,
        ORGAN_COL,
    ]
    df_neuro = df_neuro[cols]

    # Update metadata file
    save_path = constants.DIR_METADATA_MAP["openneuro_parsed"]
    df_neuro.to_csv(save_path, index=False)


def openneuro_parse_age(text, choice="lower"):
    """
    Parse age for OpenNeuro

    Parameters
    ----------
    text : Any
        Contains age string as either age or range (10-15)
    choice : str
        Method of reduction if range provided. One of (lower, mid upper)
    """
    # Null-like
    if pd.isnull(text):
        return None
    # Float/integer
    elif isinstance(text, (int, float)) or is_numeric_dtype(text):
        return text

    # Remove whitespace
    text = text.strip()

    # Contains "n/a" in text
    # NOTE: Dataset `ds004928` encodes missing age as "A"
    if "n/a" in text.lower() or "na" in text.lower() or text == "A":
        return None

    # Replace "-" with "," so it can be parsed
    if "-" in text:
        text = text.replace("-", ",")

    # Handle starting with 0
    if len(text) > 1 and text[0] == "0":
        text = text[1:]

    # Handle endings
    # CASE 1: Ends with months
    if text.endswith("M") or text.endswith("months"):
        text = text.replace("M", "").replace("months", "").strip()
        if not text:
            return None
        try:
            age_in_months = ast.literal_eval(text)
        except:
            print(f"Failed to parse (months): `{text}`")
            return None
        return age_in_months / 12

    # CASE 2: Ends with days
    if text.endswith("D") or text.endswith("days"):
        text = text.replace("D", "").replace("days", "").strip()
        if not text:
            return None
        try:
            age_days = ast.literal_eval(text)
        except:
            print(f"Failed to parse (days): `{text}`")
            return None
        return age_days / 365

    # CASE 3: Ends with year
    if text.endswith("Y") or text.endswith("years"):
        text = text.replace("Y", "").replace("years", "").strip()

    # CASE 4: Ends with "+"
    if text.endswith("+"):
        text = text.replace("+", "")

    # Parse age
    try:
        parsed_age = ast.literal_eval(text)
    # TODO: Remove this
    except:
        assert False, text

    # Early return, if numeric
    if isinstance(parsed_age, (int, float)):
        return parsed_age
    assert isinstance(parsed_age, (tuple, list)), f"Unexpected type: {type(parsed_age)}"

    # CASE 1: Midpoint
    if choice == "mid":
        return sum(parsed_age) / len(parsed_age)
    # CASE 2: Lower/Upper
    idx = 0 if choice == "lower" else -1
    return parsed_age[idx]


################################################################################
#                                MIDRC Specific                                #
################################################################################
def summarize_midrc():
    """
    Parse age-related information from all cases in MIDRC.

    Note
    ----
    This function creates the data/metadata/midrc_metadata_parsed.csv
    """
    # Load MIDRC data
    df_metadata = pd.read_csv(constants.DIR_METADATA_MAP["midrc"])

    # Remap modalities
    map_modalities = {
        "DX": "X-ray",
        "CR": "X-ray",
        "MR": "MRI",
        "CT": "CT",
        "Ultrasound": "US",
    }

    # Filter based on modality
    modality_col = "imaging_studies_0_study_modality_0"
    filter_modalities = []
    modality_mask = df_metadata[modality_col].isin(map_modalities.keys())
    df_metadata = df_metadata[modality_mask]

    # Remap modality
    df_metadata[MODALITY_COL] = df_metadata[modality_col].map(map_modalities.get)
    modalities = sorted(df_metadata[MODALITY_COL].unique())

    # Split by project
    project_col = "project_id"
    project_ids = sorted(df_metadata[project_col].unique().tolist())
    accum_rows = []
    for project_id in project_ids:
        df_project = df_metadata[df_metadata[project_col] == project_id] 
        
        # Filter for each modality
        for curr_modality in modalities:
            ret = summarize_midrc_project(df_project, curr_modality)
            if ret:
                accum_rows.append(ret)

    # Create metadata dataframe
    df_midrc = pd.DataFrame(accum_rows)

    # Store only select number of columns
    cols = [
        "Dataset Name",
        "Is Age Explicitly Documented",
        AGE_DOCUMENTED_HOW_COL,
        "Is Age Complete",
        SOURCE_COL,
        DEMOGRAPHICS_COL,
        SAMPLE_SIZE_COL,
        MODALITY_COL,
        PEDS_VS_ADULT_COL,
        ORGAN_COL,
    ]
    df_midrc = df_midrc[cols]

    # Update metadata file
    save_path = constants.DIR_METADATA_MAP["midrc_parsed"]
    df_midrc.to_csv(save_path, index=False)


def summarize_midrc_project(df_project, modality=None):
    """
    Extract summary information for specific project subset of MIDRC

    Parameters
    ----------
    df_project : pd.DataFrame
        Metadata filtered for a specific project
    modality : str
        Name of modality to filter for if any

    Returns
    -------
    dict
        Contains metadata about MIDRC particular to the modality. Returns
        empty dictionary if no data provided
    """
    accum_data = {}
    age_col = "age_at_index"

    # Project name
    project_id = str(df_project["project_id"].iloc[0])

    # CASE: Filtering for modality
    if modality:
        df_project = df_project[df_project[MODALITY_COL] == modality]

        # Add specific tag for modality
        project_id += f" / {modality}"

    # Early exit, if no data
    if df_project.empty:
        return accum_data

    # Column: Dataset Name
    accum_data["Dataset Name"] = f"MIDRC ({project_id})"

    # Column: Institution
    contributors = sorted(df_project["data_contributor_0"].unique().tolist())
    accum_data[SOURCE_COL] = "\n".join([f"{c} (USA)" for c in contributors])

    # Column: Modality
    accum_data[MODALITY_COL] = ", ".join(sorted(df_project[MODALITY_COL].unique()))

    # Column: Age documentation
    accum_data["Is Age Explicitly Documented"] = True
    accum_data[AGE_DOCUMENTED_HOW_COL] = "Patient-Level"

    # Column: Is Age Complete
    col = "Is Age Complete"
    accum_data[col] = not df_project[age_col].isna().any()

    # Filter for organs representing at least 1% of the data
    old_organ_col = "imaging_studies_0_body_part_examined_0"
    organ_prop = df_project[old_organ_col].str.lower().value_counts(normalize=True)
    organ_perc = (100 * organ_prop).round(2) 
    organ_perc = organ_perc[organ_perc >= 1]
    organ_perc["others"] = round(100 - organ_perc.sum(), 2)
    organ_str = "\n".join([
        f"{organ} ({perc}%)"
        for organ, perc in organ_perc.to_dict().items()
    ])
    accum_data[ORGAN_COL] = organ_str

    # Column: Number of samples
    col = SAMPLE_SIZE_COL
    num_subjects = len(df_project)
    if df_project[MODALITY_COL].nunique() == 1: 
        unit = "Images" if modality == "X-ray" else "Sequences"
    else:
        modalities = df_project[MODALITY_COL].unique().tolist()
        print(
            "[summarize_midrc_project] More than 1 modality detected! "
            f"Assuming unit = sequence for all modalities: {modalities}"
        )
        unit = "Sequences"
    accum_data[col] = f"Patients: {num_subjects}\n{unit}: {num_subjects}"

    # CASE: No age columns
    if df_project[age_col].isna().all():
        accum_data["Is Age Explicitly Documented"] = False
        accum_data[AGE_DOCUMENTED_HOW_COL] = None
        accum_data[DEMOGRAPHICS_COL] = None
        accum_data[PEDS_VS_ADULT_COL] = None
        return accum_data

    # Column: Demographics
    descs = []
    ages = df_project[age_col].dropna()
    descs.append(f"Avg. Age: {ages.mean():.2f} years")
    descs.append(f"Age Range: {ages.min():.2f} to {ages.max():.2f} years")
    accum_data[DEMOGRAPHICS_COL] = "\n".join(descs)

    # Column: Peds vs. Adult
    col = PEDS_VS_ADULT_COL
    contains_peds = ages.min() < 18
    contains_adults = ages.max() >= 18
    val = "Adult"
    if (ages < 18).all():
        val = "Peds"
    elif contains_peds and contains_adults:
        prop_adult = floor_to_decimal(100*(ages >= 18).mean(), 2) 
        val = f"Peds, Adult (>{prop_adult}%)"
    accum_data[col] = val

    return accum_data


################################################################################
#                               Helper Functions                               #
################################################################################
def get_dataset_counts_broken_by_modality_task(filter_peds_vs_adult=True):
    """
    Break down datasets by imaging modality and task

    Parameters
    ----------
    filter_peds_vs_adult : bool, optional
        Whether to filter for datasets that contain children or not, by default
        True

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with task as the index and modality as the columns.
        Each value is the estimated number of sequences of that modality, which
        contains children.
    """
    # Accumulate all dataset metadata
    # NOTE: Now, we assume that fill in missing % adult with the avg. % of peds
    #       data in Peds, Adult datasets
    df_annotations = load_all_dataset_annotations()

    # Remove datasets without knowing the amount of adult/pediatric data
    df_annotations = df_annotations[~df_annotations["Prop. Adult"].isna()]

    # Filter for data that has peds or adults, if specified
    if filter_peds_vs_adult:
        mask = df_annotations[CONTAINS_CHILDREN_COL].str.contains("Peds")
    else:
        mask = ~df_annotations[CONTAINS_CHILDREN_COL].str.contains("Peds Only")

    df_annotations = df_annotations[mask]

    # For each modality, check the number of data points
    accum_stats = []
    modalities = ["CT", "MRI", "X-ray", "US", "Fundus"]
    tasks = ["ALL", "Condition/Disease Classification", "Anatomy/Organ Segmentation/Detection", "Lesion/Tumor Segmentation/Detection", "Image Reconstruction/Generation"]
    rename_tasks = ["All", "Condition Classification", "Anatomy Segmentation", "Lesion Segmentation", "Image Enhancement"]
    for modality in modalities:
        # At the dataset level, check how many datasets are peds only
        mask = df_annotations[MODALITY_COL].map(lambda x: modality in str(x).split(", "))
        df_curr = df_annotations[mask]
        prop_subgroup = df_curr["Prop. Adult"]
        if filter_peds_vs_adult:
            prop_subgroup = (1 - prop_subgroup)
        # Get proportion of data that this modality represents, if multi-modal dataset
        # NOTE: If proportion for each modality is not annotated, assume it's
        #       equally split across modalities
        modality_prop = df_curr.apply(
            lambda row: row["Modalities"][modality] if row["Modalities"] and modality in row["Modalities"]
                        else 1/len(row[MODALITY_COL].split(", ")),
            axis=1
        )
        # Get the correct estimate on the number of sequences for the modality
        col = "num_sequences" if modality in ["CT", "MRI", "US"] else "num_images"
        num_points = df_curr[col] * modality_prop
        # Skip, if no modality-specific data
        if modality_prop.empty or num_points.sum() == 0:
            continue
        # Filter on specific tasks
        for task_idx, task in enumerate(tasks):
            curr_stats = {}
            curr_stats["Task"] = rename_tasks[task_idx]
            # Get the datasets with this task
            if task == "ALL":
                curr_num_points = num_points
                curr_prop_subgroup = prop_subgroup
            else:
                mask_task = df_curr[TASK_COL].str.contains(task).fillna(False)
                curr_num_points = num_points[mask_task]
                curr_prop_subgroup = prop_subgroup[mask_task]
            # Now, estimate how much of these modality-specific sequences are peds
            curr_stats[f"{modality}"] = int((curr_num_points * curr_prop_subgroup).sum())
            accum_stats.append(curr_stats)

    # Combine dictionaries for each task
    task_to_dict = {}
    for curr_stats in accum_stats:
        task = curr_stats["Task"]
        if task not in task_to_dict:
            task_to_dict[task] = {}
        task_to_dict[task].update(curr_stats)

    # Sort values by ALL
    accum_stats = list(task_to_dict.values())
    df_stats = pd.DataFrame(accum_stats)
    df_stats = df_stats.set_index("Task").T
    df_stats = df_stats.sort_values("All", ascending=False)
    order = ["X-ray", "US", "CT", "MRI", "Fundus"]
    df_stats = df_stats.loc[order]

    return df_stats


def load_all_dataset_annotations(
        filter_categories=None,
        exclude_cancer=False,
    ):
    """
    Load annotations for all datasets with fine-grained annotations, which includes:
        1. Challenges
        2. Benchmarks
        3. Well-Cited Datasets
        4. Collection Datasets (specifically for Stanford AIMI and TCIA)

    Parameters
    ----------
    filter_categories : list, optional
        If specified, only load annotations for the specified categories
    exclude_cancer : bool, optional
        Whether to exclude datasets that are primarily for cancer imaging,
        by default False

    Returns
    -------
    pd.DataFrame
        A table containing annotations for all datasets.
    """
    if exclude_cancer:
        print("[Loading annotations] Excluding cancer datasets...")

    categories = ["challenges", "benchmarks", "datasets", "collections_datasets"]
    if filter_categories:
        categories = filter_categories
    accum_annotations = []
    for category in categories:
        df_curr = load_annotations(category)
        df_curr["Category"] = category
        # Parse columns
        try:
            df_curr, _ = parse_sample_size_column(df_curr, assume_sequence=True)
        except:
            print(f"Failed to parse sample size column for category: {category}")
        try:
            df_curr = parse_demographics_columns(df_curr)
        except:
            print(f"Failed to parse demographics columns for category: {category}")
        df_curr, _ = parse_age_columns(df_curr)

        # If specified, filter out cancer datasets based on keywords in columns
        if exclude_cancer:
            cols = ["Dataset Name", "Description", "Inclusion Criteria", TASK_PATH_COL]
            for col in cols:
                if col not in df_curr.columns:
                    continue
                regex = r"tcia|cancer|tumor|tumour|carcinoma|sarcoma|glioma|metastasis|metastases"
                mask_cancer = df_curr[col].str.contains(regex, case=False, na=False)
                df_curr = df_curr[~mask_cancer]

        # Parse proportion of each modality for multi-modal datasets
        if "Modalities" in df_curr.columns:
            df_curr["Modalities"] = df_curr["Modalities"].map(parse_perc_from_text)
        else:
            df_curr["Modalities"] = None

        accum_annotations.append(df_curr)

    # Combine annotations and remove duplicates
    df_accum = pd.concat(accum_annotations, ignore_index=True)
    df_accum = df_accum.drop_duplicates(subset=["Dataset Name"])
    return df_accum


def load_annotations(data_category="challenges",
                     metadata_path=constants.DIR_METADATA_MAP["open_data"]):
    """
    Load annotations from an XLSX metadata file based on the specified data category.

    Parameters
    ----------
    data_category : str, optional
        The category of data to load annotations for. Default is "challenges".
        Possible values are:
        - "papers"
        - "challenges"
        - "benchmarks"
        - "datasets"
        - "collections"
        - "collections_datasets"    # Loads the individual datasets for the following
        - "stanford_aimi"
        - "tcia"
        - "midrc_parsed"
        - "openneuro_parsed"
    metadata_path : str, optional
        The path to the XLSX metadata file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the metadata for the specified data category.

    Raises
    ------
    KeyError
        If the specified data category is not found in the mapping.
    """
    # SPECIAL CASE: If data category is `collections_datasets`, load OpenNeuro/MIDRC/AIMI/TCIA
    if data_category == "collections_datasets":
        df_metadata = pd.concat([
            pd.read_excel(metadata_path, CATEGORY_TO_SHEET[curr_category])
            for curr_category in ["openneuro_parsed", "midrc_parsed", "stanford_aimi", "tcia"]
        ], ignore_index=True, axis=0)
        df_metadata["Image Collection"] = df_metadata["Image Collection"].fillna(method="ffill")
        return df_metadata

    ############################################################################
    #                  Aggregate Data Collection Metadata                      #
    ############################################################################
    # SPECIAL CASE: If data category is `collections_openneuro`, load OpenNeuro
    if data_category == "collections_openneuro":
        df_metadata = pd.read_csv(constants.DIR_METADATA_MAP["openneuro"])
        # Filter for anatomical MRI
        df_metadata = df_metadata[df_metadata["modalities"].str.contains("MRI - anat").fillna(False)]
        # Parse minimum and maximum age
        df_metadata["age_range"] = df_metadata["ages"].map(extract_list_of_age_ranges)

        # Create column for Peds vs. Adult
        contains_children = df_metadata["age_range"].map(lambda x: x[0] < 18 if x is not None else False)
        contains_adults = df_metadata["age_range"].map(lambda x: x[1] >= 18 if x is not None else False)
        df_metadata[PEDS_VS_ADULT_COL] = None
        df_metadata.loc[contains_children, PEDS_VS_ADULT_COL] = "Peds"
        df_metadata.loc[contains_adults, PEDS_VS_ADULT_COL] = "Adult"
        df_metadata.loc[contains_children & contains_adults, PEDS_VS_ADULT_COL] = "Peds, Adult"

        # Create a column for age documented as Patient-Level
        mask = df_metadata["age_range"].notnull()
        df_metadata.loc[mask, "Age Documented How"] = "Patient-Level"

        # Create column for modalities
        df_metadata[MODALITY_COL] = "MRI"

        # Filter out animal studies, based on title
        animal_terms = [
            "animal", "animals",
            "yeast", "drosophila",
            "mouse", "mice", "rat", "rats", "rodent", "rodents"
            "dog", "dogs",
            "cat", "cats",
            "monkey", "monkeys", "primate",
            "dolphin", "dolphins",
            "sheep", "sheeps",
            "pig", "pigs",
            "rabbit", "rabbits",
            "bat", "bats",
        ]
        human_mask = df_metadata["dataset_name"].map(
            lambda x: not any(term in split_camel_case(x).replace("_", " ").lower().split() for term in animal_terms)
            if isinstance(x, str) else True
        )
        df_metadata = df_metadata[human_mask]

        # Rename column for number of participants
        df_metadata.rename(columns={"num_subjects": "num_patients"}, inplace=True)
        return df_metadata

    # SPECIAL CASE: If data category is `collections_midrc`, load collections and filter only for MIDRC
    if data_category == "collections_midrc":
        df_metadata = pd.read_excel(metadata_path, sheet_name=CATEGORY_TO_SHEET["collections"])
        df_metadata = df_metadata[df_metadata["Image Collection"] == "MIDRC"]
        return df_metadata

    ############################################################################
    #                         Other Data Category                              #
    ############################################################################
    # DEFAULT CASE: Any other category
    df_metadata = pd.read_excel(metadata_path, sheet_name=CATEGORY_TO_SHEET[data_category])

    # CASE 1: If data category is `papers`, filter for those in in inclusion criteria
    if data_category == "papers":
        # Drop missing
        df_metadata = df_metadata.dropna(subset=["Peds vs. Adult"])
        df_metadata.rename(columns={"Modality": MODALITY_COL}, inplace=True)
        # Create mask to filter for papers in inclusion criteria
        valid_modalities = ["CT", "MRI", "X-ray", "Ultrasound", "Fundus"]
        accum_bool = [df_metadata[MODALITY_COL].str.contains(modality) for modality in valid_modalities]
        valid_mask = reduce(lambda x, y: x | y, accum_bool)
        # Remove explicitly excluded modalities
        valid_mask = valid_mask & ~df_metadata[MODALITY_COL].str.contains("Exclusion Criteria")
        # Exclude Simulated-only data
        public_data_col = "Private/Public Data"
        valid_mask = valid_mask & (df_metadata[public_data_col] != "Simulated")
        print(f"Number of Valid / Total Papers: {valid_mask.sum()} / {len(valid_mask)}")

        # Filter data
        df_metadata = df_metadata[valid_mask]

    # CASE 2: If data category is `benchmarks`, forward fill benchmark name
    elif data_category == "benchmarks":
        df_metadata["Benchmark"] = df_metadata["Benchmark"].fillna(method="ffill")

    # CASE 3: If data category is `collection`, forward fill benchmark name
    elif "Image Collection" in df_metadata.columns.tolist():
        df_metadata["Image Collection"] = df_metadata["Image Collection"].fillna(method="ffill")

    # CASE 4: If data category is Repackaging (Secondary), parse out Secondary Datasets column
    elif data_category == "repackaging_1st":
        col = "Secondary Datasets"
        df_metadata[col] = df_metadata[col].str.split("\n")

    # CASE 5: If data category is Repackaging (Secondary), remove KiTS23
    elif data_category == "repackaging_2nd":
        df_metadata = df_metadata[df_metadata["Dataset Name"] != "KiTS23"]

    return df_metadata


def parse_sample_size_column(df_metadata, assume_sequence=True):
    """
    Parse the 'Sample Size' column in the metadata DataFrame to estimate the
    number of patients, sequences, and images for each dataset. This function
    attempts to extract numerical values from the 'Sample Size' column for
    patients, sequences, and images, and handles missing values by making
    assumptions based on modality types.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        The metadata DataFrame containing a 'Sample Size' column with
        information about the number of patients, sequences, and images.
    assume_sequence : bool, optional
        If True, if "Sequences: " is missing, first try to convert the # of
        Patients to # of Sequences. If # Patients is not available, try # of
        Images

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with additional columns for 'num_patients',
        'num_sequences', and 'num_images'.
    dict
        Descriptions containing the total number of sequences, images, and
        patients across all datasets, with assumptions applied for missing
        patient annotations.
    """
    df_metadata = df_metadata.copy()
    descriptions = {}

    # 1. How many patients are there?
    data_sizes = df_metadata[SAMPLE_SIZE_COL].str.split("\n").fillna("")
    data_sizes = data_sizes.map(lambda x: [item for item in x if "N/A" not in item])
    # NOTE: When estimating the number of patients, ignore datasets without annotated number of patients
    df_metadata["num_patients"] = data_sizes.map(
        lambda x: sum([int(item.split("Patients: ")[-1].replace(",",""))
                        for item in x if "Patient" in item]) or None
    )
    df_metadata["num_sequences"] = data_sizes.map(
        lambda x: sum([int(item.split("Sequences: ")[-1].replace(",",""))
                        for item in x if "Sequences" in item]) or None
    )
    df_metadata["num_images"] = data_sizes.map(
        lambda x: sum([int(item.split("Images: ")[-1].replace(",",""))
                        for item in x if "Images" in item]) or None
    )

    # Handle missing patient annotation
    missing_patient = df_metadata["num_patients"].isna()
    # CASE 1: For CT and MRI, assume 1 sequence = 1 patient
    curr_mask = missing_patient & (df_metadata[MODALITY_COL].str.contains("CT") | df_metadata[MODALITY_COL].str.contains("MRI"))
    df_metadata.loc[curr_mask, "num_patients"] = df_metadata.loc[curr_mask, "num_sequences"]
    # CASE 2: For Fundus, assume 2 images = 1 patient
    curr_mask = missing_patient & (df_metadata[MODALITY_COL].str.contains("Fundus"))
    df_metadata.loc[curr_mask, "num_patients"] = (df_metadata.loc[curr_mask, "num_images"] / 2).map(np.ceil)

    # If making assumption for number of sequence
    if assume_sequence:
        LOGGER.debug(
            "[Parse Sample Size] If 'Sequences: ' is missing, assuming # Patients "
            "-> # Sequences. If # Patients is missing, use # Images instead!"
        )
        missing_mask = df_metadata["num_sequences"].isna()
        df_metadata.loc[missing_mask, "num_sequences"] = df_metadata.loc[missing_mask].apply(
            lambda row: row["num_patients"] if not pd.isnull(row["num_patients"])
                        else row["num_images"],
            axis=1
        )

    # Store numbers of sequences/images
    # NOTE: Number of patients is underestimated since datasets without patient count
    descriptions["Number of Sequences"] = int(df_metadata["num_sequences"].sum())
    descriptions["Number of Images"] = int(df_metadata["num_images"].sum())
    descriptions["Number of Patients"] = int(df_metadata["num_patients"].sum())

    # Update modified metadata table
    return df_metadata, descriptions


def parse_age_columns(df_metadata):
    """
    Parse age-related columns to determine the distribution of adult and pediatric data.

    This function analyzes the metadata DataFrame to determine the proportion of data
    categorized by age groups (adults and children). It adds columns to indicate whether
    the age is documented and whether the dataset contains children or adults. It also
    calculates and includes the proportion of adult patients, and provides a summary
    of the number of children and adults.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        The metadata DataFrame containing age-related information.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with added age-related columns.
    dict
        Descriptions containing the number and proportion of children and adults.
    """
    descriptions = {}
    df_metadata = df_metadata.copy()

    # 2. What proportion of the data is from children?
    # 2.0. How many datasets mention the patient's age?
    mask_adult_only = df_metadata[PEDS_VS_ADULT_COL].str.startswith("Adult").fillna(False)
    mask_peds_only = (df_metadata[PEDS_VS_ADULT_COL] == "Peds").fillna(False)
    mask_peds_and_adult = (df_metadata[PEDS_VS_ADULT_COL].str.startswith("Peds, Adult")).fillna(False)

    # 2.1. Add column for Age Documented
    df_metadata["Age Documented"] = "No"
    age_documented = ~df_metadata[PEDS_VS_ADULT_COL].isna()
    df_metadata.loc[age_documented,"Age Documented"] = "Yes"

    descriptions["Number of Patients (With Age Documented)"] = int(df_metadata.loc[age_documented, "num_patients"].sum())

    # 2.2. Add column for Contains Children
    df_metadata[CONTAINS_CHILDREN_COL] = "Unknown"
    df_metadata.loc[mask_adult_only, CONTAINS_CHILDREN_COL] = "Adult Only"
    df_metadata.loc[mask_peds_only, CONTAINS_CHILDREN_COL] = "Peds Only"
    df_metadata.loc[mask_peds_and_adult, CONTAINS_CHILDREN_COL] = "Peds & Adult"

    # 2.4. Add column for Proportion of Patients are Adults
    prop_adult = "Prop. Adult"
    df_metadata[prop_adult] = None
    df_metadata.loc[mask_adult_only, prop_adult] = 1.
    df_metadata.loc[mask_peds_only, prop_adult] = 0.
    df_metadata.loc[mask_peds_and_adult, prop_adult] = df_metadata.loc[mask_peds_and_adult, PEDS_VS_ADULT_COL].map(
        parse_percentage_from_text) / 100

    # 2.5. Plot Proportion of Patients are Children
    # NOTE: Filtering on datasets where it is known if there are adult/children
    mask = df_metadata[PEDS_VS_ADULT_COL].notnull() & df_metadata["num_patients"].notnull() & df_metadata[prop_adult].notnull()
    num_children = round((df_metadata[mask]["num_patients"] * (1 - df_metadata[mask][prop_adult])).sum())
    total_num_patients = df_metadata[mask]["num_patients"].sum()
    descriptions["Prop. Children"] = round(num_children / total_num_patients, 4)
    descriptions["Num. Children"] = int(num_children)
    descriptions["Num. Adult"] = int(total_num_patients - num_children)
    df_child_count = pd.DataFrame({
        "is_child": (["Child"] * int(num_children)) + (["Adult"] * int(total_num_patients - num_children)),
    })

    return df_metadata, descriptions


def parse_demographics_columns(df_metadata):
    """
    Parse demographics-related columns in the metadata table.

    This function processes the 'Patient Demographics / Covariates / Metadata' column
    from the input DataFrame by converting its contents into a structured format.
    It appends the parsed demographic data as new columns to the DataFrame.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        The metadata DataFrame containing demographics-related information.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with additional columns for each demographic attribute.
    """
    df_metadata = df_metadata.copy()

    # Reset index
    df_metadata = df_metadata.reset_index(drop=True)

    # Add demographics columns
    demographics_data = pd.DataFrame.from_dict(df_metadata[DEMOGRAPHICS_COL].map(parse_text_to_dict).tolist())
    df_metadata = pd.concat([df_metadata, demographics_data], axis=1)

    return df_metadata


def parse_text_to_dict(text):
    """
    Parse a string of text into a dictionary.

    The input string is split into individual lines, and each line is split into
    a key-value pair by the first ": " encountered. The resulting dictionary is
    returned.

    Parameters
    ----------
    text : str
        The string of text to parse

    Returns
    -------
    dict
        A dictionary representation of the input string
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = text.split("\n")
    accum_ret = {}
    for line in lines:
        if not line.strip():
            continue
        # NOTE: Ensure all lines have ":"
        try:
            key, value = line.split(": ")
        except ValueError:
            assert False, (f"Warning: Line doesn't contain `: `! Line: {line}")
        accum_ret[key.strip()] = value.strip()
    return accum_ret


def extract_years_from_str(text):
    """
    Extract years from string

    Parameters
    ----------
    text : str
        String to extract years from

    Returns
    -------
    int or None
        Approximate age in years
    """
    # Early return
    if not text or not isinstance(text, str):
        return None
    try:
        # CASE 1: If years provided
        if "years" in text:
            years = int(float(text.replace(" years", "")))
            return years
        # CASE 2: If months provided
        if "months" in text:
            months = float(text.replace("months", ""))
            return int(months / 12)
        # CASE 3: If days provided
        if "D" in text:
            days = float(text.replace("days", ""))
            return int(days / 365)
    except ValueError as error_msg:
        print(error_msg)
        return None


def parse_percentage_from_text(text, default_val=None):
    """
    Parse a string of text into a float representing a percentage.

    Parameters
    ----------
    text : str
        The string of text to parse
    default_val : float, optional
        The default value to return if failed to parse, by default None

    Returns
    -------
    float or None
        The parsed percentage or None if failed to parse
    """
    parts = text.split("(")

    # Early return with None, if failed to parse
    if len(parts) == 1:
        return default_val

    # Attempt to parse
    try:
        curr_text = parts[-1].replace(">", "")
        percentage = float(curr_text.split("%)")[0])
        return percentage
    except ValueError:
        print(f"Failed to parse percentage from the text: `{text}`")
        return default_val


def convert_age_range_to_int(text):
    """
    Convert age range to a tuple of ints

    Parameters
    ----------
    age_range : str
        The age range to convert

    Returns
    -------
    tuple
        A tuple of ints representing the min and max age (in years)
    """
    if not isinstance(text, str) or not text.strip():
        return None
    # CASE 1: Days
    if text.endswith(" days"):
        text = text.replace(" days", "")
        lower, upper = map(float, text.split(" to "))
        if upper < 365:
            return 0, 0
        return lower//365, upper//365
    # SPECIAL CASE: Ends with "and above"
    if text.endswith(" and above"):
        text = text.replace(" and above", "")
        text = text.replace(" years", "")
        return int(float(text)), None
    # CASE 2: Otherwise, assume years
    sep = " to " if " to " in text else "-"
    lower, upper = list(map(parse_age_to_years, text.split(sep)))
    return lower, upper


def parse_age_to_years(text):
    """
    Parse age in years from text

    Parameters
    ----------
    text : str
        Text to parse

    Returns
    -------
    int
        Number of years
    """
    text = text.strip()
    if text.endswith("days"):
        return float(text.replace("days", "")) / 365
    if text.endswith("months"):
        return float(text.replace("months", "")) / 12
    return float(text.replace("years", ""))


def parse_female_prop(text):
    """
    Parse a string containing gender proportions into a dictionary.

    Parameters
    ----------
    text : str
        A string of gender labels and their proportions.

    Returns
    -------
    float
        Female proportion
    """
    if not isinstance(text, str) or text is None:
        return None
    parts = text.split(", ")
    gender_to_perc = {
        part.split(" (")[0].strip(): parse_percentage_from_text(part) / 100
        for part in parts
    }
    # CASE 1: Female proportion exists
    if "F" in gender_to_perc:
        return gender_to_perc["F"]
    # CASE 2: Only male proportion exists
    elif "M" in gender_to_perc and len(gender_to_perc) == 1:
        return 1 - gender_to_perc["M"]
    else:
        return None


def parse_perc_from_text(text):
    """
    Parse a string containing percentages into a dictionary.

    Note
    ----
    The text must look like "Column: Item1 (__%), Item2 (__%), ..."

    Parameters
    ----------
    text : str
        A string of gender labels and their proportions.

    Returns
    -------
    dict
        Dictionary of items to proportions
    """
    if not isinstance(text, str):
        return None
    # Remove demographics column from text, if any
    if ": " in text:
        text = text.split(": ")[-1]

    # Split into items
    parts = text.split(", ")

    # Extract percentage for each item
    item_to_perc = {
        part.split(" (")[0].strip(): parse_percentage_from_text(part) / 100
        for part in parts
    }
    return item_to_perc


def extract_list_of_age_ranges(age_ranges_str, sep="-"):
    """
    Extract minimum and maximum age from a string of age ranges.

    Parameters
    ----------
    age_ranges_str : str
        String containing comma-separated age ranges, where each range is
        separated by a hyphen (or specified separator).
        Example: "0-2, 3-5, 6-10"
    sep : str, optional
        Separator used between numbers in each range. Default is "-".

    Returns
    -------
    tuple or None
        A tuple containing (min_age, max_age) across all ranges.
        Returns None if input is not a string or is empty.
    """
    if not isinstance(age_ranges_str, str) or not age_ranges_str:
        return None
    age_ranges = age_ranges_str.split(", ")
    # NOTE: When extracting age ranges, make assumption to ignore cases where
    #       upper age bound is not definite (e.g., 65+)
    age_ranges = [[int(i.replace("+", "")) for i in curr_range.split(sep)] for curr_range in age_ranges]
    # Get minimum and maximum age
    min_age = min([curr_range[0] for curr_range in age_ranges])
    max_age = max([curr_range[-1] for curr_range in age_ranges])
    return (min_age, max_age)


def prop_to_perc(proportion):
    """
    Convert proportion to percentage
    """
    return round(100 * proportion, 2)


def floor_to_decimal(x, decimals=0):
    """
    Round down number

    Parameters
    ----------
    x : float
        Number
    decimals : int
        Number of decimals
    """
    factor = 10 ** decimals
    return math.floor(x * factor) / factor


def split_camel_case(text):
    """
    Split camel case into words

    Parameters
    ----------
    text : str
        Any text

    Returns
    -------
    str
        Text where camelcase words are split (e.g., HiMyNameIs -> Hi My Name Is)
    """
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)


def extract_from_multi_str(split_str, index=0):
    """
    Extract specific data (A1, A2, A3) or (B1, B2, B3) from strings of the form:
    ```
        A1 (B1)
        A2 (B2)
        A3 (B3)
    ```

    Parameters
    ----------
    split_str : str
        String
    index : int, optional
        Index into first (A) or second (B), by default 0
    """
    if not isinstance(split_str, str):
        return []
    lines = split_str.split("\n")
    accum_data = []
    for idx, line in enumerate(lines):
        if not line:
            continue
        try:
            item = line.split("(")[index].split(")")[0].strip()
            accum_data.append(item)
        except:
            print(f"Failed to extract data (index={index}) from line: \n\t`{line}`")
    return accum_data


def flatten_nested_list(nested_lst):
    """
    Flatten a nested list

    Parameters
    ----------
    nested_lst : list
        A nested list (just two levels)

    Returns
    -------
    list
        A flattened list
    """
    return [item for sublist in nested_lst for item in sublist]


def compute_summary_stats(df_curr):
    """
    Compute summary statistics for a given DataFrame of dataset annotations.

    Parameters
    ----------
    df_curr : pd.DataFrame
        Processed annotations for each dataset

    Returns
    -------
    dict
        Summary statistics including number of institutions, countries,
        datasets, and patients.
    """
    accum_data = {}

    # Get all unique countries
    country_per_dset = df_curr[SOURCE_COL].map(
        lambda x: extract_from_multi_str(x, index=1),
    ).tolist()
    countries = sorted(set(flatten_nested_list(country_per_dset)))
    countries = [c for c in countries if c and c not in ["N/A", "NA", "None"]]
    accum_data["num_countries"] = len(countries)

    # Get all unique institutions
    institutions_per_dset = df_curr[SOURCE_COL].map(
        lambda x: extract_from_multi_str(x, index=0),
    ).tolist()
    institutions = sorted(set(flatten_nested_list(institutions_per_dset)))
    institutions = [i for i in institutions if i and i not in ["N/A", "NA", "None"]]
    accum_data["num_institutions"] = len(institutions)

    # Get number of datasets
    # 1. Total
    accum_data["num_datasets_total"] = len(df_curr)

    # 2. Filtered
    filter_mask = df_curr["Prop. Adult"].notnull() & df_curr["num_patients"].notnull()
    accum_data["num_datasets_filtered"] = filter_mask.sum()

    # Get number of patients in total
    df_filtered = df_curr[filter_mask]
    num_adults = (df_filtered["Prop. Adult"] * df_filtered["num_patients"]).sum()
    num_children = ((1 - df_filtered["Prop. Adult"]) * df_filtered["num_patients"]).sum()

    # 1. Total
    accum_data["num_total"] = int(num_children + num_adults)

    # 2. Children
    accum_data["num_children"] = int(num_children)

    # 3. Percentage of Children
    accum_data["perc_children"] = round(100 * num_children / (num_children + num_adults), 2)

    return accum_data


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
