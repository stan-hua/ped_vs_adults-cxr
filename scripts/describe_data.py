"""
describe_data.py

Description: Used to create figures from open medical imaging metadata
"""

# Standard libraries
import logging
import json
import os
import warnings
from functools import reduce

# Non-standard libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fire import Fire
from matplotlib.colors import ListedColormap

# Custom libraries
from config import constants
from src.utils.data import viz_data


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
    "stanford_aimi",        # "Image Collection (Stanford AIMI)",
    "tcia",                 # "Image Collection (TCIA)",
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
    "collections_datasets": "Data Dollections (Stanford AIMI & TCIA)",
    "collections_openneuro": "Data Dollection (OpenNeuro)",
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
    df_annotations = load_all_dataset_annotations(**load_kwargs)

    # Print the proportion of datasets with peds
    print("==All Datasets==:")
    print("Number of Datasets:", len(df_annotations))
    df_count_and_perc = df_annotations[CONTAINS_CHILDREN_COL].value_counts().to_frame()
    df_count_and_perc["Proportion"] = df_count_and_perc["count"] / df_count_and_perc["count"].sum()
    print(df_count_and_perc)

    # Print the proportion of children in datasets with both adult and peds
    # mask = df_annotations[CONTAINS_CHILDREN_COL] == "Peds & Adult"
    # print("% of Children in Peds & Adult Datasets:", prop_to_perc(1-df_annotations.loc[mask, "Prop. Adult"].mean()))

    # Print the number of estimated adults vs. children
    has_prop_adult_mask = df_annotations["Prop. Adult"].notnull() & df_annotations["num_patients"].notnull()
    df_ages = df_annotations[has_prop_adult_mask]
    num_adults = (df_ages["Prop. Adult"] * df_ages["num_patients"]).sum()
    num_children = ((1 - df_ages["Prop. Adult"]) * df_ages["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print(f"[Num. Datasets w/ Age: {float(has_prop_adult_mask.sum()):.0f}] {num_children} children / {int(num_adults+num_children)} ({round(prop_children, 4)}) overall")
    print("")

    ############################################################################
    #             Sensitivity Analysis without Cancer Datasets                 #
    ############################################################################
    df_annot_wo_cancer = load_all_dataset_annotations(exclude_cancer=True)

    # Print the number of estimated adults vs. children
    has_prop_adult_mask = df_annot_wo_cancer["Prop. Adult"].notnull() & df_annot_wo_cancer["num_patients"].notnull()
    df_ages = df_annot_wo_cancer[has_prop_adult_mask]
    num_adults = (df_ages["Prop. Adult"] * df_ages["num_patients"]).sum()
    num_children = ((1 - df_ages["Prop. Adult"]) * df_ages["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print(f"==Sensitivity Analysis Without Cancer Datasets==: ({len(df_annot_wo_cancer)}/{len(df_annotations)})")
    print(f"[Num. Datasets w/ Age: {round(has_prop_adult_mask.sum())}] {num_children} children / {num_adults+num_children} ({round(prop_children, 4)}) overall")
    print("")

    ############################################################################
    #                              Challenges                                  #
    ############################################################################
    df_challenges = load_all_dataset_annotations(filter_categories=["challenges"], **load_kwargs)

    # Number of children vs. adults
    num_adults = (df_challenges["Prop. Adult"] * df_challenges["num_patients"]).sum()
    num_children = ((1 - df_challenges["Prop. Adult"]) * df_challenges["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print("==Challenges==:")
    print(f"[Challenges] {int(num_children)} children / {int(num_adults+num_children)} overall ({100*prop_children:.2f}%)")

    # Filter for datasets with both adult and peds
    mask = df_challenges[CONTAINS_CHILDREN_COL] == "Peds & Adult"
    df_peds_and_adult = df_challenges[mask]
    null_mask = df_peds_and_adult[["Prop. Adult", "num_patients"]].isna().any(axis=1)
    num_adults = (df_peds_and_adult["Prop. Adult"] * df_peds_and_adult["num_patients"]).sum()
    num_children = ((1 - df_peds_and_adult["Prop. Adult"]) * df_peds_and_adult["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print(f"[Num. Datasets w/ Age: {(~null_mask).sum()}] {num_children} children / {num_adults+num_children} overall")

    # How many don't report age?
    age_missing = df_challenges["Prop. Adult"].isna()
    prop_age_missing = age_missing.mean()
    print(f"[Challenges] {age_missing.sum()} datasets missing age / {len(age_missing)} overall ({100*prop_age_missing:.2f}%)")
    print("")

    ############################################################################
    #                              Benchmarks                                  #
    ############################################################################
    df_benchmarks = load_all_dataset_annotations(filter_categories=["benchmarks"], **load_kwargs)

    # Number of children vs. adults
    null_mask = df_benchmarks[["Prop. Adult", "num_patients"]].isna().any(axis=1)
    num_adults = (df_benchmarks["Prop. Adult"] * df_benchmarks["num_patients"]).sum()
    num_children = ((1 - df_benchmarks["Prop. Adult"]) * df_benchmarks["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print("==Benchmarks==:")
    print(f"[Benchmarks] {int(num_children)} children / {int(num_adults+num_children)} overall ({100*prop_children:.2f}%)")

    # How many don't report age?
    age_missing = df_benchmarks["Prop. Adult"].isna()
    prop_age_missing = age_missing.mean()
    print(f"[Benchmarks] {age_missing.sum()} datasets missing age / {len(age_missing)} overall ({100*prop_age_missing:.2f}%)")
    print("")


    #############################################################################
    #                         Highly-Cited Datasets                             #
    #############################################################################
    df_datasets = load_all_dataset_annotations(filter_categories=["datasets"], **load_kwargs)
    null_mask = df_datasets[["Prop. Adult", "num_patients"]].isna().any(axis=1)
    num_adults = (df_datasets["Prop. Adult"] * df_datasets["num_patients"]).sum()
    num_children = ((1 - df_datasets["Prop. Adult"]) * df_datasets["num_patients"]).sum()
    prop_children = num_children / (num_children + num_adults)
    print("==Highly-Cited Datasets==:")
    print(f"[Datasets] {int(num_children)} children / {int(num_adults+num_children)} overall ({100*prop_children:.2f}%)")

    # Proportion of datasets missing age
    num_missing_age = df_datasets["Age Documented How"].isna().sum()
    num_total = len(df_datasets)
    prop_missing_age = num_missing_age / num_total
    print(f"[Datasets] {num_missing_age} datasets missing age / {num_total} overall ({100*prop_missing_age:.2f}%)")
    print("")

    ############################################################################
    #                             Collections                                  #
    ############################################################################
    # Get number of datasets missing age among dataset collections
    # 1. OpenNeuro
    df_openneuro = load_annotations("collections_openneuro")
    num_missing_age = df_openneuro['Age Documented How'].isna().sum()
    num_total = len(df_openneuro)

    # 2. Stanford AIMI and TCIA
    df_aimi_tcia = load_all_dataset_annotations(filter_categories=["collections_datasets"], **load_kwargs)
    num_total += len(df_aimi_tcia)
    num_missing_age += df_aimi_tcia[CONTAINS_CHILDREN_COL].isna().sum()
    num_missing_age += (df_aimi_tcia[CONTAINS_CHILDREN_COL] == "Unknown").sum()

    # Proportion of datasets missing age
    prop_missing_age = num_missing_age / num_total
    print("==Collections (OpenNeuro, Stanford AIMI, TCIA)==:")
    print(f"[Collections] {num_missing_age} datasets missing age / {num_total} overall ({100*prop_missing_age:.2f}%)")


def describe_peds_broken_by_modality_task(filter_peds_vs_adult=True, load_kwargs=None):
    """
    Create table that shows the number of data points by modality and task.

    Parameters
    ----------
    filter_peds_vs_adult : bool, optional
        If True, filter for pediatric data. If False, filter for adult data.
        Default is True.
    """
    load_kwargs = load_kwargs or {}

    # Accumulate all dataset metadata
    # NOTE: Now, we assume that fill in missing % adult with the avg. % of peds
    #       data in Peds, Adult datasets
    df_annotations = load_all_dataset_annotations(**load_kwargs)

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
    for modality in modalities:
        # At the dataset level, check how many datasets are peds only
        mask = df_annotations[MODALITY_COL].map(lambda x: modality in str(x).split(", "))
        df_curr = df_annotations[mask]
        # NOTE: Assume adult-only dataset, if age is unknown
        prop_peds = df_curr["Prop. Adult"]
        if filter_peds_vs_adult:
            prop_peds = (1 - prop_peds)
        # Get proportion of data that this modality represents, if multi-modal dataset
        # NOTE: If proportion for each modality is not annotated, assume it's
        #       equally split across modalities
        modality_prop = df_curr.apply(
            lambda row: row["Modalities"][modality] if row["Modalities"] and modality in row["Modalities"]
                        else 1/len(row[MODALITY_COL].split(", ")),
            axis=1
        )
        # Get the correct estimate on the number of sequences for the modality
        num_sequences = df_curr["num_sequences"] * modality_prop
        # Skip, if no modality-specific data
        if modality_prop.empty or num_sequences.sum() == 0:
            continue
        # Filter on specific tasks
        for task in tasks:
            curr_stats = {}
            curr_stats["Task"] = task
            # Get the datasets with this task
            if task == "ALL":
                curr_num_sequences = num_sequences
                curr_prop_peds = prop_peds
            else:
                mask_task = df_curr[TASK_COL].str.contains(task)
                curr_num_sequences = num_sequences[mask_task]
                curr_prop_peds = prop_peds[mask_task]
            # Now, estimate how much of these modality-specific sequences are peds
            curr_stats[f"{modality}"] = int((curr_num_sequences * curr_prop_peds).sum())
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
    df_stats = df_stats.sort_values("ALL", ascending=False)
    order = ["X-ray", "US", "CT", "MRI", "Fundus"]
    df_stats = df_stats.loc[order]
    print(df_stats)
    df_stats.to_csv("peds.csv" if filter_peds_vs_adult else "adult.csv")


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
    for data_category in ["papers", "challenges", "benchmarks", "datasets"]:
        df_curr = load_annotations(data_category)
        age_documented_percs = compute_age_documented_percentage(df_curr)
        age_documented_percs["Category"] = CATEGORY_TO_STRING[data_category]
        accum_percentages.append(age_documented_percs)
        accum_age_range_given.append(summary_stats_contain_age_summary(df_curr))

    # SPECIAL CASE: Collections
    # 1. Dataset-level Annotations
    accum_collection_percentages = []
    accum_collection_age_range_given = []
    for collection in ["collections_openneuro", "tcia", "stanford_aimi"]:
        df_curr = load_annotations(collection)
        accum_collection_percentages.append(compute_age_documented_percentage(df_curr))
        accum_collection_age_range_given.append(summary_stats_contain_age_summary(df_curr))

    # 2. Collection-level Annotations
    df_metadata_collections = load_annotations("collections")
    num_total_collections = len(df_metadata_collections)
    # HACK: Filter for UK Biobank and MIDRC. This needs to be changed if more collections are added
    mask = df_metadata_collections["Image Collection"].isin(["UK BioBank", "MIDRC"])
    df_metadata_collections = df_metadata_collections[mask]
    accum_collection_percentages.append(
        df_metadata_collections.groupby("Image Collection").apply(
        compute_age_documented_percentage).reset_index(drop=True))
    # Average percentages across collections
    df_collections = pd.concat(accum_collection_percentages, ignore_index=True, axis=0)
    df_collections_agg = df_collections.groupby(age_documented_col).apply(lambda df: df["Percentage"].sum() / num_total_collections).reset_index()
    df_collections_agg.columns = [age_documented_col, "Percentage"]
    df_collections_agg["Category"] = CATEGORY_TO_STRING["collections"]
    accum_percentages.append(df_collections_agg)

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
    for idx, category in enumerate(df_percentages_all["Category"].unique()):
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


def plot_countries(countries=None):
    countries = countries or ("USA", "China", "Netherlands", "Canada", "Germany")
    # Count the number of times each specified country has appeared in a dataset
    country_to_count = {country: 0 for country in countries}
    num_datasets = 0
    for data_category in ["challenges", "benchmarks", "datasets", "stanford_aimi", "tcia"]:
        df_curr = load_annotations(data_category)
        num_datasets += len(df_curr)
        # For each row, check if each of the country is included
        rows = df_curr[SOURCE_COL].tolist()
        for institutions_str in rows:
            if not isinstance(institutions_str, str):
                continue
            for country in countries:
                if f"({country})" in institutions_str:
                    country_to_count[country] += 1

    # Divide by the number of datasets to get the percentage
    country_to_perc = {country: 100 * count / num_datasets for country, count in country_to_count.items()}
    # Convert to dataframe
    df_perc = pd.DataFrame(country_to_perc.items(), columns=["Country", "Percentage"])
    df_perc = df_perc.sort_values(by="Percentage", ascending=False)
    order = df_perc["Country"].tolist()

    # Create bar plot
    viz_data.set_theme(figsize=(12, 8), tick_scale=3)
    viz_data.catplot(
        df_perc, x="Percentage", y="Country",
        plot_type="bar", color="#512200", saturation=0.75,
        tick_params={"axis":"y", "left": False},
        xlabel="Percentage (%)", ylabel="",
        order=order,
        title="What Countries Contribute the Most Data?",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="countries(bar).svg",
    )


def plot_task_types(task_types=None):
    task_types = task_types or [
        "Anatomy/Organ Segmentation/Detection",
        "Lesion/Tumor Segmentation/Detection",
        "Disease (Risk) Segmentation/Detection",
        "Condition/Disease Classification",
        "Image Reconstruction/Generation (Enhancement/Registration)",
    ]
    # Count the number of times each specified task type has appeared in a dataset
    task_type_to_count = {task_type: 0 for task_type in task_types}
    num_datasets = 0
    for data_category in ["challenges", "benchmarks", "datasets", "stanford_aimi", "tcia"]:
        df_curr = load_annotations(data_category)
        num_datasets += len(df_curr)
        # For each row, check if each of the task_type is included
        rows = df_curr[TASK_COL].tolist()
        for element_str in rows:
            if not isinstance(element_str, str):
                continue
            for task_type in task_types:
                if f"{task_type}" in element_str:
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
        xlabel="Percentage (%)", ylabel="",
        tick_params={"axis":"y", "left": False},
        order=order,
        title="Most Common Tasks",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="task_categories(bar).svg",
    )


def plot_modalities(modalities=None):
    modalities = modalities or ["CT", "MRI", "X-ray", "US", "Fundus"]
    # Count the number of times each specified modality has appeared in a dataset
    modality_to_count = {modality: 0 for modality in modalities}
    num_datasets = 0
    for data_category in ["challenges", "benchmarks", "datasets", "stanford_aimi", "tcia"]:
        df_curr = load_annotations(data_category)
        num_datasets += len(df_curr)
        # For each row, check if each of the modality is included
        rows = df_curr[MODALITY_COL].tolist()
        for element_str in rows:
            if not isinstance(element_str, str):
                continue
            for modality in modalities:
                if f"{modality}" in element_str:
                    modality_to_count[modality] += 1

    # Divide by the number of datasets to get the percentage
    modality_to_perc = {modality: 100 * count / num_datasets for modality, count in modality_to_count.items()}
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


def plot_table_dataset_breakdown():
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
        # NOTE: Assume adult-only dataset, if age is unknown
        prop_peds = df_curr["Prop. Adult"]
        if filter_peds_vs_adult:
            prop_peds = (1 - prop_peds)
        # Get proportion of data that this modality represents, if multi-modal dataset
        # NOTE: If proportion for each modality is not annotated, assume it's
        #       equally split across modalities
        modality_prop = df_curr.apply(
            lambda row: row["Modalities"][modality] if row["Modalities"] and modality in row["Modalities"]
                        else 1/len(row[MODALITY_COL].split(", ")),
            axis=1
        )
        # Get the correct estimate on the number of sequences for the modality
        num_sequences = df_curr["num_sequences"] * modality_prop
        # Skip, if no modality-specific data
        if modality_prop.empty or num_sequences.sum() == 0:
            continue
        # Filter on specific tasks
        for task_idx, task in enumerate(tasks):
            curr_stats = {}
            curr_stats["Task"] = rename_tasks[task_idx]
            # Get the datasets with this task
            if task == "ALL":
                curr_num_sequences = num_sequences
                curr_prop_peds = prop_peds
            else:
                mask_task = df_curr[TASK_COL].str.contains(task)
                curr_num_sequences = num_sequences[mask_task]
                curr_prop_peds = prop_peds[mask_task]
            # Now, estimate how much of these modality-specific sequences are peds
            curr_stats[f"{modality}"] = int((curr_num_sequences * curr_prop_peds).sum())
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
        fill_missing_prop_adult=False,
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
    fill_missing_prop_adult : bool, optional
        Whether to fill in missing values for the `Prop. Adult` column for
        datasets that are marked as (Peds, Adult), where the specific percentage
        of children is not known
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
    cat_to_avg_prop_peds = {
        "challenges": 0.0189,
        "benchmarks": 0.0073,
        "datasets": 0.0494,
        "collections_datasets": 0.0216,
    }
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
                regex = r"cancer|tumor|tumour|carcinoma|sarcoma|glioma|metastasis|metastases"
                mask_cancer = df_curr[col].str.contains(regex, case=False, na=False)
                df_curr = df_curr[~mask_cancer]

        # If specified, fill in missing % adult with the hard-coded averages
        if fill_missing_prop_adult and category in cat_to_avg_prop_peds:
            avg_adult_prop = 1 - cat_to_avg_prop_peds[category]
            # NOTE: It should never be None
            df_curr["Prop. Adult"] = df_curr.apply(
                lambda row: avg_adult_prop if "Peds, Adult" in str(row[PEDS_VS_ADULT_COL]) and pd.isnull(row["Prop. Adult"])
                            else row["Prop. Adult"],
                axis=1
            )

        # Parse proportion of each modality for multi-modal datasets
        if "Modalities" in df_curr.columns:
            df_curr["Modalities"] = df_curr["Modalities"].map(parse_prop_from_text)
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
        - "collections_openneuro"
        - "collections_datasets" (loads the two below)
        - "stanford_aimi"
        - "tcia"
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

        # Rename column for number of participants
        df_metadata.rename(columns={"num_subjects": "num_patients"}, inplace=True)
        return df_metadata

    # SPECIAL CASE: If data category is `collections_datasets`, load Stanford AIMI and TCIA
    if data_category == "collections_datasets":
        df_metadata = pd.concat([
            pd.read_excel(metadata_path, CATEGORY_TO_SHEET[curr_category])
            for curr_category in ["stanford_aimi", "tcia"]
        ], ignore_index=True, axis=0)
        return df_metadata

    # SPECIAL CASE: If data category is `collections_midrc`, load collections and filter only for MIDRC
    if data_category == "collections_midrc":
        df_metadata = pd.read_excel(metadata_path, sheet_name=CATEGORY_TO_SHEET["collections"])
        df_metadata = df_metadata[df_metadata["Image Collection"] == "MIDRC"]
        return df_metadata

    # SPECIAL CASE: If data category is `collections_uk_biobank`, load collections and filter only for UK BioBank
    if data_category == "collections_uk_biobank":
        df_metadata = pd.read_excel(metadata_path, sheet_name=CATEGORY_TO_SHEET["collections"])
        df_metadata = df_metadata[df_metadata["Image Collection"] == "UK Biobank"]
        return df_metadata

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
    # CASE 2: If data category is Repackaging (Secondary), parse out Secondary Datasets column
    elif data_category == "repackaging_1st":
        col = "Secondary Datasets"
        df_metadata[col] = df_metadata[col].str.split("\n")

    # CASE 3: If data category is Repackaging (Secondary), remove KiTS23
    elif data_category == "repackaging_2nd":
        df_metadata = df_metadata[df_metadata["Dataset Name"] != "KiTS23"]

    return df_metadata


def parse_sample_size_column(df_metadata, assume_sequence=False):
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
        Images to # of Sequences. If only num. of patients provided, assume
        each patient has 1 sequence

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
    data_sizes = df_metadata["Sample Size"].str.split("\n").fillna("")
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
            "[Parse Sample Size] If 'Sequences: ' is missing, assuming # Images "
            "-> # Sequences. If # Images is missing, use # Patients instead!"
        )
        missing_mask = df_metadata["num_sequences"].isna()
        df_metadata.loc[missing_mask, "num_sequences"] = df_metadata.loc[missing_mask].apply(
            lambda row: row["num_images"] if not pd.isnull(row["num_images"])
                        else row["num_patients"],
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


def parse_prop_from_text(text):
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


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
