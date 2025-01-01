"""
py

Description: Used to create figures from open medical imaging metadata
"""

# Standard libraries
import json
import os
import warnings
from functools import reduce

# Non-standard libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire

# Custom libraries
from config import constants
from src.utils.data import viz_data


################################################################################
#                                    Config                                    #
################################################################################
warnings.filterwarnings("ignore")

# Configure plotting
viz_data.set_theme(figsize=(16, 8), tick_scale=1.9)


################################################################################
#                                  Constants                                   #
################################################################################
# Mapping of data category to sheet number in XLSX metadata file
CATEGORY_TO_SHEET = {
    "challenges": 0,
    "benchmarks": 1,
    "datasets": 2,
    "collections": 3,
    "papers": 6,
    # Following are specific image collections
    "stanford_aimi": 4, # "Image Collection (Stanford AIMI)",
    "tcia": 5, #"Image Collection (TCIA)",
}

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
CATEGORY_ORDER = ["Data Collections", "Challenges", "Benchmarks", "Well-Cited Datasets", "Conference Papers"]

# Metadata columns
DEMOGRAPHICS_COL = "Patient Demographics / Covariates / Metadata"
MODALITY_COL = "Imaging Modality(ies)"
ORGAN_COL = "Organ / Body Part"
TASK_COL = "Task Category"
CONTAINS_CHILDREN_COL = "Contains Children"
HAS_FINDINGS_COL = "Patients With Findings"
SOURCE_COL = "Source / Institutions (Location)"


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
        self.descriptions["Number of Datasets (With Age)"] = self.df_metadata["Peds vs. Adult"].notnull().sum()

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
        self.descriptions["Number of Unique Institutions"] = institutions.nunique()
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
        self.descriptions["Number of Unique Countries"] = countries.nunique()
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
        peds_col = "Peds vs. Adult"
        df_metadata = df_metadata.dropna(subset=peds_col)

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
        2. Add a column for Age Mentioned
        3. Add a column for Contains Children
        4. Plot Age Mentioned
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

        # Plot Age Mentioned
        ylabel_str = data_category_str if data_category_str == "Challenges" else "Datasets"
        ylabel = f"Number of {ylabel_str}"
        viz_data.catplot(
            df_metadata, x="Age Mentioned", hue="Contains Children",
            xlabel="", ylabel=ylabel,
            plot_type="count",
            order=["Yes", "No"],
            title=f"Do {data_category_str} Describe The Patients Age?",
            legend=True,
            save_dir=save_dir,
            save_fname="age_mentioned(bar).png",
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
def visualize_annotated_data():
    cat_to_descriptions = {}
    # ["collections", "challenges", "benchmarks", "datasets", "papers", "stanford_aimi", "tcia"]
    categories = ["collections_datasets", "collections_openneuro"]
    categories = ["collections", "challenges", "benchmarks", "datasets", "papers"]
    for category in categories:
        df_curr = load_annotations(category)
        visualizer = OpenDataVisualizer(df_curr, category, create_plots=False, save=False)
        try:
            cat_to_descriptions[category] = visualizer.describe()
        except:
            cat_to_descriptions[category] = visualizer.descriptions

    print(json.dumps(cat_to_descriptions, indent=4))


def describe_collections():
    cat_to_descriptions = {}
    for collection in ["stanford_aimi", "tcia"]:
        df_curr = load_annotations(collection)
        visualizer = OpenDataVisualizer(df_curr, collection)
        cat_to_descriptions[collection] = visualizer.describe()
    print(json.dumps(cat_to_descriptions, indent=4))


################################################################################
#                          Aggregating Plot Functions                          #
################################################################################
def plot_age_mentioned():
    """
    Create horizontal stacked bar plot of how age is mentioned across all data
    categories.
    """
    age_mentioned_col = "Age Mentioned How"
    DEMOGRAPHICS_COL = "Patient Demographics / Covariates / Metadata"

    def compute_age_mentioned_percentage(df):
        # Fill missing with "Not Mentioned"
        df[age_mentioned_col] = df[age_mentioned_col].fillna("Not Mentioned")
        # Summarize the percentages of age mentioned
        age_mentioned_percs = (100 * df[age_mentioned_col].value_counts(normalize=True)).round(1).reset_index()
        age_mentioned_percs.columns = [age_mentioned_col, "Percentage"]
        return age_mentioned_percs

    # What proportion of datasets providing summary statistics is enough to know
    # if there are children
    def summary_stats_contain_age_summary(df):
        mask = df[age_mentioned_col] == "Summary Statistics"
        if not mask.sum():
            print("Data provided does not have rows with summary statistics")
            return None
        if DEMOGRAPHICS_COL not in df.columns.tolist():
            print("Demographics column not found in the dataframe")
            return None
        df_summary = df[mask]
        stats = {
            "contain_age_range": round(100 * df_summary[DEMOGRAPHICS_COL].str.contains("Age Range:").mean(), 2),
            "contain_avg_age": round(100 * df_summary[DEMOGRAPHICS_COL].str.contains("Avg. Age:").mean(), 2),
            "contain_median_age": round(100 * df_summary[DEMOGRAPHICS_COL].str.contains("Median Age:").mean(), 2),
        }
        return stats

    # Summarize the percentages of age mentioned for each category
    accum_percentages = []
    accum_age_range_given = []
    for data_category in ["challenges", "benchmarks", "datasets", "papers"]:
        df_curr = load_annotations(data_category)
        age_mentioned_percs = compute_age_mentioned_percentage(df_curr)
        age_mentioned_percs["Category"] = CATEGORY_TO_STRING[data_category]
        accum_percentages.append(age_mentioned_percs)
        accum_age_range_given.append(summary_stats_contain_age_summary(df_curr))

    # SPECIAL CASE: Collections
    # 1. Dataset-level Annotations
    accum_collection_percentages = []
    accum_collection_age_range_given = []
    for collection in ["collections_openneuro", "tcia", "stanford_aimi"]:
        df_curr = load_annotations(collection)
        accum_collection_percentages.append(compute_age_mentioned_percentage(df_curr))
        accum_collection_age_range_given.append(summary_stats_contain_age_summary(df_curr))

    # 2. Collection-level Annotations
    df_metadata_collections = load_annotations("collections")
    num_total_collections = len(df_metadata_collections)
    # HACK: Filter for UK Biobank and MIDRC. This needs to be changed if more collections are added
    mask = df_metadata_collections["Image Collection"].isin(["UK Biobank", "MIDRC"])
    df_metadata_collections = df_metadata_collections[mask]
    accum_collection_percentages.append(
        df_metadata_collections.groupby("Image Collection").apply(
        compute_age_mentioned_percentage).reset_index(drop=True))
    # Average percentages across collections
    df_collections = pd.concat(accum_collection_percentages, ignore_index=True, axis=0)
    df_collections_agg = df_collections.groupby(age_mentioned_col).apply(lambda df: df["Percentage"].sum() / num_total_collections).reset_index()
    df_collections_agg.columns = [age_mentioned_col, "Percentage"]
    df_collections_agg["Category"] = CATEGORY_TO_STRING["collections"]
    accum_percentages.append(df_collections_agg)

    # Concatenate percentages
    df_percentages_all = pd.concat(accum_percentages, ignore_index=True, axis=0)

    # Sort by the following Age Mentioned
    how_order = ["Not Mentioned", "Task/Data Description", "Summary Statistics", "Binned Patient-Level", "Patient-Level"]
    how_colors = ["#960f0b", "#b0630c", "#300859", "#bab21e", "#056b19"]
    how_order_and_color = list(reversed(tuple(zip(how_order, how_colors))))
    df_percentages_all[age_mentioned_col] = pd.Categorical(df_percentages_all[age_mentioned_col], categories=how_order, ordered=True)
    df_percentages_all = df_percentages_all.sort_values(by=age_mentioned_col).reset_index(drop=True)
    df_cum_percentages = df_percentages_all.copy()
    # Add them together iteratively to create the heights needed in the plot
    for idx, category in enumerate(df_percentages_all["Category"].unique()):
        mask = df_cum_percentages["Category"] == category
        df_cum_percentages.loc[mask, "Percentage"] = df_percentages_all.loc[mask, "Percentage"].cumsum()

    # Create a bar plot
    viz_data.set_theme(figsize=(16, 8), tick_scale=2.4)
    fig, ax = plt.subplots()
    new_colors = []
    for mentioned_how, how_color in how_order_and_color:
        df_curr_age_mentioned = df_cum_percentages[df_cum_percentages[age_mentioned_col] == mentioned_how]
        viz_data.catplot(
            df_curr_age_mentioned, x="Percentage", y="Category",
            plot_type="bar", saturation=0.6,
            color=how_color,
            order=CATEGORY_ORDER,
            hue_order=how_order,
            xlabel="Percentage (%)", ylabel="",
            x_lim=(0, 100),
            tick_params={"axis": "y", "left": False, "labelleft": False},
            title="How is Age Mentioned?",
            legend=False,
            ax=ax,
        )
        curr_plotted_colors = set(patch.get_facecolor() for patch in ax.patches)
        curr_plotted_colors = curr_plotted_colors.difference(set(new_colors))
        new_colors.append(list(curr_plotted_colors)[0])

    # Create custom legend at the bottom
    legend_handles = [
        mpatches.Patch(color=new_colors[idx], label=mentioned_how)
        for idx, mentioned_how in enumerate(reversed(how_order))
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
    save_fname = "age_mentioned_how (bar).svg"
    fig.savefig(os.path.join(save_dir, save_fname), bbox_inches="tight")


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
    viz_data.set_theme(figsize=(12, 8), tick_scale=2)
    viz_data.catplot(
        df_perc, x="Percentage", y="Country",
        plot_type="bar", color="#274a7a", saturation=0.75,
        tick_params={"axis":"y", "left": False},
        xlabel="Percentage (%)", ylabel="",
        order=order,
        title="What Countries Contribute the Most Data?",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="countries(bar).svg",
    )


def plot_task_types(task_types):
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
    viz_data.set_theme(figsize=(12, 8), tick_scale=2)
    viz_data.catplot(
        df_perc, x="Percentage", y="Task Type",
        plot_type="bar", color="#274a7a", saturation=0.75,
        xlabel="Percentage (%)", ylabel="",
        tick_params={"axis":"y", "left": False},
        order=order,
        title="Most Common Tasks",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="task_categories(bar).svg",
    )


def plot_modalities(modalities):
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
    viz_data.set_theme(figsize=(12, 8), tick_scale=2)
    viz_data.catplot(
        df_perc, x="Percentage", y="Modality",
        plot_type="bar", color="#274a7a", saturation=0.75,
        xlabel="Percentage (%)", ylabel="",
        tick_params={"axis":"y", "left": False},
        order=order,
        title="Most Common Imaging Modalities",
        legend=False,
        save_dir=os.path.join(constants.DIR_FIGURES_EDA, "open_mi"),
        save_fname="modalities(bar).svg",
    )


################################################################################
#                               Helper Functions                               #
################################################################################
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
        df_metadata["Peds vs. Adult"] = None
        df_metadata.loc[contains_children, "Peds vs. Adult"] = "Peds"
        df_metadata.loc[contains_adults, "Peds vs. Adult"] = "Adult"
        df_metadata.loc[contains_children & contains_adults, "Peds vs. Adult"] = "Peds, Adult"

        # Create a column for age mentioned as Patient-Level
        mask = df_metadata["age_range"].notnull()
        df_metadata.loc[mask, "Age Mentioned How"] = "Patient-Level"
        return df_metadata

    # SPECIAL CASE: If data category is `collections_datasets`, load Stanford AIMI and TCIA
    if data_category == "collections_datasets":
        df_metadata = pd.concat([
            pd.read_excel(metadata_path, CATEGORY_TO_SHEET[curr_category])
            for curr_category in ["stanford_aimi", "tcia"]
        ], ignore_index=True, axis=0)
        return df_metadata

    # DEFAULT CASE: Any other category
    df_metadata = pd.read_excel(metadata_path, sheet_name=CATEGORY_TO_SHEET[data_category])
    
    # CASE 1: If data category is `papers`, filter for those in in inclusion criteria
    if data_category == "papers":
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

        # Print number of papers with public data
        public_mask = df_metadata[public_data_col].str.contains("Public")
        print(f"Number of Papers with Public Data: {public_mask.sum()} / {len(public_mask)} ({public_mask.mean().round(4)})")

        # Print number of papers that mention age
        age_mentioned_mask = df_metadata["Is Age Explicitly Mentioned"]
        print(f"Number of Papers that Mention Age: {age_mentioned_mask.sum()} / {len(age_mentioned_mask)} ({age_mentioned_mask.mean().round(4)})")

        # Print number of paper with peds data
        contains_peds_mask = df_metadata["Peds vs. Adult"].str.contains("Peds")
        print(f"Number of Papers Known To Have Peds Data: {contains_peds_mask.sum()} / {len(contains_peds_mask)} ({contains_peds_mask.mean().round(4)})")

        # Number of datasets where we can infer adult/peds
        dataset_col = "Dataset Name/s (if any)"
        num_datasets = df_metadata.loc[contains_peds_mask, dataset_col].str.split(", ").explode().nunique()
        print(f"Number of Datasets Known To Have Peds Data: {num_datasets}")

    return df_metadata


def parse_sample_size_column(df_metadata):
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
    the age is mentioned and whether the dataset contains children or adults. It also
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
    peds_col = "Peds vs. Adult"
    mask_adult_only = df_metadata[peds_col].str.startswith("Adult").fillna(False)
    mask_peds_only = (df_metadata[peds_col] == "Peds").fillna(False)
    mask_peds_and_adult = (df_metadata[peds_col].str.startswith("Peds, Adult")).fillna(False)

    # 2.1. Add column for Age Mentioned
    df_metadata["Age Mentioned"] = "No"
    age_mentioned = ~df_metadata[peds_col].isna()
    df_metadata.loc[age_mentioned,"Age Mentioned"] = "Yes"

    descriptions["Number of Patients (With Age Mentioned)"] = int(df_metadata.loc[age_mentioned, "num_patients"].sum())

    # 2.2. Add column for Contains Children
    df_metadata[CONTAINS_CHILDREN_COL] = "Unknown"
    df_metadata.loc[mask_adult_only, CONTAINS_CHILDREN_COL] = "Adult Only"
    df_metadata.loc[mask_peds_only, CONTAINS_CHILDREN_COL] = "Peds Only"
    df_metadata.loc[mask_peds_and_adult, CONTAINS_CHILDREN_COL] = "Peds & Adult"

    # 2.4. Add column for Proportion of Patients are Adults
    # NOTE: If no annotation for child/adult, we will assume the proportion
    #       of children matches the world population statistics (29.85%)
    # Sources: https://data.unicef.org/how-many/how-many-children-under-18-are-in-the-world/
    #          https://data.unicef.org/how-many/how-many-people-are-in-the-world/
    prop_adult = "Prop. Adult"
    df_metadata[prop_adult] = None
    df_metadata.loc[mask_adult_only, prop_adult] = 1.
    df_metadata.loc[mask_peds_only, prop_adult] = 0.
    df_metadata.loc[mask_peds_and_adult, prop_adult] = df_metadata.loc[mask_peds_and_adult, peds_col].map(
        parse_percentage_from_text) / 100

    # 2.5. Plot Proportion of Patients are Children
    # NOTE: Filtering on datasets where it is known if there are adult/children
    mask = ~df_metadata[peds_col].isna()
    num_children = round((df_metadata[mask]["num_patients"] * (1 - df_metadata[mask][prop_adult]))).sum()
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


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
