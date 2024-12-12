"""
describe_data.py

Description: Used to create figures from open medical imaging metadata
"""

# Standard libraries
import json
import os
import warnings

# Non-standard libraries
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
viz_data.set_theme()


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

    def __init__(
        self,
        data_category="challenges",
        metadata_path=constants.DIR_METADATA_MAP["open_data"],
    ):
        self.data_category = data_category
        cat_to_str = {
            "challenges": "Challenges",
            "benchmarks": "Benchmarks",
            "datasets": "Highly-Cited Datasets",
            "collections": "Image Collections (Datasets)",
            "stanford_aimi": "Image Collection (Stanford AIMI)",
            "tcia": "Image Collection (TCIA)",
        }
        self.data_category_str = cat_to_str[data_category]

        # Mapping of data category to sheet number in XLSX metadata file
        cat_to_sheet = {
            "challenges": 3,
            "benchmarks": 4,
            "datasets": 5,
            "collections": 6,
            "papers": 1,
            # Following are specific image collections
            "stanford_aimi": 8, # "Image Collection (Stanford AIMI)",
            "tcia": 9, #"Image Collection (TCIA)",
        }
        sheet_name = cat_to_sheet[data_category]

        # Load metadata table
        self.df_metadata = pd.read_excel(metadata_path, sheet_name=sheet_name)

        # Mapping of data category to save directory
        cat_to_dir = {
            "challenges": constants.DIR_FIGURES_EDA_CHALLENGES,
            "benchmarks": constants.DIR_FIGURES_EDA_BENCHMARKS,
            "datasets": constants.DIR_FIGURES_EDA_DATASETS,
            "papers": constants.DIR_FIGURES_EDA_PAPERS,
            "collections": constants.DIR_FIGURES_EDA_COLLECTIONS,
            "stanford_aimi": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "stanford_aimi"),
            "tcia": os.path.join(constants.DIR_FIGURES_EDA_COLLECTIONS, "tcia"),
        }

        # Store save directory
        self.save_dir = cat_to_dir[data_category]

        # Store constants to be filled in
        self.descriptions = {}


    def describe(self):
        self.descriptions["Number of Datasets"] = len(self.df_metadata)
        self.descriptions["Number of Datasets (With Age)"] = self.df_metadata["Peds vs. Adult"].notnull().sum()

        # Parse sample size and age columns
        self.parse_sample_size_column()
        self.parse_age_columns()

        # Call functions
        self.describe_data_provenance()
        self.describe_patients()
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

        # Set theme
        viz_data.set_theme()

        # 1. Where is the data hosted?
        # NOTE: Ignore if it's on Kaggle (but unofficial)
        if "Data Location" in df_metadata.columns:
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
        if self.data_category == "challenges":
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
        source_col = "Source / Institutions (Location)"
        df_metadata[source_col] = df_metadata[source_col].str.split("\n")

        # 3.2. Plot proportion of challenges with complete/partially complete/missing data source
        df_metadata["Is Data Source Known"] = df_metadata[source_col].map(
            lambda x: "Missing" if not isinstance(x, list) else (
                "Partial" if any("N/A" in item for item in x) else "Complete"
            )
        )
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
        institute_country = df_metadata[source_col].dropna().explode()
        institutions = institute_country[~institute_country.str.contains("N/A")]
        institutions = institutions.str.split(r" \(").str[0]
        institutions.name = "Institutions"
        self.descriptions["Number of Unique Institutions"] = institutions.nunique()
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
        institute_country = df_metadata[source_col].dropna().explode()
        countries = institute_country.str.split(r" \(").str[1].str.split(")").str[0]
        countries.name = "Country"
        countries.reset_index(drop=True, inplace=True)
        self.descriptions["Number of Unique Countries"] = countries.nunique()
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
        demographics_col = "Patient Demographics / Covariates / Metadata"
        save_dir = self.save_dir
        data_category_str = self.data_category_str

        # Set theme
        viz_data.set_theme()

        # Drop datasets without age annotation
        peds_col = "Peds vs. Adult"
        df_metadata = df_metadata.dropna(subset=peds_col)

        # Add demographics columns
        demographics_data = pd.DataFrame.from_dict(df_metadata[demographics_col].map(parse_text_to_dict).tolist())
        df_metadata = pd.concat([df_metadata, demographics_data], axis=1)

        # 3. Plot the kinds of youngest, central and oldest present
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

        # 4. Plot gender/sex
        # 4.1. Standardize sex to gender
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
        modality_col = "Imaging Modality(ies)"
        organ_col = "Organ / Body Part"
        task_col = "Task / Pathology"
        contains_children_col = "Contains Children"
        save_dir = self.save_dir
        data_category_str = self.data_category_str

        # Set theme
        viz_data.set_theme()

        # 1. Imaging modalities
        df_metadata[modality_col] = df_metadata[modality_col].str.split(", ")
        df_modalities = df_metadata.explode(modality_col).reset_index(drop=True)
        # Exclude PET imaging
        df_modalities = df_modalities[df_modalities[modality_col] != "PET"]
        self.descriptions["Modalities"] = df_modalities.groupby(contains_children_col)[modality_col].value_counts().round(2).to_dict()
        viz_data.catplot(
            df_modalities, y=modality_col, hue=contains_children_col,
            xlabel=f"Number of {data_category_str}", ylabel="",
            plot_type="count",
            order=df_modalities[modality_col].value_counts().index,
            title="What Imaging Modalities Are Most Commonly Used?",
            legend=True,
            save_dir=save_dir,
            save_fname="img_modalities(bar).png",
        )

        # 2. Organ / Body Part
        df_metadata[organ_col] = df_metadata[organ_col].str.split(", ")
        df_organs = df_metadata.explode(organ_col).reset_index(drop=True)
        self.descriptions["Organs"] = df_organs.groupby(contains_children_col)[organ_col].value_counts().round(2).to_dict()
        viz_data.catplot(
            df_organs, y=organ_col, hue=contains_children_col,
            xlabel=f"Number of {data_category_str}", ylabel="",
            plot_type="count",
            order=df_organs[organ_col].value_counts().index,
            title="What Organ / Body Part Are Most Commonly Captured?",
            legend=True,
            save_dir=save_dir,
            save_fname="organs(bar).png",
        )

        # 3. Task
        df_metadata[task_col] = df_metadata[task_col].str.split(", ")
        df_tasks = df_metadata.explode(task_col).reset_index(drop=True)
        df_tasks[task_col] = df_tasks[task_col].str.split(" ").str[-1]
        self.descriptions["Tasks"] = df_tasks.groupby(contains_children_col)[task_col].value_counts().round(2).to_dict()
        viz_data.catplot(
            df_tasks, y=task_col, hue=contains_children_col,
            xlabel=f"Number of {data_category_str}", ylabel="",
            plot_type="count",
            order=df_tasks[task_col].value_counts().index,
            title="What Are The Most Common Types of Tasks?",
            legend=True,
            save_dir=save_dir,
            save_fname="tasks(bar).png",
        )


    def parse_sample_size_column(self):
        """
        Parse sample size column to estimate the number of patients, sequences,
        and images in each dataset.
        """
        df_metadata = self.df_metadata.copy()
        modality_col = "Imaging Modality(ies)"

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
        curr_mask = missing_patient & (df_metadata[modality_col].str.contains("CT") | df_metadata[modality_col].str.contains("MRI"))
        df_metadata.loc[curr_mask, "num_patients"] = df_metadata.loc[curr_mask, "num_sequences"]
        # CASE 2: For Fundus, assume 2 images = 1 patient
        curr_mask = missing_patient & (df_metadata[modality_col].str.contains("Fundus"))
        df_metadata.loc[curr_mask, "num_patients"] = (df_metadata.loc[curr_mask, "num_images"] / 2).map(np.ceil)

        # Store numbers of sequences/images
        # NOTE: Number of patients is underestimated since datasets without patient count
        self.descriptions["Number of Sequences"] = int(df_metadata["num_sequences"].sum())
        self.descriptions["Number of Images"] = int(df_metadata["num_images"].sum())
        self.descriptions["Number of Patients"] = int(df_metadata["num_patients"].sum())

        # Update modified metadata table
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
        df_metadata = self.df_metadata.copy()
        data_category_str = self.data_category_str
        save_dir = self.save_dir

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

        self.descriptions["Number of Patients (With Age Mentioned)"] = int(df_metadata.loc[age_mentioned, "num_patients"].sum())

        # 2.2. Add column for Contains Children
        contains_children_col = "Contains Children"
        df_metadata[contains_children_col] = "Unknown"
        df_metadata.loc[mask_adult_only, contains_children_col] = "Adult Only"
        df_metadata.loc[mask_peds_only, contains_children_col] = "Peds Only"
        df_metadata.loc[mask_peds_and_adult, contains_children_col] = "Peds & Adult"

        # 2.3. Plot Age Mentioned
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
        self.descriptions["Prop. Children"] = round(num_children / total_num_patients, 4)
        self.descriptions["Num. Children"] = int(num_children)
        self.descriptions["Num. Adult"] = int(total_num_patients - num_children)
        df_child_count = pd.DataFrame({
            "is_child": (["Child"] * int(num_children)) + (["Adult"] * int(total_num_patients - num_children)),
        })

        # Update metadata table
        self.df_metadata = df_metadata


################################################################################
#                              Calling Functions                               #
################################################################################
def visualize_annotated_data():
    # ["challenges", "benchmarks", "datasets", "collections", "papers", "stanford_aimi", "tcia"]
    categories = ["challenges", "benchmarks", "datasets", "collections"]
    cat_to_descriptions = {}
    for category in categories:
        visualizer = OpenDataVisualizer(category)
        cat_to_descriptions[category] = visualizer.describe()

    print(json.dumps(cat_to_descriptions, indent=4))


def describe_collections():
    
    for collection in ["stanford_aimi", "tcia"]:
        visualizer = describe_data.OpenDataVisualizer("tcia")
        visualizer.describe()


################################################################################
#                               Helper Functions                               #
################################################################################
def parse_text_to_dict(text):
    """
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
    text = text.replace(" years", "")
    sep = " to " if " to " in text else "-"
    lower, upper = map(int, map(float, text.split(sep)))
    return lower, upper


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



################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
