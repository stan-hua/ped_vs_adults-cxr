"""
openneuro_utils.py

Description: Helps extract institutional information from OpenNeuro datasets.
"""

# Standard libraries
import json
import re
import requests
import time
from typing import Dict, List, Optional

# Non-standard libraries
import pandas as pd


################################################################################
#                                   Classes                                    #
################################################################################
class OpenNeuroExtractor:
    """Extract information from OpenNeuro datasets via GitHub."""

    def __init__(self):
        self.base_github_url = "https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/master/README"
        self.dataset_description_url = "https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/master/dataset_description.json"

    def get_readme(self, dataset_id: str) -> Optional[str]:
        """Fetch README content from GitHub."""
        url = self.base_github_url.format(dataset_id=dataset_id)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"  README not found for {dataset_id}")
                return None
        except Exception as e:
            print(f"  Error fetching README for {dataset_id}: {e}")
            return None

    def get_dataset_description(self, dataset_id: str) -> Optional[Dict]:
        """Fetch dataset_description.json from GitHub."""
        url = self.dataset_description_url.format(dataset_id=dataset_id)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"  Error fetching dataset_description.json for {dataset_id}: {e}")
            return None

    def extract_institutions(self, readme_text: str) -> List[str]:
        """Extract institutional affiliations from README and dataset description."""
        institutions = []

        if not readme_text:
            return institutions

        # Look for author sections
        author_sections = re.findall(
            r'(?:Author[s]?|PI|Principal Investigator|Affiliation)[:\s]+([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\Z)',
            readme_text,
            re.IGNORECASE
        )

        # Common university/institution keywords
        institution_keywords = [
            r'University',
            r'Institut[e]?',
            r'College',
            r'Hospital',
            r'Medical Center',
            r'Medical School',
            r'School of',
            r'Department of',
        ]

        text_to_search = readme_text
        if author_sections:
            text_to_search = ' '.join(author_sections)

        # Extract sentences containing institutional keywords
        sentences = re.split(r'[.;]', text_to_search)
        for sentence in sentences:
            for keyword in institution_keywords:
                if re.search(keyword, sentence, re.IGNORECASE):
                    institutions.append(sentence.strip())
                    break

        return list(set(institutions))

    def process_dataset(self, dataset_id: str) -> Dict:
        """Process a single dataset and extract information."""
        print(f"Processing {dataset_id}...")

        readme = self.get_readme(dataset_id)
        description = self.get_dataset_description(dataset_id) or {}

        institutions = []

        if readme:
            institutions = self.extract_institutions(readme)

        # Get dataset name from description
        dataset_name = description.get("Name")
        # Get ethics approval if available
        ethics_approval = description.get("EthicsApprovals")
        if isinstance(ethics_approval, (tuple, list)):
            ethics_approval = "\n".join(ethics_approval)

        result = {
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'institutions': '\n'.join(institutions[:3]) if institutions else '',  # Limit to first 3
            'readme_available': readme is not None,
            "ethics_approval": ethics_approval,
        }

        return result

    def process_datasets(self, dataset_ids: List[str], delay: float = 1.0) -> pd.DataFrame:
        """Process multiple datasets with rate limiting."""
        results = []

        for i, dataset_id in enumerate(dataset_ids):
            result = self.process_dataset(dataset_id)
            results.append(result)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset_ids)} datasets")

            # Rate limiting
            time.sleep(delay)

        return pd.DataFrame(results)
