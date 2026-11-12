"""
utils/disease_info.py
---------------------
Loads and queries the disease_info.csv reference table.

The CSV maps every class label to:
  - common disease name
  - affected plant species
  - short description
  - recommended treatment / management steps

This module is imported by app.py and predict.py to enrich predictions
with actionable agronomic information.
"""

import os
from typing import Optional

import pandas as pd

from config import DISEASE_INFO_CSV
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
#  CSV Schema (column names)
# ─────────────────────────────────────────────
COL_CLASS_NAME   = "class_name"
COL_DISEASE_NAME = "disease_name"
COL_PLANT        = "plant_species"
COL_DESCRIPTION  = "description"
COL_TREATMENT    = "treatment"
COL_SEVERITY     = "severity"       # low | medium | high
COL_IS_HEALTHY   = "is_healthy"     # True | False

REQUIRED_COLUMNS = [COL_CLASS_NAME, COL_DISEASE_NAME, COL_PLANT, COL_DESCRIPTION, COL_TREATMENT]


# ─────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────

def load_disease_info(csv_path: str = DISEASE_INFO_CSV) -> pd.DataFrame:
    """
    Load the disease information CSV into a DataFrame.

    The CSV is expected to live at data/disease_info.csv.
    If the file is missing, a warning is logged and an empty
    DataFrame with the correct schema is returned so that the
    rest of the app can degrade gracefully.

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(csv_path):
        logger.warning(
            "Disease info CSV not found at '%s'. "
            "Returning empty DataFrame — populate data/disease_info.csv "
            "to enable disease descriptions.",
            csv_path,
        )
        return _empty_dataframe()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read '%s': %s", csv_path, exc)
        return _empty_dataframe()

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "Disease info CSV is missing columns: %s. "
            "Some features may be unavailable.",
            missing,
        )

    # Normalise class_name to lower-case stripped string for robust lookup
    df[COL_CLASS_NAME] = df[COL_CLASS_NAME].astype(str).str.strip().str.lower()

    logger.info("Disease info loaded: %d entries from '%s'.", len(df), csv_path)
    return df


# ─────────────────────────────────────────────
#  Query Helpers
# ─────────────────────────────────────────────

def get_disease_details(
    class_name: str,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Retrieve all disease details for a predicted class label.

    Parameters
    ----------
    class_name : str
        The raw class name as returned by the model (e.g.
        'Tomato___Early_blight').
    df : pd.DataFrame | None
        Pre-loaded DataFrame; loads from CSV if None.

    Returns
    -------
    dict
        Keys: disease_name, plant_species, description, treatment,
              severity, is_healthy.
        Returns a placeholder dict if the class is not found.
    """
    if df is None:
        df = load_disease_info()

    normalised = class_name.strip().lower()
    matches = df[df[COL_CLASS_NAME] == normalised]

    if matches.empty:
        logger.warning("No disease info found for class '%s'.", class_name)
        return _unknown_disease(class_name)

    row = matches.iloc[0]
    return {
        "disease_name":  row.get(COL_DISEASE_NAME, "Unknown"),
        "plant_species": row.get(COL_PLANT, "Unknown"),
        "description":   row.get(COL_DESCRIPTION, "No description available."),
        "treatment":     row.get(COL_TREATMENT, "Consult a local agronomist."),
        "severity":      row.get(COL_SEVERITY, "Unknown"),
        "is_healthy":    bool(row.get(COL_IS_HEALTHY, False)),
    }


def get_all_class_names(df: Optional[pd.DataFrame] = None) -> list:
    """
    Return every class name present in the CSV.

    Parameters
    ----------
    df : pd.DataFrame | None

    Returns
    -------
    list[str]
    """
    if df is None:
        df = load_disease_info()
    if df.empty:
        return []
    return df[COL_CLASS_NAME].tolist()


def filter_by_plant(
    plant: str,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return rows that belong to a specific plant species.

    Parameters
    ----------
    plant : str
        Case-insensitive species name (e.g. "tomato").
    df : pd.DataFrame | None

    Returns
    -------
    pd.DataFrame
    """
    if df is None:
        df = load_disease_info()
    return df[df[COL_PLANT].str.lower().str.contains(plant.lower(), na=False)]


def filter_by_severity(
    level: str,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return rows with a specific severity level.

    Parameters
    ----------
    level : str
        "low", "medium", or "high".
    df : pd.DataFrame | None

    Returns
    -------
    pd.DataFrame
    """
    if df is None:
        df = load_disease_info()
    return df[df[COL_SEVERITY].str.lower() == level.lower()]


# ─────────────────────────────────────────────
#  Private Helpers
# ─────────────────────────────────────────────

def _empty_dataframe() -> pd.DataFrame:
    """Return a schema-correct but empty DataFrame."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS + [COL_SEVERITY, COL_IS_HEALTHY])


def _unknown_disease(class_name: str) -> dict:
    """Placeholder dict returned when no match is found in the CSV."""
    return {
        "disease_name":  class_name.replace("_", " ").replace("___", " — "),
        "plant_species": "Unknown",
        "description":   "No detailed information available for this class.",
        "treatment":     "Please consult a certified plant pathologist or agronomist.",
        "severity":      "Unknown",
        "is_healthy":    "healthy" in class_name.lower(),
    }

