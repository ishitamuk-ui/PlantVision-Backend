"""Dataset ingestion, validation, and sample data generation for DataSleuth."""

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from datasleuth.config import CONFIG


@dataclass
class IngestionResult:
    """Represents the outcome of a dataset loading and validation operation."""
    success: bool
    df: pd.DataFrame | None = None
    error_message: str | None = None
    warnings: list[str] | None = None
    metadata: dict[str, Any] | None = None


def format_bytes(num_bytes: int) -> str:
    """Format a byte count into a human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def get_dataset_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate high-level structural metadata for a DataFrame."""
    memory_bytes = int(df.memory_usage(deep=True).sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in bool_cols and c not in datetime_cols
    ]
    
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_bytes": memory_bytes,
        "memory_formatted": format_bytes(memory_bytes),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "bool_columns": bool_cols,
        "datetime_columns": datetime_cols,
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def validate_and_deduplicate_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Ensure column names are unique, stripped of whitespace, and valid.
    
    Returns:
        (sanitized_columns, list_of_renamed_warnings)
    """
    seen: dict[str, int] = {}
    sanitized: list[str] = []
    warnings: list[str] = []
    
    for idx, col in enumerate(columns):
        clean_name = str(col).strip()
        if not clean_name:
            clean_name = f"unnamed_column_{idx + 1}"
            warnings.append(f"Empty column header at position {idx + 1} renamed to '{clean_name}'.")
            
        if clean_name in seen:
            seen[clean_name] += 1
            new_name = f"{clean_name}_{seen[clean_name]}"
            warnings.append(f"Duplicate column '{clean_name}' renamed to '{new_name}'.")
            sanitized.append(new_name)
        else:
            seen[clean_name] = 0
            sanitized.append(clean_name)
            
    return sanitized, warnings


def load_csv_from_bytes(
    file_bytes: bytes,
    filename: str = "dataset.csv",
    max_size_mb: int = CONFIG.MAX_FILE_SIZE_MB,
) -> IngestionResult:
    """Parse, validate, and sanitize a CSV file from raw bytes.
    
    Performs rigorous checks on file size, encoding, structure, and column uniqueness.
    """
    # 1. Check extension
    if not filename.lower().endswith(".csv"):
        return IngestionResult(
            success=False,
            error_message=f"Unsupported file format: '{filename}'. Please upload a standard CSV (.csv) file.",
        )
        
    # 2. Check file size
    size_bytes = len(file_bytes)
    if size_bytes == 0:
        return IngestionResult(
            success=False,
            error_message="The uploaded CSV file is completely empty (0 bytes).",
        )
        
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return IngestionResult(
            success=False,
            error_message=f"File size ({format_bytes(size_bytes)}) exceeds the maximum allowed limit of {max_size_mb} MB.",
        )
        
    warnings: list[str] = []
    
    # 3. Handle Encodings & Delimiters
    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    decoded_str = None
    
    for enc in encodings_to_try:
        try:
            decoded_str = file_bytes.decode(enc)
            if enc != "utf-8":
                warnings.append(f"File was decoded using '{enc}' encoding.")
            break
        except UnicodeDecodeError:
            continue
            
    if decoded_str is None:
        return IngestionResult(
            success=False,
            error_message="Unable to decode file text. Please ensure the CSV is encoded in UTF-8 or Latin-1.",
        )
        
    # 4. Delimiter Sniffing / Parsing
    delimiters = [",", ";", "\t", "|"]
    df: pd.DataFrame | None = None
    parse_error = None
    
    for sep in delimiters:
        try:
            buffer = io.StringIO(decoded_str)
            test_df = pd.read_csv(buffer, sep=sep, nrows=10)
            if test_df.shape[1] > 1:
                # Good candidate delimiter found
                buffer = io.StringIO(decoded_str)
                df = pd.read_csv(buffer, sep=sep)
                if sep != ",":
                    warnings.append(f"Detected and used non-standard delimiter: '{sep}'.")
                break
        except Exception as e:
            parse_error = e
            continue
            
    if df is None:
        # Fallback to standard comma read
        try:
            buffer = io.StringIO(decoded_str)
            df = pd.read_csv(buffer, sep=",")
        except Exception as e:
            return IngestionResult(
                success=False,
                error_message=f"Failed to parse CSV: {str(e or parse_error)}. Please check for malformed rows or quotes.",
            )
            
    # 5. Sanity checks on parsed data
    if df.empty or len(df) == 0:
        return IngestionResult(
            success=False,
            error_message="The dataset contains 0 data rows.",
        )
        
    if len(df.columns) <= 1 and len(df) > 1:
        # Check if single column looks like an unresolved delimited string
        first_col_val = str(df.iloc[0, 0])
        if any(d in first_col_val for d in [";", "\t", "|"]):
            warnings.append("Dataset loaded with only 1 column; verify if an unusual delimiter was used.")

    # 6. Sanitize and deduplicate column names
    clean_cols, col_warnings = validate_and_deduplicate_columns(list(df.columns))
    df.columns = pd.Index(clean_cols)
    warnings.extend(col_warnings)
    
    metadata = get_dataset_metadata(df)
    
    return IngestionResult(
        success=True,
        df=df,
        warnings=warnings,
        metadata=metadata,
    )


def get_sample_datasets() -> dict[str, dict[str, Any]]:
    """Metadata catalog of available built-in benchmark datasets."""
    return {
        "heart_disease": {
            "name": "Heart Disease Clinical Risk",
            "task": "classification",
            "target": "heart_disease",
            "description": "Clinical health indicators for cardiovascular diagnosis (binary classification with mixed types).",
        },
        "housing_pricing": {
            "name": "California Real Estate Pricing",
            "task": "regression",
            "target": "median_house_value",
            "description": "Geographic and neighborhood real estate attributes (continuous regression with outliers and skewness).",
        },
        "customer_churn": {
            "name": "Telco Customer Churn",
            "task": "classification",
            "target": "churn",
            "description": "Subscription and usage attributes predicting customer cancellation (mixed types, imbalanced target).",
        },
    }


def load_sample_dataset(dataset_key: str) -> pd.DataFrame:
    """Generate realistic, genuine sample datasets with authentic statistical distributions."""
    rng = np.random.RandomState(42)
    n = 1000
    
    if dataset_key == "heart_disease":
        age = rng.normal(54, 9, n).clip(29, 78).round().astype(int)
        sex = rng.choice(["Male", "Female"], size=n, p=[0.68, 0.32])
        chest_pain = rng.choice(["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"], size=n, p=[0.48, 0.17, 0.28, 0.07])
        resting_bp = rng.normal(131, 17, n).clip(94, 200).round().astype(int)
        cholesterol = rng.normal(246, 51, n).clip(126, 564).round().astype(int)
        fasting_blood_sugar = rng.choice(["<= 120 mg/dl", "> 120 mg/dl"], size=n, p=[0.85, 0.15])
        rest_ecg = rng.choice(["normal", "st_t_wave_abnormality", "hypertrophy"], size=n, p=[0.50, 0.48, 0.02])
        max_heart_rate = (220 - age - rng.uniform(5, 45, n)).clip(71, 202).round().astype(int)
        exercise_angina = rng.choice(["No", "Yes"], size=n, p=[0.67, 0.33])
        oldpeak = rng.exponential(1.0, n).clip(0.0, 6.2).round(2)
        slope = rng.choice(["upsloping", "flat", "downsloping"], size=n, p=[0.47, 0.46, 0.07])
        major_vessels = rng.choice([0, 1, 2, 3], size=n, p=[0.58, 0.22, 0.13, 0.07])
        
        # Genuine risk equation for target
        logit = (
            -3.5
            + 0.03 * (age - 50)
            + 0.8 * (sex == "Male")
            + 0.9 * (chest_pain == "asymptomatic")
            + 0.01 * (resting_bp - 130)
            + 0.005 * (cholesterol - 240)
            - 0.02 * (max_heart_rate - 150)
            + 0.9 * (exercise_angina == "Yes")
            + 0.5 * oldpeak
            + 0.6 * major_vessels
        )
        prob = 1 / (1 + np.exp(-logit))
        heart_disease = (rng.uniform(0, 1, n) < prob).astype(int)
        
        df = pd.DataFrame({
            "patient_id": [f"PID-{1000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "chest_pain_type": chest_pain,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "rest_ecg": rest_ecg,
            "max_heart_rate": max_heart_rate,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak,
            "st_slope": slope,
            "major_vessels": major_vessels,
            "heart_disease": heart_disease,
        })
        
        # Inject realistic missing values in a few columns
        chol_mask = rng.choice([True, False], size=n, p=[0.04, 0.96])
        df.loc[chol_mask, "cholesterol"] = np.nan
        oldpeak_mask = rng.choice([True, False], size=n, p=[0.02, 0.98])
        df.loc[oldpeak_mask, "oldpeak"] = np.nan
        return df

    elif dataset_key == "housing_pricing":
        med_inc = rng.gamma(3.0, 1.3, n).clip(0.5, 15.0).round(2)
        house_age = rng.uniform(1, 52, n).round().astype(int)
        ave_rooms = (med_inc * 0.8 + rng.normal(3.0, 1.2, n)).clip(2.0, 14.0).round(2)
        ave_bedrms = (ave_rooms * 0.22 + rng.normal(0.2, 0.1, n)).clip(0.5, 4.0).round(2)
        population = rng.lognormal(6.8, 0.8, n).clip(100, 12000).round().astype(int)
        ave_occup = (rng.normal(3.0, 0.8, n)).clip(1.2, 8.0).round(2)
        latitude = rng.uniform(32.5, 42.0, n).round(2)
        longitude = rng.uniform(-124.3, -114.3, n).round(2)
        ocean_proximity = rng.choice(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], size=n, p=[0.44, 0.32, 0.13, 0.10, 0.01])
        
        # Realistic house price formula in $100k
        base_val = (
            0.5
            + 0.45 * med_inc
            + 0.015 * house_age
            - 0.05 * ave_rooms
            - 0.03 * ave_occup
            + (ocean_proximity == "NEAR BAY") * 0.6
            + (ocean_proximity == "NEAR OCEAN") * 0.5
            - (ocean_proximity == "INLAND") * 0.4
            + rng.normal(0, 0.35, n)
        )
        median_house_value = (base_val * 100000).clip(45000, 500000).round(-2)
        
        df = pd.DataFrame({
            "property_id": [f"PROP-{2000 + i}" for i in range(n)],
            "median_income": med_inc,
            "housing_median_age": house_age,
            "average_rooms": ave_rooms,
            "average_bedrooms": ave_bedrms,
            "population": population,
            "average_occupancy": ave_occup,
            "latitude": latitude,
            "longitude": longitude,
            "ocean_proximity": ocean_proximity,
            "median_house_value": median_house_value,
        })
        # Add realistic missing values
        bed_mask = rng.choice([True, False], size=n, p=[0.05, 0.95])
        df.loc[bed_mask, "average_bedrooms"] = np.nan
        return df

    elif dataset_key == "customer_churn":
        tenure = rng.randint(1, 72, n)
        contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
        internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
        tech_support = rng.choice(["Yes", "No", "No internet"], size=n, p=[0.29, 0.49, 0.22])
        monthly_charges = (
            20.0
            + (internet_service == "DSL") * 25.0
            + (internet_service == "Fiber optic") * 45.0
            + rng.uniform(0, 25, n)
        ).round(2)
        total_charges = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(20.0).round(2)
        senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
        payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
        
        # Churn probability calculation
        churn_logit = (
            -1.2
            - 0.04 * tenure
            + 0.8 * (contract == "Month-to-month")
            - 0.9 * (contract == "Two year")
            + 0.6 * (internet_service == "Fiber optic")
            - 0.5 * (tech_support == "Yes")
            + 0.015 * (monthly_charges - 60)
        )
        churn_prob = 1 / (1 + np.exp(-churn_logit))
        churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)
        
        df = pd.DataFrame({
            "customer_id": [f"CUST-{5000 + i}" for i in range(n)],
            "senior_citizen": senior_citizen,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        })
        # Add realistic missing values
        tot_mask = rng.choice([True, False], size=n, p=[0.03, 0.97])
        df.loc[tot_mask, "total_charges"] = np.nan
        return df

    else:
        raise ValueError(f"Unknown sample dataset key: {dataset_key}")
