import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def tool_fill_missing(df, column, method):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    before = int(df[column].isnull().sum())
    if before == 0:
        return df, f"SKIP: No missing values in '{column}'."
    if method == "median" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    elif method == "mean" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mean())
    elif method == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])
    elif method == "ffill":
        df[column] = df[column].ffill()
    elif method == "bfill":
        df[column] = df[column].bfill()
    elif method.startswith("constant:"):
        value = method.split(":", 1)[1]
        try:
            value = float(value) if '.' in value else int(value)
        except ValueError:
            pass
        df[column] = df[column].fillna(value)
    else:
        return df, f"SKIP: Unknown method '{method}'."
    after = int(df[column].isnull().sum())
    return df, f"Filled {before - after} missing values in '{column}' using {method}."


def tool_drop_column(df, column):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    df = df.drop(columns=[column])
    return df, f"Dropped column '{column}'."


def tool_remove_outliers(df, column, method):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, f"SKIP: Column '{column}' is not numeric."
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if method == "iqr_clip":
        df[column] = df[column].clip(lower=lower, upper=upper)
        return df, f"Clipped outliers in '{column}' to [{lower:.2f}, {upper:.2f}]."
    elif method == "iqr_drop":
        before = len(df)
        df = df[(df[column] >= lower) & (df[column] <= upper)]
        return df, f"Dropped {before - len(df)} outlier rows in '{column}'."
    return df, f"SKIP: Unknown method '{method}'."


def tool_cast_type(df, column, method):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    if method == "int":
        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
    elif method == "float":
        df[column] = pd.to_numeric(df[column], errors='coerce')
    elif method == "str":
        df[column] = df[column].astype(str)
    elif method == "datetime":
        df[column] = pd.to_datetime(df[column], errors='coerce')
    else:
        return df, f"SKIP: Unknown type '{method}'."
    return df, f"Cast '{column}' to {method}."

def tool_standardize_gender(df, column):
    if column not in df.columns:
        return df, "SKIP: Column '" + column + "' not found."
    mapping = {
        'male': 'Male', 'm': 'Male', 'M': 'Male',
        'female': 'Female', 'f': 'Female', 'F': 'Female',
        'Male': 'Male', 'Female': 'Female'
    }
    before = df[column].nunique()
    df[column] = df[column].map(lambda x: mapping.get(str(x).strip(), x) if pd.notna(x) else x)
    after = df[column].nunique()
    return df, "Standardized Gender: " + str(before) + " variants -> " + str(after) + " variants (Male/Female)."
def tool_deduplicate(df, method):
    before = len(df)
    keep = "first" if method == "keep_first" else "last"
    df = df.drop_duplicates(keep=keep)
    return df, f"Removed {before - len(df)} duplicate rows."


def tool_encode_categorical(df, column, method):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    if method == "label":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        return df, f"Label encoded '{column}'."
    elif method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
        return df, f"One-hot encoded '{column}' into {dummies.shape[1]} columns."
    return df, f"SKIP: Unknown method '{method}'."


def tool_standardize_categorical(df, column, mapping):
    if column not in df.columns:
        return df, f"SKIP: Column '{column}' not found."
    
    # Apply the mapping to standardize values
    original_unique_count = df[column].nunique(dropna=False)
    
    # Case-insensitive mapping: create mapping for all case variations
    case_insensitive_mapping = {}
    for old_val, new_val in mapping.items():
        old_lower = str(old_val).lower()
        # Find all unique values that match (case-insensitive)
        if df[column].dtype == 'object' or df[column].dtype == 'string':
            for actual_val in df[column].unique():
                if pd.notna(actual_val) and str(actual_val).lower() == old_lower:
                    case_insensitive_mapping[actual_val] = new_val
    
    # Apply mapping
    df[column] = df[column].map(case_insensitive_mapping).fillna(df[column])
    new_unique_count = df[column].nunique(dropna=False)
    
    return df, f"Standardized '{column}': reduced from {original_unique_count} values to {new_unique_count} values using mapping {mapping}."


TOOL_MAP = {
    "fill_missing":            lambda df, step: tool_fill_missing(df, step["column"], step["method"]),
    "drop_column":             lambda df, step: tool_drop_column(df, step["column"]),
    "remove_outliers":         lambda df, step: tool_remove_outliers(df, step["column"], step["method"]),
    "cast_type":               lambda df, step: tool_cast_type(df, step["column"], step["method"]),
    "deduplicate":             lambda df, step: tool_deduplicate(df, step["method"]),
    "standardize_categorical": lambda df, step: tool_standardize_categorical(df, step["column"], step.get("mapping", {})),
    "encode_categorical":      lambda df, step: tool_encode_categorical(df, step["column"], step["method"]),
    "standardize_gender": lambda df, step: tool_standardize_gender(df, step["column"]),
}


def execute_tool_call(df, tool_call):
    action = tool_call.get("action")
    column = tool_call.get("column", "__all__")
    reason = tool_call.get("reason", "")

    print(f"  TOOL -> [{action}] column='{column}' method='{tool_call.get('method', '')}'")
    if action == "standardize_categorical":
        print(f"  Mapping: {tool_call.get('mapping', {})}")
    print(f"  Reason: {reason}")

    if action not in TOOL_MAP:
        msg = f"ERROR: Unknown tool '{action}'"
        print(f"  {msg}")
        return df, msg

    try:
        df, result = TOOL_MAP[action](df, tool_call)
        print(f"  Result: {result}")
        return df, result
    except Exception as e:
        msg = f"ERROR in '{action}' on '{column}': {e}"
        print(f"  {msg}")
        return df, msg
