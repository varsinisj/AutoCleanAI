import numpy as np

def profile_data(df):
    profile = {}
    profile['shape'] = df.shape
    profile['columns'] = list(df.columns)
    profile['missing'] = df.isnull().sum().to_dict()
    profile['missing_percent'] = (df.isnull().mean() * 100).round(2).to_dict()
    profile['types'] = df.dtypes.astype(str).to_dict()
    profile['duplicates'] = int(df.duplicated().sum())
    profile['sample_rows'] = df.head(5).fillna("NULL").to_dict(orient="records")

    outlier_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            outlier_cols.append({"column": col, "outlier_count": int(outliers)})
    profile['outliers'] = outlier_cols

    cat_info = {}
    for col in df.select_dtypes(include=['object']).columns:
        cat_info[col] = int(df[col].nunique())
    profile['categorical_cardinality'] = cat_info

    return profile
