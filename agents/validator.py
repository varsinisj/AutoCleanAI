def validate(df):
    issues = {}
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    if missing:
        issues['missing'] = missing
    dups = int(df.duplicated().sum())
    if dups > 0:
        issues['duplicates'] = dups
    return len(issues) == 0, issues
