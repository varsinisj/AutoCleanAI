import pandas as pd
import sys
from agents.profiler import profile_data
from agents.decision import run_agent
from agents.executor import execute_tool_call
from agents.validator import validate

# Create a simple test dataset with known issues
test_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': ['25', '30', 'forty', 35, ''],  # String ages, one invalid, one missing
    'Gender': ['Female', 'male', 'M', 'F', 'Female'],  # Inconsistent case/format
    'Salary': [50000, 60000, '9999999', 55000, 'N/A'],  # Invalid values and mix of types
}

df = pd.DataFrame(test_data)

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)
print(df)
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values: {df.isnull().sum().to_dict()}")
print(f"\nGender unique values: {df['Gender'].unique()}")
print(f"Age unique values: {df['Age'].unique()}")
print(f"Salary unique values: {df['Salary'].unique()}")

print("\n" + "=" * 60)
print("RUNNING CLEANING PIPELINE")
print("=" * 60)

profile = profile_data(df)
print(f"\nProfile: {profile}")

df_cleaned, logs, tool_calls = run_agent(
    profile=profile,
    df=df,
    execute_tool_fn=execute_tool_call,
    max_steps=30
)

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)
print(df_cleaned)
print(f"\nDtypes:\n{df_cleaned.dtypes}")
print(f"\nMissing values: {df_cleaned.isnull().sum().to_dict()}")
print(f"\nGender unique values: {df_cleaned['Gender'].unique()}")
print(f"Age unique values: {df_cleaned['Age'].unique()}")
print(f"Salary unique values: {df_cleaned['Salary'].unique()}")

print("\n" + "=" * 60)
print("TOOL CALLS EXECUTED")
print("=" * 60)
for i, tc in enumerate(tool_calls, 1):
    print(f"{i}. {tc['action']} - {tc.get('reason', '')}")

print("\n" + "=" * 60)
print("LOGS")
print("=" * 60)
for log in logs:
    print(f"  - {log}")

is_clean, issues = validate(df_cleaned)
print("\n" + "=" * 60)
print(f"VALIDATION: {'PASS' if is_clean else 'FAIL'}")
print("=" * 60)
if issues:
    print(f"Remaining issues: {issues}")
else:
    print("✓ Dataset is clean!")
