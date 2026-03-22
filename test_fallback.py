import pandas as pd
from agents.profiler import profile_data
from agents.decision import run_agent
from agents.executor import execute_tool_call
from agents.validator import validate

# Load the actual test file
temp_path = 'uploads/current.csv'
df = pd.read_csv(temp_path)

print("=" * 70)
print("BEFORE CLEANING")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Total nulls: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"\nGender unique values: {df['Gender'].unique()}")
print(f"Age dtype: {df['Age'].dtype}")
print(f"Salary dtype: {df['Salary'].dtype}")
print(f"\nNull counts by column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

profile = profile_data(df)

print("\n" + "=" * 70)
print("CLEANING WITH FALLBACK (No API)")
print("=" * 70)

df_cleaned, logs, tool_calls = run_agent(
    profile=profile,
    df=df,
    execute_tool_fn=execute_tool_call,
    max_steps=30
)

print("\n" + "=" * 70)
print("AFTER CLEANING")
print("=" * 70)
print(f"Shape: {df_cleaned.shape}")
print(f"Total nulls: {df_cleaned.isnull().sum().sum()}")
print(f"Duplicates: {df_cleaned.duplicated().sum()}")
print(f"\nGender unique values: {df_cleaned['Gender'].unique()}")
print(f"Age dtype: {df_cleaned['Age'].dtype}")
print(f"Salary dtype: {df_cleaned['Salary'].dtype}")
print(f"\nNull counts by column:\n{df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0]}")

print("\n" + "=" * 70)
print("TOOL CALLS EXECUTED")
print("=" * 70)
for i, tc in enumerate(tool_calls, 1):
    print(f"{i}. {tc['action']} - {tc.get('reason', '')}")

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
is_clean, issues = validate(df_cleaned)
print(f"Status: {'✓ PASS' if is_clean else '✗ FAIL'}")
if issues:
    print(f"Remaining issues: {issues}")
