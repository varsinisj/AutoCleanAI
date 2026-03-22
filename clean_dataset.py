from agents.profiler import profile_data
from agents.decision import run_agent
from agents.executor import execute_tool_call
from agents.validator import validate

MAX_RETRIES = 3

def clean_dataset(df):
    print("=" * 60)
    print("FULLY AGENTIC DATA CLEANING PIPELINE")
    print("=" * 60)

    all_logs = []
    all_tool_calls = []

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAGENT RUN {attempt}/{MAX_RETRIES}")
        print("-" * 40)

        profile = profile_data(df)
        missing_now = {k: v for k, v in profile['missing'].items() if v > 0}
        print(f"Shape: {profile['shape']} | Missing: {missing_now} | Dups: {profile['duplicates']}")

        df, logs, tool_calls = run_agent(
            profile=profile,
            df=df,
            execute_tool_fn=execute_tool_call,
            max_steps=30
        )

        all_logs.extend(logs)
        all_tool_calls.extend(tool_calls)

        is_clean, issues = validate(df)

        if is_clean:
            print(f"\nValidation PASSED after run {attempt}!")
            break
        else:
            print(f"\nValidation failed after run {attempt}: {issues}")
            if attempt == MAX_RETRIES:
                print("Max retries reached.")

    print("\n" + "=" * 60)
    print("CLEANING REPORT")
    print("=" * 60)
    print(f"Total tool calls: {len(all_tool_calls)}")
    for i, tc in enumerate(all_tool_calls, 1):
        print(f"  {i}. [{tc['action']}] {tc.get('column','')} -> {tc.get('method','')}")
        print(f"     {tc.get('reason','')}")
    print(f"\nFinal shape: {df.shape}")
    print(f"Remaining nulls: {df.isnull().sum().sum()}")
    print(f"Remaining duplicates: {df.duplicated().sum()}")

    return df
