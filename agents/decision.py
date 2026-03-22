import requests
import json
import time
import os
import numpy as np
import pandas as pd

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
MODEL = "openrouter/auto"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fill_missing",
            "description": "Fill missing/null values in a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {"type": "string", "enum": ["median", "mean", "mode", "ffill", "bfill"]},
                    "reason": {"type": "string"}
                },
                "required": ["column", "method", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop_column",
            "description": "Drop a column that is more than 60% missing or irrelevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["column", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_outliers",
            "description": "Handle outliers in a numeric column using IQR method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {"type": "string", "enum": ["iqr_clip", "iqr_drop"]},
                    "reason": {"type": "string"}
                },
                "required": ["column", "method", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cast_type",
            "description": "Convert a column to the correct data type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {"type": "string", "enum": ["int", "float", "str", "datetime"]},
                    "reason": {"type": "string"}
                },
                "required": ["column", "method", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deduplicate",
            "description": "Remove duplicate rows. Always do this first if duplicates exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["keep_first", "keep_last"]},
                    "reason": {"type": "string"}
                },
                "required": ["method", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_categorical",
            "description": "Standardize inconsistent categorical values (e.g., female/Female/F to Female, M/Male/male to Male).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "mapping": {"type": "object", "description": "JSON mapping of old values to new standardized values"},
                    "reason": {"type": "string"}
                },
                "required": ["column", "mapping", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "encode_categorical",
            "description": "Encode a categorical string column into numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "method": {"type": "string", "enum": ["label", "onehot"]},
                    "reason": {"type": "string"}
                },
                "required": ["column", "method", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call ONLY when ALL issues are fixed: zero nulls, zero duplicates, zero outliers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"}
                },
                "required": ["summary"]
            }
        }
    }
]


def get_remaining_issues(df):
    """Return list of all remaining issues: nulls, duplicates, data type problems, outliers, categorical inconsistencies."""
    issues = []

    # Missing values
    for col, count in df.isnull().sum().items():
        if count > 0:
            dtype = str(df[col].dtype)
            issues.append("  - '" + col + "': " + str(count) + " missing values (dtype: " + dtype + ")")

    # Duplicates
    dups = int(df.duplicated().sum())
    if dups > 0:
        issues.append("  - " + str(dups) + " duplicate rows")

    # Data type problems: string columns that should be numeric
    for col in df.select_dtypes(include=['object']).columns:
        if col.lower() in ['age', 'salary', 'price', 'income', 'amount', 'count', 'quantity', 'value']:
            # Check if this looks like it should be numeric
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Try to convert and check for invalid values
                numeric_count = 0
                for val in non_null:
                    try:
                        float(str(val))
                        numeric_count += 1
                    except ValueError:
                        pass
                invalid_count = len(non_null) - numeric_count
                if invalid_count > 0:
                    issues.append("  - '" + col + "': " + str(invalid_count) + " non-numeric values like '" + 
                                str(non_null[non_null.astype(str).str.lower().str.contains('forty|n/a|not_disclosed|unknown', na=False)].iloc[0] if 
                                (non_null.astype(str).str.lower().str.contains('forty|n/a|not_disclosed|unknown', na=False)).any() else 'N/A') + 
                                "' — MUST cast_type to float first")

    # Data type check: columns that are numeric should actually be numeric
    for col in df.columns:
        dtype = str(df[col].dtype)
        if col.lower() in ['age', 'salary', 'price', 'income', 'amount', 'count'] and dtype == 'object':
            issues.append("  - '" + col + "': dtype is string but should be numeric — call cast_type('" + col + "', 'float')")

    # Categorical inconsistency: check for case/encoding inconsistencies
    for col in df.select_dtypes(include=['object']).columns:
        if col.lower() in ['gender', 'sex', 'category', 'status', 'type', 'group']:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                # Check for mixed case or inconsistent representations
                lower_vals = [str(v).lower() for v in unique_vals]
                lower_unique = set(lower_vals)
                if len(lower_unique) < len(unique_vals):
                    issues.append("  - '" + col + "': " + str(len(unique_vals)) + " inconsistent values " + str(list(unique_vals)) + 
                                " — standardize to consistent casing/format")

    # Outliers in numeric columns (only check columns that are already numeric)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if outlier_count > 0:
            issues.append("  - '" + col + "': " + str(outlier_count) + " outliers detected (use remove_outliers iqr_clip)")

    return issues


def run_agent(profile, df, execute_tool_fn, max_steps=30):

    system_prompt = (
        "You are an autonomous data cleaning agent. Fix ALL issues in the CORRECT ORDER.\n\n"
        "STRICT EXECUTION ORDER (DO NOT DEVIATE):\n"
        "1. deduplicate() - remove duplicate rows FIRST\n"
        "2. cast_type() - convert string columns to numeric (for Age, Salary, etc.) — THIS MUST HAPPEN BEFORE outlier detection!\n"
        "3. fill_missing() - fill null values with median (numeric) or mode (string)\n"
        "4. standardize categorical columns (Gender, etc.) - standardize inconsistent values to single format\n"
        "5. remove_outliers() - only after cast_type() has made columns numeric!\n"
        "6. encode_categorical() - label encode categorical columns if needed\n"
        "7. done() - ONLY when issue list shows NONE remaining\n\n"
        "CRITICAL RULES:\n"
        "- Age, Salary must be cast to float BEFORE you attempt remove_outliers\n"
        "- String columns with non-numeric values must use cast_type with errors='coerce' to convert\n"
        "- Standardize Gender/categorical columns to consistent format (Female/Male, not M/F/male/female)\n"
        "- Do NOT call done() unless the issue list shows NONE remaining\n"
        "- For every action, provide specific reason why\n"
        "- You MUST fix every single issue before calling done()"
    )

    profile_clean = {k: v for k, v in profile.items() if k != "sample_rows"}
    sample_str = json.dumps(profile.get("sample_rows", [])[:3], indent=2)
    remaining = get_remaining_issues(df)
    remaining_str = "\n".join(remaining) if remaining else "  None"

    initial_msg = (
        "Dataset profile:\n" + json.dumps(profile_clean, indent=2) +
        "\n\nSample rows:\n" + sample_str +
        "\n\nIssues to fix:\n" + remaining_str +
        "\n\nTotal nulls: " + str(df.isnull().sum().sum()) +
        " | Duplicates: " + str(int(df.duplicated().sum())) +
        "\n\nBegin cleaning. Fix every issue above."
    )

    messages = [{"role": "user", "content": initial_msg}]
    tool_calls_made = []
    logs = []

    print("\nAI Agent starting. Issues: " + str(len(remaining)))

    for step in range(max_steps):
        total_nulls = int(df.isnull().sum().sum())
        total_dups = int(df.duplicated().sum())
        remaining_now = get_remaining_issues(df)

        print("\n--- Step " + str(step + 1) + "/" + str(max_steps) +
              " | Nulls: " + str(total_nulls) +
              " | Dups: " + str(total_dups) +
              " | Issues: " + str(len(remaining_now)) + " ---")

        if len(remaining_now) == 0:
            print("All issues resolved. Stopping.")
            logs.append("Dataset fully cleaned.")
            break

        # Call OpenRouter API
        result = None
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer " + OPENROUTER_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": MODEL,
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                        "tools": TOOL_SCHEMAS,
                        "temperature": 0.0,
                        "max_tokens": 400
                    },
                    timeout=30
                )
                result = response.json()
                if "choices" in result:
                    break
                print("API error (attempt " + str(attempt + 1) + "): " + str(result.get("error", result)))
                time.sleep(3)
            except Exception as e:
                print("Request error (attempt " + str(attempt + 1) + "): " + str(e))
                time.sleep(3)

        if not result or "choices" not in result:
            print("API failed after 3 attempts.")
            break

        message = result["choices"][0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls if tool_calls else None
        })

        if not tool_calls:
            # Nudge AI back to using tools
            null_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
            print("No tool call. Nudging AI...")
            nudge = "You must call a tool now. Remaining issues:\n" + "\n".join(remaining_now)
            if null_cols:
                method = "median" if pd.api.types.is_numeric_dtype(df[null_cols[0]]) else "mode"
                nudge += "\nCall fill_missing on '" + null_cols[0] + "' with method '" + method + "' right now."
            messages.append({"role": "user", "content": nudge})
            continue

        # Process each tool call
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            tc_id = tc.get("id", "call_" + str(step))
            try:
                fn_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except Exception:
                fn_args = {}

            print("AI calls: " + fn_name + "(" + str(fn_args) + ")")

            # Block done() if issues remain
            if fn_name == "done":
                issues_left = get_remaining_issues(df)
                if issues_left:
                    print("Blocked done() - issues remain: " + str(len(issues_left)))
                    rejection = "REJECTED. These issues still remain - fix them all before calling done():\n"
                    rejection += "\n".join(issues_left)
                    rejection += "\nContinue cleaning."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": rejection
                    })
                    continue
                else:
                    summary = fn_args.get("summary", "Cleaning complete.")
                    logs.append("Done: " + summary)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": "Complete."})
                    return df, logs, tool_calls_made

            # Execute the tool
            step_dict = {
                "action": fn_name,
                "column": fn_args.get("column", "__all__"),
                "method": fn_args.get("method", ""),
                "mapping": fn_args.get("mapping", {}),
                "reason": fn_args.get("reason", "")
            }
            df, result_msg = execute_tool_fn(df, step_dict)
            logs.append(result_msg)
            tool_calls_made.append(step_dict)

            # Feed result + updated issue list back to AI
            issues_after = get_remaining_issues(df)
            issues_str = "\n".join(issues_after) if issues_after else "  NONE - all clean!"
            next_action = "Call done() now." if not issues_after else "Continue - fix the next issue."

            feedback = (
                result_msg +
                "\n\nREMAINING ISSUES:\n" + issues_str +
                "\n\n" + next_action
            )
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": feedback})

    print("Agent finished. Final nulls: " + str(df.isnull().sum().sum()))
    return df, logs, tool_calls_made
