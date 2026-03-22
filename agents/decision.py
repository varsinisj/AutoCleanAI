import requests
import json
import time
import os
import numpy as np
import pandas as pd

OPENROUTER_API_KEY = os.getenv("GROQ_API_KEY", "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")
MODEL = "openrouter/auto"

# Print warning if API key is missing
if not OPENROUTER_API_KEY:
    print("\n⚠️  WARNING: OPENROUTER_API_KEY environment variable not set!")
    print("Please set your valid API key before running cleaning:")
    print("  Windows:   set OPENROUTER_API_KEY=your-key")
    print("  Linux/Mac: export OPENROUTER_API_KEY=your-key\n")


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
    """Return list of all remaining issues: nulls, duplicates, outliers."""
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

    # Outliers in numeric columns
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
        "You are an autonomous data cleaning agent. Fix ALL issues: missing values, duplicates, AND outliers.\n\n"
        "STRICT RULES:\n"
        "1. Call tools ONE AT A TIME\n"
        "2. Fix in this ORDER: deduplicate first, then remove_outliers, then fill_missing, then done\n"
        "3. Do NOT call done() unless the issue list shows NONE remaining\n"
        "4. For outliers: ALWAYS use remove_outliers with iqr_clip\n"
        "5. For missing numeric columns: use fill_missing with median\n"
        "6. For missing string/categorical columns: use fill_missing with mode\n"
        "7. You MUST fix every single issue before calling done()"
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
