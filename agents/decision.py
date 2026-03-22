import requests
import json
import time
import os
import numpy as np
import pandas as pd

# ── API CONFIGURATION ──
API_KEY = os.getenv("GROQ_API_KEY", "")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL   = "llama3-8b-8192"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


def build_prompt(profile, previous_issues=None):
    missing_cols = [col for col, v in profile.get("missing", {}).items() if v > 0]
    outlier_cols = [o["column"] for o in profile.get("outliers", [])]
    types        = profile.get("types", {})
    duplicates   = profile.get("duplicates", 0)
    cat_cols     = list(profile.get("categorical_cardinality", {}).keys())

    retry = ""
    if previous_issues:
        retry = "\n\nPrevious attempt left these issues: " + json.dumps(previous_issues) + "\nFix only these.\n"

    string_numeric_cols = []
    for col in ["Age", "Salary", "YearsExp", "Bonus", "Productivity", "Rating", "Score", "Income", "Price"]:
        if col in types and "object" in str(types.get(col, "")):
            string_numeric_cols.append(col)

    steps = []
    if duplicates > 0:
        steps.append("1. DEDUPLICATE " + str(duplicates) + " rows -> action: deduplicate, method: keep_first")
    if string_numeric_cols:
        steps.append("2. CAST TO FLOAT: " + str(string_numeric_cols) + " -> action: cast_type, method: float")
    if outlier_cols:
        steps.append("3. REMOVE OUTLIERS: " + str(outlier_cols) + " -> action: remove_outliers, method: iqr_clip")
    if missing_cols:
        steps.append("4. FILL MISSING: " + str(missing_cols) + " -> action: fill_missing, method: median(numeric) or mode(string)")
    for col in cat_cols:
        if col.lower() in ["gender", "sex"]:
            steps.append("5. STANDARDIZE GENDER column '" + col + "' -> action: standardize_gender, method: title_case")

    prompt = (
        "You are a data cleaning expert. Return ONLY a valid JSON array. No explanation. No markdown. Just JSON.\n\n"
        "Dataset profile:\n" + json.dumps({k: v for k, v in profile.items() if k != "sample_rows"}, indent=2) +
        "\n\nSample rows:\n" + json.dumps(profile.get("sample_rows", [])[:3], indent=2) +
        retry +
        "\n\nGenerate steps for ALL of these:\n" + "\n".join(steps) +
        """

Supported actions:
- deduplicate: method = "keep_first"
- cast_type: method = "float" or "int" or "datetime"
- remove_outliers: method = "iqr_clip"
- fill_missing: method = "median" (numeric) or "mode" (string/category)
- standardize_gender: method = "title_case"
- encode_categorical: method = "label"

Return ONLY JSON array:
[
  {"action": "deduplicate", "column": "__all__", "method": "keep_first", "reason": "duplicate rows"},
  {"action": "cast_type", "column": "Age", "method": "float", "reason": "stored as string"},
  {"action": "remove_outliers", "column": "Age", "method": "iqr_clip", "reason": "outliers detected"},
  {"action": "fill_missing", "column": "Age", "method": "median", "reason": "missing values"},
  {"action": "standardize_gender", "column": "Gender", "method": "title_case", "reason": "inconsistent values"}
]"""
    )
    return prompt


def generate_plan(profile, previous_issues=None):
    if DEMO_MODE:
        print("DEMO MODE: Using rule-based plan")
        return rule_based_plan(profile)

    prompt = build_prompt(profile, previous_issues)

    for attempt in range(3):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": "Bearer " + API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1500
                },
                timeout=30
            )
            result = response.json()

            if "choices" not in result:
                print("API error (attempt " + str(attempt+1) + "): " + str(result.get("error", result)))
                time.sleep(2)
                continue

            text = result["choices"][0]["message"]["content"].strip()
            print("AI response: " + text[:300])

            start = text.find("[")
            end   = text.rfind("]") + 1
            if start != -1 and end > start:
                text = text[start:end]

            plan = json.loads(text)
            print("Plan has " + str(len(plan)) + " steps")
            return plan

        except json.JSONDecodeError as e:
            print("JSON parse error (attempt " + str(attempt+1) + "): " + str(e))
            time.sleep(2)
        except Exception as e:
            print("Request error (attempt " + str(attempt+1) + "): " + str(e))
            time.sleep(2)

    print("AI failed — using rule-based plan")
    return rule_based_plan(profile)


def rule_based_plan(profile):
    """100% reliable — works without any API key."""
    plan       = []
    missing    = profile.get("missing", {})
    types      = profile.get("types", {})
    outliers   = profile.get("outliers", [])
    duplicates = profile.get("duplicates", 0)
    cat_cols   = profile.get("categorical_cardinality", {})

    if duplicates > 0:
        plan.append({"action": "deduplicate", "column": "__all__", "method": "keep_first",
                     "reason": str(duplicates) + " duplicates"})

    for col in ["Age", "Salary", "YearsExp", "Bonus", "Productivity", "Rating", "Score"]:
        if col in types and "object" in str(types.get(col, "")):
            plan.append({"action": "cast_type", "column": col, "method": "float",
                         "reason": col + " stored as string"})

    for o in outliers:
        plan.append({"action": "remove_outliers", "column": o["column"], "method": "iqr_clip",
                     "reason": str(o["outlier_count"]) + " outliers in " + o["column"]})

    for col, count in missing.items():
        if count == 0:
            continue
        col_type = str(types.get(col, "object"))
        method = "median" if ("int" in col_type or "float" in col_type) else "mode"
        plan.append({"action": "fill_missing", "column": col, "method": method,
                     "reason": str(count) + " missing in " + col})

    for col in cat_cols:
        if col.lower() in ["gender", "sex"]:
            plan.append({"action": "standardize_gender", "column": col, "method": "title_case",
                         "reason": "Inconsistent gender values"})

    return plan


def run_agent(profile, df, execute_tool_fn, max_steps=30):
    from agents.validator import validate
    from agents.profiler import profile_data

    all_logs        = []
    all_tool_calls  = []
    previous_issues = None

    for attempt in range(3):
        print("\nGenerating plan (attempt " + str(attempt+1) + ")...")

        plan = generate_plan(profile, previous_issues)

        if not plan:
            print("No plan generated.")
            break

        print("Executing " + str(len(plan)) + " steps:")
        for i, step in enumerate(plan):
            print("  " + str(i+1) + ". [" + step.get("action","") + "] "
                  + step.get("column","") + " -> " + step.get("method","")
                  + " | " + step.get("reason",""))
            df, msg = execute_tool_fn(df, step)
            all_logs.append(msg)
            all_tool_calls.append(step)
            print("     -> " + msg)

        is_clean, issues = validate(df)
        if is_clean:
            print("Validation PASSED!")
            break
        else:
            print("Still issues: " + str(issues))
            previous_issues = issues
            profile = profile_data(df)
            profile["shape"] = list(profile["shape"])

    return df, all_logs, all_tool_calls