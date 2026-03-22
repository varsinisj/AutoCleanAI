# AutoClean AI 🧬

An Agentic AI Data Cleaning Pipeline that automatically 
cleans messy datasets in under 60 seconds.

## What it does
- Detects missing values, outliers, duplicates automatically
- AI agent calls cleaning tools one by one
- Sees result of each action and decides next step
- Gives downloadable clean CSV file

## Tech Stack
- Python, Flask, Pandas
- OpenRouter API (Free AI models)
- scikit-learn
- HTML, CSS, JavaScript

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Architecture
Profiler → Decision Agent (AI) → Executor → Validator → Clean CSV