from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os

from agents.profiler import profile_data
from agents.decision import run_agent
from agents.executor import execute_tool_call
from agents.validator import validate

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Only CSV and Excel files supported'}), 400

        profile = profile_data(df)
        profile['shape'] = list(profile['shape'])
        profile['sample_rows'] = df.head(10).fillna('NULL').to_dict(orient='records')

        temp_path = os.path.join(UPLOAD_FOLDER, 'current.csv')
        df.to_csv(temp_path, index=False)

        return jsonify({
            'success': True,
            'profile': profile,
            'preview': df.head(10).fillna('').to_dict(orient='records'),
            'columns': list(df.columns),
            'shape': list(df.shape)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clean', methods=['POST'])
def clean():
    temp_path = os.path.join(UPLOAD_FOLDER, 'current.csv')
    if not os.path.exists(temp_path):
        return jsonify({'error': 'No dataset uploaded'}), 400

    try:
        df = pd.read_csv(temp_path)
        all_logs = []
        all_tool_calls = []
        MAX_RETRIES = 3

        for attempt in range(1, MAX_RETRIES + 1):
            profile = profile_data(df)
            profile['shape'] = list(profile['shape'])

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
                break

        cleaned_path = os.path.join(UPLOAD_FOLDER, 'cleaned.csv')
        df.to_csv(cleaned_path, index=False)

        final_profile = profile_data(df)
        final_profile['shape'] = list(final_profile['shape'])

        return jsonify({
            'success': True,
            'logs': all_logs,
            'tool_calls': all_tool_calls,
            'preview': df.head(10).fillna('').to_dict(orient='records'),
            'columns': list(df.columns),
            'shape': list(df.shape),
            'final_profile': final_profile,
            'remaining_nulls': int(df.isnull().sum().sum()),
            'remaining_duplicates': int(df.duplicated().sum())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download')
def download():
    cleaned_path = os.path.join(UPLOAD_FOLDER, 'cleaned.csv')
    if not os.path.exists(cleaned_path):
        return jsonify({'error': 'No cleaned file available'}), 400
    return send_file(cleaned_path, as_attachment=True, download_name='cleaned_output.csv')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
