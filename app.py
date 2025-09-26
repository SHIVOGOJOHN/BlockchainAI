from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import requests
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from github import Github, Auth
import io
from dotenv import load_dotenv
from pathlib import Path
from flask_caching import Cache

load_dotenv() # Load environment variables from .env file
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

# Configure Cache
app.config['CACHE_TYPE'] = "SimpleCache"  # Can be 'redis', 'memcached', etc. for production
app.config['CACHE_DEFAULT_TIMEOUT'] = 60 # Cache timeout in seconds
cache = Cache(
    app)

#LEDGER_PATH = Path('/data/ledger.json') # Path relative to flask_ui_app/app.py
LEDGER_PATH = Path(__file__).parent / 'ledger.json'

def load_ledger_data():
    if not LEDGER_PATH.exists():
        return []
    try:
        with open(LEDGER_PATH, 'r') as f:
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

FASTAPI_PREDICT_URL = "https://federatedsys.onrender.com/fed_sys"

# Define default values for the input form
DEFAULT_INPUT_VALUES = {
    "Age": 62,
    "Income": 55000.0,
    "Loyalty_Score": 98,
    "Prior_Purchases": 12,
    "Avg_Spend": 20.75,
    "Recency": 15,
    "Browsing_Time": 5.3,
    "Clicks_On_Promo": 3,
    "Purchase_Frequency": 1.8,
    "High_Value_Customer": 1,
    "Engagement_Score": 5,
    "Gender_Male": 1,
    "Promotion_Type_Discount": 1,
    "Promotion_Type_FlashSale": 0,
    "Promotion_Type_LoyaltyPoints": 0,
    "Channel_Email": 1,
    "Channel_In_store": 1,
    "Channel_SMS": 0,
    "Time_of_Day_Evening": 1,
    "Time_of_Day_Morning": 0
}

# Define the expected order of features for the FastAPI model

EXPECTED_FEATURES = [
    "Age", "Income", "Loyalty_Score", "Prior_Purchases", "Avg_Spend",
    "Recency", "Browsing_Time", "Clicks_On_Promo", "Purchase_Frequency",
    "High_Value_Customer", "Engagement_Score", "Gender_Male",
    "Promotion_Type_Discount", "Promotion_Type_FlashSale",
    "Promotion_Type_LoyaltyPoints", "Channel_Email", "Channel_In_store",
    "Channel_SMS", "Time_of_Day_Evening", "Time_of_Day_Morning"
]

FEATURE_TYPES = {
    "Age": int,
    "Income": float,
    "Loyalty_Score": int,
    "Prior_Purchases": int,
    "Avg_Spend": float,
    "Recency": int,
    "Browsing_Time": float,
    "Clicks_On_Promo": int,
    "Purchase_Frequency": float,
    "High_Value_Customer": int,
    "Engagement_Score": int,
    "Gender_Male": int,
    "Promotion_Type_Discount": int,
    "Promotion_Type_FlashSale": int,
    "Promotion_Type_LoyaltyPoints": int,
    "Channel_Email": int,
    "Channel_In_store": int,
    "Channel_SMS": int,
    "Time_of_Day_Evening": int,
    "Time_of_Day_Morning": int,
}


FEATURE_DESCRIPTIONS = {
    "Age": "Age of the customer in years.",
    "Income": "Annual income of the customer.",
    "Loyalty_Score": "A score indicating customer loyalty (1-100).",
    "Prior_Purchases": "Number of purchases made by the customer previously.",
    "Avg_Spend": "Average amount spent per purchase by the customer.",
    "Recency": "Number of days since the customer's last purchase.",
    "Browsing_Time": "Average time (in minutes) the customer spends browsing.",
    "Clicks_On_Promo": "Number of times the customer clicked on a promotion.",
    "Purchase_Frequency": "How often the customer makes purchases (Prior Purchases / Age).",
    "High_Value_Customer": "Binary indicator: 1 if customer's average spend is in the top 25%, 0 otherwise.",
    "Engagement_Score": "A calculated score reflecting customer engagement with promotions and browsing.",
    "Gender_Male": "Binary indicator: 1 if customer is Male, 0 if Female.",
    "Promotion_Type_Discount": "Binary indicator: 1 if the promotion type is 'Discount', 0 otherwise.",
    "Promotion_Type_FlashSale": "Binary indicator: 1 if the promotion type is 'FlashSale', 0 otherwise.",
    "Promotion_Type_LoyaltyPoints": "Binary indicator: 1 if the promotion type is 'LoyaltyPoints', 0 otherwise.",
    "Channel_Email": "Binary indicator: 1 if the promotion channel is 'Email', 0 otherwise.",
    "Channel_In_store": "Binary indicator: 1 if the promotion channel is 'In-store', 0 otherwise.",
    "Channel_SMS": "Binary indicator: 1 if the promotion channel is 'SMS', 0 otherwise.",
    "Time_of_Day_Evening": "Binary indicator: 1 if the time of day for promotion is 'Evening', 0 otherwise.",
    "Time_of_Day_Morning": "Binary indicator: 1 if the time of day for promotion is 'Morning', 0 otherwise."
}

RECOMMENDATION_MAP = {
    "Recency": "This customer has not purchased in a while. A personalized re-engagement campaign with a time-sensitive offer could be effective.",
    "Loyalty_Score": "A low loyalty score indicates disengagement. Consider offering exclusive benefits or a tiered loyalty program to increase commitment.",
    "Browsing_Time": "The customer spent very little time browsing. Improve website navigation, product discovery, or offer personalized content to increase engagement.",
    "Clicks_On_Promo": "Low clicks on promotions suggest current offers are not appealing. Experiment with different promotion types, messaging, or channels to find what resonates.",
    "Engagement_Score": "Overall engagement is significantly low. This is a critical area. Focus on personalized communication, interactive content, and exclusive offers to re-ignite interest.",
    "Purchase_Frequency": "This customer purchases infrequently. Implement subscription options, bundle deals, or loyalty rewards for repeat purchases to increase frequency.",
    "Avg_Spend": "The average spend is low. Consider upselling or cross-selling strategies, premium product recommendations, or minimum purchase incentives.",
    "Time_of_Day_Morning": "Promotions sent in the morning might not be effective for this customer. Analyze their typical engagement times and adjust future outreach accordingly.",
    "Age": "Customer's age might be a factor. Tailor product recommendations and messaging to better suit their demographic segment.",
    "Income": "Income level might influence purchasing power. Adjust product recommendations and pricing strategies to align with their financial profile.",
    "Channel_Email": "Email channel might not be effective for this customer. Explore other communication channels like SMS or in-app notifications for future promotions.",
    "Promotion_Type_FlashSale": "Flash sales might not be appealing. Consider offering different promotion types like discounts or loyalty points based on customer preferences."
}

def generate_recommendations(feature_importances):
    if not feature_importances:
        return ["No specific recommendations available as feature importances could not be determined."]

    negative_contributors = {f: v for f, v in feature_importances.items() if v < 0}
    sorted_negative = sorted(negative_contributors.items(), key=lambda item: item[1]) # Sort by value (most negative first)
    
    recommendations = []
    for feature, value in sorted_negative:
        map_key = feature.replace(' ', '_')
        if map_key in RECOMMENDATION_MAP:
            recommendations.append(RECOMMENDATION_MAP[map_key])
        
        if len(recommendations) >= 3:
            break
    
    if not recommendations:
        recommendations.append("This prediction looks positive! The key factors driving this are already strong. Keep up the good work!")
    elif len(recommendations) < 3 and sorted_negative:
        recommendations.append("Consider a deeper analysis of other negatively contributing factors to further improve the outcome.")

    return recommendations

# --- GitHub CSV Update Function ---
def update_github_csv(new_data_df: pd.DataFrame, repo_owner: str, repo_name: str, github_file_path: str, commit_message: str, github_branch: str = "main"):
    github_pat = os.getenv("GITHUB_PAT") # Assuming a single PAT can access both repositories

    if not github_pat:
        print(f"Error: GITHUB_PAT not set in environment variables.")
        return False

    try:
        g = Github(auth=Auth.Token(github_pat))
        repo = g.get_user(repo_owner).get_repo(repo_name)
        
        # Try to get existing file content
        try:
            contents = repo.get_contents(github_file_path, ref=github_branch)
            existing_content = contents.decoded_content.decode('utf-8')
            existing_df = pd.read_csv(io.StringIO(existing_content))
            
            # Append new data
            updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            
            # Convert to CSV string
            new_csv_content = updated_df.to_csv(index=False)
            
            # Update file on GitHub
            repo.update_file(contents.path, commit_message, new_csv_content, contents.sha, branch=github_branch)
            print(f"Successfully updated {github_file_path} in {repo_owner}/{repo_name}.")
            return True
        except Exception as e:
            # File does not exist, create it
            print(f"File {github_file_path} not found in {repo_owner}/{repo_name}, creating it. Error: {e}")
            new_csv_content = new_data_df.to_csv(index=False)
            repo.create_file(github_file_path, commit_message, new_csv_content, branch=github_branch)
            print(f"Successfully created {github_file_path} in {repo_owner}/{repo_name}.")
            return True
            
    except Exception as e:
        print(f"Error updating GitHub CSV for {repo_owner}/{repo_name}: {e}")
        return False
# --- End GitHub CSV Update Function ---

@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual_predict', methods=['GET', 'POST'])
def manual_predict_form():
    if request.method == 'POST':
        previous_input = json.loads(request.form.get('previous_input'))
        return render_template('manual_predict.html', features=EXPECTED_FEATURES, default_values=previous_input, descriptions=FEATURE_DESCRIPTIONS)
    else:
        return render_template('manual_predict.html', features=EXPECTED_FEATURES, default_values=DEFAULT_INPUT_VALUES, descriptions=FEATURE_DESCRIPTIONS)

@app.route('/csv_predict', methods=['GET'])
def csv_predict_form():
    return render_template('csv_predict.html')

@app.route('/dashboard')
@cache.cached()
def dashboard():
    accuracy = "N/A"
    global_importance_plot = "<p><i>No data to generate global importance plot.</i></p>"
    pdp_plot = "<p><i>No data to generate PDP plot.</i></p>"
    errors = []
    round_10_global_accuracy = "N/A" # Initialize new variable

    # Load ledger data and find Round 10 accuracy
    ledger_data = load_ledger_data()
    for entry in ledger_data:
        if entry.get("round") == 10:
            # Convert from float to percentage string
            round_10_global_accuracy = f"{entry.get('global_accuracy', 0) * 100:.2f}%"
            break

    try:
        response = requests.get(FASTAPI_PREDICT_URL.replace('/fed_sys', '/dashboard_data'))
        response.raise_for_status()
        dashboard_data = response.json()

        if "error" in dashboard_data:
            errors.append(dashboard_data["error"])
        else:
            accuracy = f"{dashboard_data.get('accuracy', 0) * 100:.2f}%"
            data_version = dashboard_data.get('data_version')
            model_version = dashboard_data.get('model_version')
            last_updated = dashboard_data.get('last_updated')

            global_importances = dashboard_data.get('global_feature_importances')
            if global_importances:
                sorted_importances = sorted(global_importances.items(), key=lambda item: item[1])
                feature_names = [item[0].replace('_', ' ') for item in sorted_importances]
                importance_values = [item[1] for item in sorted_importances]

                fig_global = go.Figure(go.Bar(
                    y=feature_names,
                    x=importance_values,
                    orientation='h',
                    marker=dict(color=['#0074D9']) # Consistent blue color
                ))
                fig_global.update_layout(
                    title_text='Overall Feature Importance',
                    template='plotly_white',
                    height=600,
                    margin=dict(l=150, r=20, t=50, b=20) # Adjust margins for long labels
                )
                global_importance_plot = fig_global.to_html(full_html=False, include_plotlyjs='cdn')

            pdp_data = dashboard_data.get('pdp_data')
            if pdp_data and pdp_data['feature_values'] and pdp_data['predictions']:
                fig_pdp = go.Figure(go.Scatter(
                    x=pdp_data['feature_values'],
                    y=pdp_data['predictions'],
                    mode='lines+markers',
                    name=pdp_data['feature_name'].replace('_', ' ')
                ))
                fig_pdp.update_layout(
                    title_text=f"Partial Dependence Plot for {pdp_data['feature_name'].replace('_', ' ')}",
                    xaxis_title=pdp_data['feature_name'].replace('_', ' '),
                    yaxis_title="Predicted Probability",
                    template='plotly_white',
                    height=500
                )
                pdp_plot = fig_pdp.to_html(full_html=False, include_plotlyjs='cdn')

    except requests.exceptions.RequestException as e:
        errors.append(f"Error connecting to dashboard API: {e}")
    except json.JSONDecodeError:
        errors.append("Error decoding JSON response from dashboard API.")
    
    return render_template('dashboard.html', accuracy=accuracy, global_importance_plot=global_importance_plot, pdp_plot=pdp_plot, errors=errors, round_10_global_accuracy=round_10_global_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    errors = []
    model_version = None
    last_updated = None
    download_filename = None
    index = 0

    if 'csv_file' in request.files and request.files['csv_file'].filename != '':
        csv_file = request.files['csv_file']
        if csv_file and csv_file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
            csv_file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
                if missing_cols:
                    errors.append(f"CSV file is missing the following required columns: {', '.join(missing_cols)}")
                else:
                    results_df = df.copy()
                    results_df['prediction'] = None
                    results_df['probability'] = None
                    for index, row in df.iterrows():
                        input_data = row[EXPECTED_FEATURES].to_dict()
                        try:
                            response = requests.post(FASTAPI_PREDICT_URL, json=input_data)
                            response.raise_for_status()
                            prediction_result = response.json()

                            plotly_plot_html = None
                            feature_importances = prediction_result.get('feature_importances')
                            if feature_importances:
                                try:
                                    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1])
                                    feature_names = [item[0].replace('_', ' ') for item in sorted_importances]
                                    importance_values = [item[1] for item in sorted_importances]

                                    fig = go.Figure(go.Bar(
                                        y=feature_names,
                                        x=importance_values,
                                        orientation='h',
                                        marker=dict(color=['#FF4136' if v < 0 else '#0074D9' for v in importance_values])
                                    ))
                                    fig.update_layout(
                                        title_text='Feature Contribution to Prediction',
                                        template='plotly_white',
                                        height=600
                                    )
                                    plotly_plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                                except Exception as e:
                                    plotly_plot_html = f"<p><i>Could not generate Plotly plot: {e}</i></p>"
                            
                            recommendations = generate_recommendations(feature_importances)

                            predictions.append({
                                'input': input_data,
                                'prediction': prediction_result.get('prediction'),
                                'probability': prediction_result.get('probability'),
                                'plotly_plot_html': plotly_plot_html,
                                'recommendations': recommendations
                            })
                            results_df.loc[index, 'prediction'] = prediction_result.get('prediction')
                            results_df.loc[index, 'probability'] = prediction_result.get('probability')
                            if not model_version:
                                model_version = prediction_result.get('model_version')
                                last_updated = prediction_result.get('last_updated')
                        except requests.exceptions.RequestException as e:
                            errors.append(f"Error predicting for row {index + 1}: {e}")
                        except json.JSONDecodeError:
                            errors.append(f"Error decoding API response for row {index + 1}.")
                    if not errors:
                        download_filename = f"results_{csv_file.filename}"
                        results_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], download_filename), index=False)
            except Exception as e:
                errors.append(f"An unexpected error occurred: {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            errors.append("Invalid file type. Please upload a CSV file.")
    else:
        input_data = {}
        for feature in EXPECTED_FEATURES:
            value = request.form.get(feature)
            if value is None or value == '':
                errors.append(f"Missing value for {feature}.")
                break
            try:
                target_type = FEATURE_TYPES.get(feature, float)
                if target_type == int:
                    input_data[feature] = int(float(value))
                else:
                    input_data[feature] = target_type(value)
            except ValueError:
                errors.append(f"Invalid value for {feature}. Expected a number of type {FEATURE_TYPES.get(feature, float).__name__}.")
                break
        
        if not errors:
            try:
                response = requests.post(FASTAPI_PREDICT_URL, json=input_data)
                response.raise_for_status()
                prediction_result = response.json()

                plotly_plot_html = None
                feature_importances = prediction_result.get('feature_importances')
                if feature_importances:
                    try:
                        sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1])
                        feature_names = [item[0].replace('_', ' ') for item in sorted_importances]
                        importance_values = [item[1] for item in sorted_importances]

                        fig = go.Figure(go.Bar(
                            y=feature_names,
                            x=importance_values,
                            orientation='h',
                            marker=dict(color=['#FF4136' if v < 0 else '#0074D9' for v in importance_values])
                        ))
                        fig.update_layout(
                            title_text='Feature Contribution to Prediction',
                            template='plotly_white',
                            height=600
                        )
                        plotly_plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    except Exception as e:
                        plotly_plot_html = f"<p><i>Could not generate Plotly plot: {e}</i></p>"
                
                recommendations = generate_recommendations(feature_importances)

                predictions.append({
                    'input': input_data,
                    'prediction': prediction_result.get('prediction'),
                    'probability': prediction_result.get('probability'),
                    'plotly_plot_html': plotly_plot_html,
                    'recommendations': recommendations
                })
                model_version = prediction_result.get('model_version')
                last_updated = prediction_result.get('last_updated')

                new_row_data = input_data.copy()
                predicted_value = prediction_result.get('prediction')
                new_row_data['Will_Buy'] = int(predicted_value) if predicted_value is not None else 0
                df_new_row = pd.DataFrame([new_row_data])
                
                github_repos_config = [
                    {
                        "owner": os.getenv("GITHUB_REPO1_OWNER"),
                        "name": os.getenv("GITHUB_REPO1_NAME"),
                        "file_path": os.getenv("GITHUB_REPO1_FILE_PATH", "data/manual_predictions_for_retraining.csv"),
                        "branch": os.getenv("GITHUB_REPO1_BRANCH", "main")
                    },
                    {
                        "owner": os.getenv("GITHUB_REPO2_OWNER"),
                        "name": os.getenv("GITHUB_REPO2_NAME"),
                        "file_path": os.getenv("GITHUB_REPO2_FILE_PATH", "data/manual_predictions_for_retraining.csv"),
                        "branch": os.getenv("GITHUB_REPO2_BRANCH", "main")
                    }
                ]
                
                commit_message = "Add new manual prediction for retraining"
                
                all_successful = True
                for repo_config in github_repos_config:
                    if not repo_config["owner"] or not repo_config["name"]:
                        errors.append(f"GitHub repository configuration incomplete for one of the targets.")
                        all_successful = False
                        continue

                    if not update_github_csv(
                        df_new_row,
                        repo_config["owner"],
                        repo_config["name"],
                        repo_config["file_path"],
                        commit_message,
                        repo_config["branch"]
                    ):
                        errors.append(f"Failed to save prediction to GitHub for {repo_config['owner']}/{repo_config['name']}.")
                        all_successful = False

            except requests.exceptions.RequestException as e:
                errors.append(f"Error connecting to prediction API: {e}")
            except json.JSONDecodeError:
                errors.append("Error decoding JSON response from API.")

    return render_template('results.html', predictions=predictions, errors=errors, model_version=model_version, last_updated=last_updated, download_filename=download_filename)

@app.route('/download_results/<filename>')
def download_results(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

