import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask import session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, IntegerField, FloatField
from werkzeug.utils import secure_filename
import joblib
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import requests
import uuid
import re
from markupsafe import Markup
import pickle
from datetime import datetime, timedelta


# PERSISTENT STORAGE CONFIGURATION
RESULTS_STORAGE_DIR = 'saved_results'
RESULTS_INDEX_FILE = os.path.join(RESULTS_STORAGE_DIR, 'results_index.json')

# Ensure storage directory exists
os.makedirs(RESULTS_STORAGE_DIR, exist_ok=True)

# In-memory store for analysis results (for current session)
TEMP_RESULT_CACHE = {}

# Persistent results index
PERSISTENT_RESULTS = {}

def load_results_index():
    """Load the persistent results index"""
    global PERSISTENT_RESULTS
    if os.path.exists(RESULTS_INDEX_FILE):
        try:
            with open(RESULTS_INDEX_FILE, 'r') as f:
                PERSISTENT_RESULTS = json.load(f)
        except Exception as e:
            print(f"Error loading results index: {e}")
            PERSISTENT_RESULTS = {}
    else:
        PERSISTENT_RESULTS = {}

def save_results_index():
    """Save the persistent results index"""
    try:
        with open(RESULTS_INDEX_FILE, 'w') as f:
            json.dump(PERSISTENT_RESULTS, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving results index: {e}")

def save_result_permanently(result_id, results_data):
    """Save analysis results permanently to disk"""
    try:
        result_file_path = os.path.join(RESULTS_STORAGE_DIR, f"{result_id}.pkl")
        
        # Save the full results data using pickle
        with open(result_file_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        # Update the index with metadata
        PERSISTENT_RESULTS[result_id] = {
            'filename': results_data.get('filename', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'anomaly_count': results_data.get('anomaly_count', 0),
            'total_count': results_data.get('total_count', 0),
            'file_path': result_file_path
        }
        
        # Save the updated index
        save_results_index()
        
        print(f"Results saved permanently: {result_id}")
        return True
        
    except Exception as e:
        print(f"Error saving results permanently: {e}")
        return False

def load_result_from_disk(result_id):
    """Load analysis results from disk"""
    try:
        if result_id not in PERSISTENT_RESULTS:
            return None
            
        result_file_path = PERSISTENT_RESULTS[result_id]['file_path']
        
        if not os.path.exists(result_file_path):
            return None
            
        with open(result_file_path, 'rb') as f:
            results_data = pickle.load(f)
            
        return results_data
        
    except Exception as e:
        print(f"Error loading results from disk: {e}")
        return None

def cleanup_old_results(days_old=30):
    """Clean up results older than specified days"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        results_to_remove = []
        
        for result_id, metadata in PERSISTENT_RESULTS.items():
            created_date = datetime.fromisoformat(metadata['created_at'])
            if created_date < cutoff_date:
                results_to_remove.append(result_id)
        
        for result_id in results_to_remove:
            result_file_path = PERSISTENT_RESULTS[result_id]['file_path']
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
            del PERSISTENT_RESULTS[result_id]
        
        save_results_index()
        print(f"Cleaned up {len(results_to_remove)} old results")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

# For LIME explanations
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load persistent results on startup
load_results_index()

# Hardcoded model path
MODEL_PATH = "solar_anomaly_detector.joblib"

# Forms
class UploadForm(FlaskForm):
    csv_file = FileField('CSV File', validators=[FileRequired()])
    n_explanations = IntegerField('Number of Explanations', default=10)
    contamination = FloatField('Contamination', default=0.05)
    submit = SubmitField('Analyze')

# Helper classes
class GroqAnalyzer:
    """Integration with Groq API for detailed anomaly analysis"""
    
    def __init__(self, api_key):
        self.api_key = "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"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.cache_file = "groq_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cached Groq responses"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _generate_cache_key(self, lime_features, sample_data, device_id, timestamp):
        """Generate a unique key for caching"""
        feature_str = "|".join([f"{f[0]}:{f[1]:.4f}" for f in lime_features])
        data_str = "|".join([f"{k}:{v:.4f}" for k, v in sample_data.items()])
        return f"{device_id}|{timestamp}|{feature_str}|{data_str}"
    
    def _format_groq_response(self, text):
        """Format Groq response with proper HTML formatting"""
        # Convert markdown-like formatting to HTML
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Convert numbered lists
        text = re.sub(r'(\d+)\.\s', r'<br><strong>\1.</strong> ', text)
        
        # Convert section headers
        text = re.sub(r'(\n|^)([A-Z\s]+):', r'<br><br><strong><u>\2:</u></strong><br>', text)
        # Convert bullet points into <ul><li>...</li></ul>
        def replace_bullets(match):
            items = match.group(0).strip().split('\n')
            list_items = ''.join(f'<li>{item[2:].strip()}</li>' for item in items if item.startswith('* '))
            return f'<ul>{list_items}</ul>'

        text = re.sub(r'((?:\* .+(?:\n|$))+)', replace_bullets, text)
        # Ensure proper line breaks
        text = text.replace('\n', '<br>')
        
        return Markup(text)
    
    def analyze_lime_results(self, lime_features, sample_data, device_id=None, timestamp=None):
        """Send LIME results to Groq for detailed analysis with caching"""
        
        # Prepare the prompt for Groq
        feature_analysis = "\n".join([f"- {feature}: Impact Score = {weight:.4f}" 
                                    for feature, weight in lime_features])
        
        sample_data_str = "\n".join([f"- {key}: {value:.4f}" 
                                   for key, value in sample_data.items()])
        
        prompt = f"""
        You are an expert solar energy system diagnostician. Analyze the following anomaly detection results from a solar power system:

        ANOMALY DETAILS:
        - Device ID: {str(device_id) if device_id else 'Unknown'}
        - Timestamp: {str(timestamp) if timestamp else 'Unknown'}
        
        LIME FEATURE ANALYSIS (Top contributing factors):
        {feature_analysis}

        SAMPLE DATA VALUES:
        {sample_data_str}

        Please provide a detailed analysis including:
        1. PROBLEM IDENTIFICATION: What specific issues are indicated by these features?
        2. ROOT CAUSE ANALYSIS: What are the most likely root causes of this anomaly?
        3. AFFECTED COMPONENTS: Which solar system components are likely affected (inverter, MPPT controllers, panels, grid connection, etc.)?
        4. SEVERITY ASSESSMENT: Rate the severity (Low/Medium/High/Critical)
        5. RECOMMENDED ACTIONS: Specific maintenance or troubleshooting steps
        6. POTENTIAL IMPACT: How this might affect system performance and energy production
        7. MONITORING PRIORITIES: What parameters should be monitored closely

        Format your response in clear sections with actionable insights.
        """
        
        # Check cache first
        cache_key = self._generate_cache_key(lime_features, sample_data, device_id, timestamp)
        if cache_key in self.cache:
            print("Using cached Groq response")
            return self._format_groq_response(self.cache[cache_key])
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert solar energy system diagnostician with deep knowledge of photovoltaic systems, inverters, MPPT controllers, and grid-tied solar installations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                # Cache the result
                self.cache[cache_key] = result
                self._save_cache()
                return self._format_groq_response(result)
            else:
                error_msg = f"Error calling Groq API: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error calling Groq API: {str(e)}"
            print(error_msg)
            return error_msg

class SolarAnomalyDetector:
    def __init__(self, contamination=0.05, groq_api_key=None):
        """
        Solar Anomaly Detection System with Groq Integration

        Args:
            contamination: Expected proportion of anomalies in the dataset
            groq_api_key: API key for Groq integration
        """
        self.contamination = contamination
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.lime_explainer = None
        self.groq_analyzer = GroqAnalyzer(groq_api_key) if groq_api_key else None
        self.detailed_results = []  # Store all detailed analysis results
        
        # Load the hardcoded model
        self.load_model(MODEL_PATH)

    def load_and_preprocess_data(self, df):
        print("Loading and preprocessing data...")

        # Clean column names (remove spaces, make lowercase for consistency)
        df.columns = df.columns.str.strip().str.lower()

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert date to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop unwanted columns if present
        cols_to_remove = ['fault', 'count_data_reading_cycle']
        df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')
        
        # --- Power efficiency features ---
        df['efficiency'] = df['total_output_power'] / (df['total_dc_power'] + 1e-5)
        df['mppt_ratio'] = df['mppt1_dc_power'] / (df['mppt2_dc_power'] + 1e-5)
        
        # --- MPPT Current Imbalance ---
        df['mppt_current_imbalance'] = df[['mppt1_dc_current', 'mppt2_dc_current']].std(axis=1) / \
                                      df[['mppt1_dc_current', 'mppt2_dc_current']].mean(axis=1)

        # --- MPPT Voltage Imbalance ---
        df['mppt_voltage_imbalance'] = df[['mppt1_dc_voltage', 'mppt2_dc_voltage']].std(axis=1) / \
                                      df[['mppt1_dc_voltage', 'mppt2_dc_voltage']].mean(axis=1)

        # --- MPPT Efficiency ---
        df['mppt1_efficiency'] = df['mppt1_dc_power'] / (df['mppt1_dc_voltage'] * df['mppt1_dc_current'] + 1e-5)
        df['mppt2_efficiency'] = df['mppt2_dc_power'] / (df['mppt2_dc_voltage'] * df['mppt2_dc_current'] + 1e-5)
        df['mppt_eff_diff'] = abs(df['mppt1_efficiency'] - df['mppt2_efficiency'])

        # --- Voltage stability ---
        df['voltage_imbalance'] = df[['grid_voltage_v_1', 'grid_voltage_v_2', 'grid_voltage_v_3']].std(axis=1) / \
                                  df[['grid_voltage_v_1', 'grid_voltage_v_2', 'grid_voltage_v_3']].mean(axis=1)

        # --- Current stability ---
        df['current_imbalance'] = df[['grid_current_i_1', 'grid_current_i_2', 'grid_current_i_3']].std(axis=1) / \
                                  df[['grid_current_i_1', 'grid_current_i_2', 'grid_current_i_3']].mean(axis=1)

        # --- Temperature ---
        df['temp_diff'] = df['cell_temperature'] - df['cabinet_temperature']
        df['temp_ratio'] = df['cell_temperature'] / (df['avg_amb_temp'] + 1e-5)

        # --- Time-based features (needs 'hour' column present) ---
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # --- Power degradation ---
        df['power_ratio'] = df['total_output_power'] / (df['apparent_power'] + 1e-5)

        # (Optional sanity check)
        print("Feature engineering complete. New shape:", df.shape)

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Original dataset shape: {df.shape}")
        return df

    def feature_selection_and_engineering(self, df):
        """Intelligent feature selection and engineering for solar anomaly detection"""
        print("Performing feature selection and engineering...")

        # Remove columns that are not useful for anomaly detection
        columns_to_remove = [
            'timestamp',  # Will use engineered time features instead
            'count_data_reading_cycle',  # Just a counter
            'date',       # Redundant with timestamp
            'today_e_increment'  # Derived feature, might cause data leakage
        ]

        # Keep only relevant columns
        df_clean = df.drop(columns=columns_to_remove, errors='ignore')

        # Feature Engineering
        print("Engineering new features...")

        # Time-based features (from timestamp)
        if 'hour' in df.columns:
            df_clean['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df_clean['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df_clean['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df_clean['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df_clean['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df_clean['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Power efficiency ratios
        df_clean['dc_to_ac_efficiency'] = df_clean['total_output_power'] / (df_clean['total_dc_power'] + 1e-6)
        df_clean['mppt1_efficiency'] = df_clean['mppt1_dc_power'] / (df_clean['mppt1_dc_voltage'] * df_clean['mppt1_dc_current'] + 1e-6)
        df_clean['mppt2_efficiency'] = df_clean['mppt2_dc_power'] / (df_clean['mppt2_dc_voltage'] * df_clean['mppt2_dc_current'] + 1e-6)

        # Power balance features
        df_clean['mppt_power_balance'] = abs(df_clean['mppt1_dc_power'] - df_clean['mppt2_dc_power'])
        df_clean['total_mppt_power'] = df_clean['mppt1_dc_power'] + df_clean['mppt2_dc_power']
        df_clean['mppt_dc_power_diff'] = abs(df_clean['total_mppt_power'] - df_clean['total_dc_power'])

        # Voltage and current ratios
        df_clean['voltage_imbalance_12'] = abs(df_clean['grid_voltage_v_1'] - df_clean['grid_voltage_v_2'])
        df_clean['voltage_imbalance_13'] = abs(df_clean['grid_voltage_v_1'] - df_clean['grid_voltage_v_3'])
        df_clean['voltage_imbalance_23'] = abs(df_clean['grid_voltage_v_2'] - df_clean['grid_voltage_v_3'])

        df_clean['current_imbalance_12'] = abs(df_clean['grid_current_i_1'] - df_clean['grid_current_i_2'])
        df_clean['current_imbalance_13'] = abs(df_clean['grid_current_i_1'] - df_clean['grid_current_i_3'])
        df_clean['current_imbalance_23'] = abs(df_clean['grid_current_i_2'] - df_clean['grid_current_i_3'])

        # Temperature-based features
        df_clean['temp_difference'] = df_clean['max_amb_temp'] - df_clean['min_amb_temp']
        df_clean['cabinet_temp_deviation'] = df_clean['cabinet_temperature'] - df_clean['avg_amb_temp']
        df_clean['cell_temp_deviation'] = df_clean['cell_temperature'] - df_clean['avg_amb_temp']

        # Power factor and reactive power features
        df_clean['power_factor'] = df_clean['total_output_power'] / (df_clean['apparent_power'] + 1e-6)
        df_clean['reactive_power_ratio'] = df_clean['reactive_power'] / (df_clean['total_output_power'] + 1e-6)

        # Irradiance-based features
        df_clean['power_per_irradiance'] = df_clean['total_output_power'] / (df_clean['avg_irrad'] + 1e-6)

        # Remove original time columns as we have engineered features
        time_cols_to_remove = ['hour', 'day_of_week', 'month']
        df_clean = df_clean.drop(columns=time_cols_to_remove, errors='ignore')

        # Handle infinite and NaN values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())

        print(f"After feature engineering shape: {df_clean.shape}")
        print(f"Features selected: {list(df_clean.columns)}")

        return df_clean

    def load_model(self, model_path):
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']

        print(f"Model loaded from {model_path}")
        return self

    def predict_anomalies(self, X_test):
        """Predict anomalies on test data"""
        print("Predicting anomalies...")

        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_test_scaled)

        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X_test_scaled)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        binary_predictions = np.where(predictions == -1, 1, 0)

        return binary_predictions, anomaly_scores, X_test_scaled

    def explain_anomalies_with_lime_and_groq(self, X_test, X_test_scaled, anomaly_indices, original_df, n_explanations=10):
        """Generate LIME explanations and get detailed Groq analysis for detected anomalies"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            print("LIME not available for explanations")
            return

        print(f"\n" + "="*70)
        print("DETAILED ANOMALY ANALYSIS WITH LIME + GROQ")
        print("="*70)

        # Explain top N anomalies
        n_to_explain = min(n_explanations, len(anomaly_indices))

        for i in range(n_to_explain):
            idx = anomaly_indices[i]
            print(f"\n" + "="*70)
            print(f"ANOMALY #{i+1} - DETAILED ANALYSIS (Sample Index: {idx})")
            print("="*70)

            # Get device ID and timestamp
            print(original_df.info())
            device_id = original_df.loc[idx, 'q_serial'] if 'q_serial' in original_df.columns else f"Device_{idx}"
            timestamp = original_df.loc[idx, 'timestamp'] if 'timestamp' in original_df.columns else "Unknown"
            
            print(f"Device ID: {device_id}")
            print(f"Timestamp: {timestamp}")
            print(f"Anomaly Score: {self.model.decision_function(X_test_scaled[idx].reshape(1, -1))[0]:.4f}")

            try:
                # Create a wrapper function for LIME
                def predict_fn(X):
                    preds = self.model.predict(X)
                    # Convert to probability-like format for LIME
                    probs = np.zeros((len(preds), 2))
                    probs[:, 0] = (preds == 1).astype(float)  # Normal
                    probs[:, 1] = (preds == -1).astype(float)  # Anomaly
                    return probs

                # Generate LIME explanation
                explanation = self.lime_explainer.explain_instance(
                    X_test_scaled[idx].reshape(1, -1)[0],  # Fix: reshape and take first element
                    predict_fn,
                    num_features=15,  # Show top 15 features
                    top_labels=1
                )

                # Get LIME feature explanations
                lime_features = explanation.as_list(label=1)  # Label 1 = Anomaly
                
                print(f"\nLIME FEATURE ANALYSIS:")
                print("-" * 40)
                for feature, weight in lime_features:
                    print(f"  {feature:<30}: {weight:>8.4f}")

                # Get sample data for Groq analysis
                sample_data = {}
                for feature_name in self.feature_names:
                    if feature_name in X_test.columns:
                        sample_data[feature_name] = X_test.iloc[idx][feature_name]

                # Get Groq analysis if available
                if self.groq_analyzer:
                    print(f"\n🤖 GROQ AI EXPERT ANALYSIS:")
                    print("-" * 40)
                    
                    groq_analysis = self.groq_analyzer.analyze_lime_results(
                        lime_features, 
                        sample_data, 
                        device_id=str(device_id),
                        timestamp=str(timestamp)
                    )
                    
                    print(groq_analysis)
                    
                    # Store detailed result
                    detailed_result = {
                        'anomaly_index': idx,
                        'device_id': device_id,
                        'timestamp': timestamp,
                        'lime_features': lime_features,
                        'groq_analysis': groq_analysis,
                        'sample_data': sample_data
                    }
                    self.detailed_results.append(detailed_result)
                    
                else:
                    print("\nGroq analysis not available (API key not provided)")

            except Exception as e:
                print(f"Error generating explanation: {str(e)}")
                import traceback
                traceback.print_exc()

        return self.detailed_results

    def create_visualizations(self, X_test, predictions, anomaly_scores, original_df):
        """Create visualizations and return as base64 encoded images"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Solar Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Anomaly score distribution
        axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=np.percentile(anomaly_scores, 5), color='red', linestyle='--',
                          label=f'5th percentile: {np.percentile(anomaly_scores, 5):.3f}')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Time series of total output power with anomalies highlighted
        if 'timestamp' in original_df.columns and 'total_output_power' in original_df.columns:
            axes[0, 1].plot(original_df['timestamp'], original_df['total_output_power'],
                           alpha=0.7, label='Normal', color='blue', linewidth=1)

            anomaly_mask = predictions == 1
            if np.any(anomaly_mask):
                axes[0, 1].scatter(original_df['timestamp'][anomaly_mask],
                                  original_df['total_output_power'][anomaly_mask],
                                  color='red', label='Anomalies', s=20, alpha=0.8)

            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Total Output Power')
            axes[0, 1].set_title('Power Output Over Time')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance (based on anomaly detection)
        if len(X_test.columns) > 0:
            # Calculate feature importance based on variance in anomaly scores
            feature_importance = []
            for col in X_test.columns:
                corr_with_score = np.corrcoef(X_test[col], anomaly_scores)[0, 1]
                feature_importance.append(abs(corr_with_score))

            top_features = sorted(zip(X_test.columns, feature_importance),
                                key=lambda x: x[1], reverse=True)[:10]

            features, importance = zip(*top_features)
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            axes[1, 0].barh(range(len(features)), importance, color=colors)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_xlabel('Correlation with Anomaly Score')
            axes[1, 0].set_title('Top 10 Most Important Features')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Anomaly detection results summary
        normal_count = np.sum(predictions == 0)
        anomaly_count = np.sum(predictions == 1)

        colors = ['#66c2a5', '#fc8d62']
        wedges, texts, autotexts = axes[1, 1].pie([normal_count, anomaly_count],
                      labels=['Normal', 'Anomaly'],
                      autopct='%1.1f%%',
                      colors=colors,
                      startangle=90)
        
        # Make the autotexts more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        axes[1, 1].set_title('Anomaly Detection Results')

        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_data

    def create_lime_plot(self, lime_features, index):
        """Create a visualization for LIME feature contributions"""
        # Extract feature names and weights
        features = [item[0] for item in lime_features]
        weights = [item[1] for item in lime_features]
        
        # Create color mapping based on positive/negative weights
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Contribution Weight')
        ax.set_title(f'LIME Feature Contributions for Anomaly #{index+1}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels to bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + (0.01 if weight >= 0 else -0.01), i, 
                   f'{weight:.4f}', 
                   va='center', ha='left' if weight >= 0 else 'right',
                   fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_data

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    
    if form.validate_on_submit():
        try:
            # Save uploaded files
            csv_filename = secure_filename(form.csv_file.data.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
            form.csv_file.data.save(csv_path)
            
            # Load data
            df = pd.read_csv(csv_path)
            
            # Initialize detector with hardcoded model
            detector = SolarAnomalyDetector(
                contamination=form.contamination.data,
                groq_api_key="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"
            )
            
            # Preprocess data
            df_cleaned = detector.load_and_preprocess_data(df)
            df_processed = detector.feature_selection_and_engineering(df_cleaned)
            
            # Ensure the test features match training features
            X_test = df_processed[detector.feature_names]
            
            # Run predictions
            predictions, anomaly_scores, X_test_scaled = detector.predict_anomalies(X_test)
            
            # Add results to DataFrame
            df['anomaly'] = predictions
            df['anomaly_score'] = anomaly_scores
            
            # Get anomaly indices
            anomaly_indices = np.where(predictions == 1)[0]
            
            # Create visualizations
            plot_data = detector.create_visualizations(X_test, predictions, anomaly_scores, df_cleaned)
            
            # Initialize LIME explainer if available
            if LIME_AVAILABLE:
                X_train_scaled = detector.scaler.transform(X_test)  # Using test data as proxy
                detector.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_scaled,
                    feature_names=detector.feature_names,
                    class_names=['Normal', 'Anomaly'],
                    mode='classification',
                    discretize_continuous=True
                )
            
            # Explain anomalies
            detailed_results = detector.explain_anomalies_with_lime_and_groq(
                X_test=X_test,
                X_test_scaled=X_test_scaled,
                anomaly_indices=anomaly_indices,
                original_df=df_cleaned,
                n_explanations=form.n_explanations.data
            )
            
            # Create LIME plots for each anomaly
            lime_plots = []
            for i, result in enumerate(detector.detailed_results):
                lime_plot = detector.create_lime_plot(result['lime_features'], i)
                lime_plots.append(lime_plot)
            
            # Generate a result ID based on filename and timestamp
            result_id = f"{os.path.splitext(csv_filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare results data
            results_data = {
                'df': df.to_dict(),
                'plot_data': plot_data,
                'detailed_results': detector.detailed_results,
                'lime_plots': lime_plots,
                'anomaly_count': len(anomaly_indices),
                'total_count': len(predictions),
                'filename': csv_filename,
                'created_at': datetime.now().isoformat(),
                'contamination': form.contamination.data,
                'n_explanations': form.n_explanations.data
            }
            
            # Store in temporary cache for immediate access
            TEMP_RESULT_CACHE[result_id] = results_data
            
            # Save permanently to disk
            save_success = save_result_permanently(result_id, results_data)
            
            # Store only the ID in session
            session['result_id'] = result_id

            if save_success:
                flash(f'Analysis completed and saved permanently! Result ID: {result_id}', 'success')
            else:
                flash('Analysis completed but there was an issue saving permanently. Results are available for this session.', 'warning')
                
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            flash(f'Error during analysis: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html', form=form)

@app.route('/results/<result_id>')
def results(result_id):
    """Display results for a specific analysis"""
    
    # First try to get from temporary cache
    results = TEMP_RESULT_CACHE.get(result_id)
    
    # If not in cache, try to load from disk
    if not results:
        results = load_result_from_disk(result_id)
    
    # If still not found, show error
    if not results:
        flash('Analysis results not found. The results may have been cleaned up or the ID is invalid.', 'error')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=results, result_id=result_id)

@app.route('/download/<result_id>')
def download(result_id):
    """Download results as CSV"""
    
    # First try to get from temporary cache
    results = TEMP_RESULT_CACHE.get(result_id)
    
    # If not in cache, try to load from disk
    if not results:
        results = load_result_from_disk(result_id)
    
    # If still not found, show error
    if not results:
        flash('Analysis results not found for download.', 'error')
        return redirect(url_for('index'))

    df = pd.DataFrame.from_dict(results['df'])

    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = results.get('filename', 'solar_anomaly_results')
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{filename}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route('/saved_results')
def saved_results():
    """Display a list of all saved analysis results"""
    
    # Get all saved results from the index
    saved_analyses = []
    
    for result_id, metadata in PERSISTENT_RESULTS.items():
        saved_analyses.append({
            'result_id': result_id,
            'filename': metadata.get('filename', 'Unknown'),
            'created_at': metadata.get('created_at', 'Unknown'),
            'anomaly_count': metadata.get('anomaly_count', 0),
            'total_count': metadata.get('total_count', 0),
            'anomaly_percentage': round((metadata.get('anomaly_count', 0) / max(metadata.get('total_count', 1), 1)) * 100, 2)
        })
    
    # Sort by creation date (newest first)
    saved_analyses.sort(key=lambda x: x['created_at'], reverse=True)
    
    return render_template('saved_results.html', saved_analyses=saved_analyses)

@app.route('/delete_result/<result_id>')
def delete_result(result_id):
    """Delete a saved analysis result"""
    
    try:
        if result_id in PERSISTENT_RESULTS:
            # Remove the file
            result_file_path = PERSISTENT_RESULTS[result_id]['file_path']
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
            
            # Remove from index
            del PERSISTENT_RESULTS[result_id]
            save_results_index()
            
            # Remove from temp cache if present
            if result_id in TEMP_RESULT_CACHE:
                del TEMP_RESULT_CACHE[result_id]
            
            flash(f'Analysis result {result_id} deleted successfully.', 'success')
        else:
            flash('Analysis result not found.', 'error')
            
    except Exception as e:
        flash(f'Error deleting result: {str(e)}', 'error')
    
    return redirect(url_for('saved_results'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        # Create DataFrame from CSV data
        df = pd.DataFrame(data['csv_data'])
        
        # Initialize detector
        detector = SolarAnomalyDetector(
            contamination=data.get('contamination', 0.05),
            groq_api_key="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"
        )
        
        # Preprocess data
        df_cleaned = detector.load_and_preprocess_data(df)
        df_processed = detector.feature_selection_and_engineering(df_cleaned)
        
        # Run predictions
        predictions, anomaly_scores, _ = detector.predict_anomalies(df_processed)
        
        # Add results to DataFrame
        df['anomaly'] = predictions
        df['anomaly_score'] = anomaly_scores
        
        # Optionally save results if requested
        if data.get('save_results', False):
            result_id = f"api_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results_data = {
                'df': df.to_dict(),
                'anomaly_count': int(np.sum(predictions)),
                'total_count': len(predictions),
                'filename': data.get('filename', 'api_data'),
                'created_at': datetime.now().isoformat(),
                'contamination': data.get('contamination', 0.05),
            }
            
            save_result_permanently(result_id, results_data)
        
        # Prepare response
        response = {
            'success': True,
            'anomaly_count': int(np.sum(predictions)),
            'total_count': len(predictions),
            'anomaly_percentage': float(np.sum(predictions) / len(predictions) * 100),
            'results': df.to_dict()
        }
        
        if data.get('save_results', False):
            response['result_id'] = result_id
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum file size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    flash('Internal server error. Please try again.', 'error')
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)