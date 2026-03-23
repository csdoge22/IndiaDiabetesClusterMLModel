import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'diabetes.csv'))
PLOT_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'plots'))
FINDINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Findings'))

# Ensure directories exist
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(FINDINGS_PATH, exist_ok=True)

# --- PREPROCESSING MODULE ---
def clean_and_impute(df: pd.DataFrame, use_group_imputation: bool = True):
    """Handles data quality issues using conditional median imputation."""
    non_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin']
    for col in non_zero_cols:
        df[col] = df[col].replace(0, np.nan)
        if use_group_imputation:
            df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

# --- ANALYSIS MODULE ---
def run_stat_tests(df: pd.DataFrame):
    """Performs Mann-Whitney U tests on features."""
    results = []
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Outcome']
    for col in features:
        g0, g1 = df[df['Outcome'] == 0][col], df[df['Outcome'] == 1][col]
        _, p_val = mannwhitneyu(g0, g1)
        results.append({'Feature': col, 'p_value': p_val})
    return pd.DataFrame(results).sort_values('p_value')

# --- MODELING MODULE ---
def evaluate_models(df: pd.DataFrame):
    """Compares Logistic Regression and Random Forest using 5-Fold CV."""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    summary_data = {}

    print("\n--- Model Comparison Report ---")
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        summary_data[name] = {'Mean': scores.mean(), 'Std': scores.std()}
        
        print(f"{name}:")
        print(f"  Mean Accuracy: {scores.mean():.2%}")
        print(f"  Stability (Std Dev): {scores.std():.2%}")

    return summary_data

# --- VISUALIZATION MODULE ---
def generate_visuals(df: pd.DataFrame):
    """Generates pairplot and individual boxplots."""
    print("Generating Pairplot...")
    sns.pairplot(df, hue='Outcome', diag_kind='kde')
    plt.savefig(os.path.join(PLOT_PATH, 'pairplot_final.png'))
    plt.close()

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Outcome']
    for column in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=column, hue='Outcome')
        plt.savefig(os.path.join(PLOT_PATH, f'boxplot_{column}.png'))
        plt.close()

def plot_feature_importance(df: pd.DataFrame):
    """Visualizes which features the Random Forest prioritized."""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Random Forest Feature Importance (Global Mode)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'feature_importance.png'))
    plt.close()
    return importance_df

# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    raw_df = pd.read_csv(DATA_PATH)
    report_content = ["=== DIABETES ML PROJECT SUMMARY ===\n"]

    for use_leakage_mode in [True, False]:
        mode_label = "GROUPED (Leakage Risk)" if use_leakage_mode else "GLOBAL (Real-World)"
        report_content.append(f"\nEXPERIMENT: {mode_label}")
        report_content.append("-" * 30)
        
        print(f"\nRunning Experiment: {mode_label}...")
        
        # 1. Preprocess fresh copy
        df_processed = clean_and_impute(raw_df.copy(), use_group_imputation=use_leakage_mode)
        
        # 2. Stats
        stats_df = run_stat_tests(df_processed)
        report_content.append(f"Top Predictor: {stats_df.iloc[0]['Feature']} (p={stats_df.iloc[0]['p_value']:.2e})")

        # 3. Model Results
        results = evaluate_models(df_processed)
        for model_name, metrics in results.items():
            report_content.append(f"{model_name}: {metrics['Mean']:.2%} (+/- {metrics['Std']:.2%})")
        
        # 4. Global-only artifacts
        if not use_leakage_mode:
            print("Generating Visuals and Importance Plot...")
            generate_visuals(df_processed)
            importance_df = plot_feature_importance(df_processed)
            
            # Save stats table
            stats_df.to_csv(os.path.join(FINDINGS_PATH, 'clinical_significance.tsv'), sep='\t', index=False)

    # Write final report to file
    report_path = os.path.join(FINDINGS_PATH, 'final_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_content))

    print(f"\n{'='*45}")
    print(f"PROJECT COMPLETE. Summary saved to: {report_path}")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()