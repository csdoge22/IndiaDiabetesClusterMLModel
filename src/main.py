import os
from typing import Optional, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'diabetes.csv'))
PLOT_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'plots'))
FINDINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Findings'))

os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(FINDINGS_PATH, exist_ok=True)

# --- PREPROCESSING MODULE ---
def clean_and_impute(df: pd.DataFrame, use_group_imputation: bool = True) -> pd.DataFrame:
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
def run_stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Performs Mann-Whitney U tests on features."""
    results = []
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Outcome']
    for col in features:
        g0, g1 = df[df['Outcome'] == 0][col], df[df['Outcome'] == 1][col]
        _, p_val = mannwhitneyu(g0, g1)
        results.append({'Feature': col, 'p_value': p_val})
    return pd.DataFrame(results).sort_values('p_value')

# --- MODELING MODULE ---

def get_model_configs(activation_function: Optional[str] = 'relu') -> Dict:
    """
     Returns a dictionary of model instances. Centralizes hyperparameter management.
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(8, 4), 
            activation=activation_function, 
            max_iter=10000, 
            random_state=42, 
            solver='lbfgs'
        )
    }

def evaluate_models(df: pd.DataFrame, models: Dict) -> Dict:
    """
    Accepts any dictionary of models and runs 5-Fold Stratified CV.
    Implements Dependency Injection for models and scalers.
    """
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(X, y))
    summary_data = {}

    print("\n--- Model Comparison Report ---")
    for name, model in models.items():
        # Encapsulated Pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        test_labels, predicted_labels, scores = [], [], []
        
        for train_idx, test_idx in folds:
            x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            
            test_labels.append(y_test)
            predicted_labels.append(y_pred)
            scores.append(accuracy_score(y_test, y_pred))
        
        # Results Aggregation
        scores = np.asarray(scores)
        y_true_final = np.concatenate(test_labels)
        y_pred_final = np.concatenate(predicted_labels)
        tn, fp, fn, tp = confusion_matrix(y_true_final, y_pred_final).ravel()
        
        summary_data[name] = {
            'Mean': scores.mean(),
            'Std': scores.std(),
            'Matrix Report': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
        }
        
        print(f"{name}:")
        print(f"  Mean Accuracy: {scores.mean():.2%}")
        print(f"  Stability (Std Dev): {scores.std():.2%}")
        print(f"  Clinical Risk: {fn} Missed Cases (Type 2), {fp} False Alarms (Type 1)")
        
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

def plot_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Visualizes which features the Random Forest prioritized."""
    X, y = df.drop(columns=['Outcome']), df['Outcome']
    
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

def plot_mlp_loss_curve(df: pd.DataFrame, activation_type: str = 'relu') -> float:
    """Visualizes learning journey using Adam solver for history tracking."""
    X, y = df.drop(columns=['Outcome']), df['Outcome']
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=(8,4), 
            max_iter=1000,
            solver='adam',
            activation=activation_type,
            random_state=42
        ))
    ])
    
    pipeline.fit(X, y)
    mlp = pipeline.named_steps['model']
    
    plt.figure(figsize=(10, 6))
    plt.grid(visible=True)
    plt.plot(mlp.loss_curve_, label=f'Activation: {activation_type}')
    plt.title(f'Neural Network Learning: {activation_type.capitalize()}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(PLOT_PATH, f'loss_curve_{activation_type}.png'))
    plt.close()
    
    return mlp.loss_

def plot_final_champion_matrix(df: pd.DataFrame):
    """
    Generates a single, final Confusion Matrix for the Logistic Regression Champion.
    Uses an 80/20 train/test split (Holdout Method) for a clear visualization.
    """
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Define your champion model and its pipeline
    # (Must match the exact config from evaluate_models)
    champion_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    
    # 1. Create a single 80/20 train/test split
    # (This provides a definitive confusion matrix from a 'test' set)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 2. Fit on training data
    print("Fitting Champion Model (Logistic Regression)...")
    champion_pipeline.fit(X_train, y_train)
    
    # 3. Predict on holdout data
    y_pred = champion_pipeline.predict(X_test)
    
    # 4. Generate the matrix visualization
    print("Generating Champion Confusion Matrix...")
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        display_labels=['Healthy', 'Diabetic'], 
        cmap='Blues', 
        colorbar=False
    )
    
    # 5. Customize and save
    plt.title('Final Champion Performance: Logistic Regression (Holdout Test)')
    plt.tight_layout()
    
    # Save a clean version for your findings
    plot_file = os.path.join(PLOT_PATH, 'champion_confusion_matrix.png')
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Champion Matrix saved to: {plot_file}")

from sklearn.model_selection import cross_val_predict

def plot_global_confusion_matrix(df: pd.DataFrame):
    """
    Uses cross_val_predict to generate a confusion matrix 
    representing the ENTIRE dataset (768 rows).
    """
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    champion_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    
    # Stratified CV ensures each fold has the same ratio of Diabetes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # This generates a 'prediction' for every single row when it was held out
    y_pred_all = cross_val_predict(champion_pipeline, X, y, cv=cv)
    
    # Now we plot using all 768 rows
    disp = ConfusionMatrixDisplay.from_predictions(
        y, 
        y_pred_all, 
        display_labels=['Healthy', 'Diabetic'], 
        cmap='Blues'
    )
    
    plt.title('Global Performance Matrix (All 768 Samples)')
    plt.savefig(os.path.join(PLOT_PATH, 'global_confusion_matrix.png'))
    plt.close()
    
    # Calculate the total counts for your report
    tn, fp, fn, tp = confusion_matrix(y, y_pred_all).ravel()
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
 
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Shows how features relate to each other and the Outcome.
    Essential for understanding why the type 2 are hard to separate.
    """
    plt.figure(figsize=(12, 10))
    # Calculate correlation matrix
    corr = df.corr()
    
    # Generate a mask for the upper triangle (cleaner look)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Clinical Feature Correlation Heatmap')
    
    plot_file = os.path.join(PLOT_PATH, 'feature_correlation.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Correlation Heatmap saved to: {plot_file}")

# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    raw_df = pd.read_csv(DATA_PATH)
    report_content = ["=== DIABETES ML PROJECT SUMMARY ===\n"]

    # We iterate through experiments to compare data leakage vs global integrity
    for use_leakage_mode in [True, False]:
        mode_label = "GROUPED (Leakage Risk)" if use_leakage_mode else "GLOBAL (Real-World)"
        report_content.append(f"\nEXPERIMENT: {mode_label}")
        report_content.append("-" * 30)
        
        print(f"\nRunning Experiment: {mode_label}...")
        
        # 1. Modular Preprocessing
        df_processed = clean_and_impute(raw_df.copy(), use_group_imputation=use_leakage_mode)
        
        # 2. Statistical Analysis
        stats_df = run_stat_tests(df_processed)
        report_content.append(f"Top Predictor: {stats_df.iloc[0]['Feature']} (p={stats_df.iloc[0]['p_value']:.2e})")

        # 3. Modular Model Evaluation (Injecting Tanh for NN)
        configs = get_model_configs(activation_function='tanh')
        results = evaluate_models(df_processed, models=configs)
        
        for model_name, metrics in results.items():
            report_content.append(f"{model_name}: {metrics['Mean']:.2%} (+/- {metrics['Std']:.2%})")
        
        # 4. Global-only Artifacts (Visualizations and diagnostic plots)
        if not use_leakage_mode:
            print("Generating Visuals, Importance, and Loss Plots...")
            generate_visuals(df_processed)
            importance_df = plot_feature_importance(df_processed)
            
            # Use the Global Matrix logic to avoid "unlucky splits"
            # This captures all 768 samples via cross_val_predict
            global_counts = plot_global_confusion_matrix(df_processed)
            
            heatmap = plot_correlation_heatmap(df_processed)
            
            # Diagnostic Loss Curves
            l_relu = plot_mlp_loss_curve(df_processed, activation_type='relu') 
            l_tanh = plot_mlp_loss_curve(df_processed, activation_type='tanh') 
            l_log  = plot_mlp_loss_curve(df_processed, activation_type='logistic') 
            
            # --- UPDATED REPORT SECTION ---
            report_content.append(f"\nGLOBAL ANALYSIS DETAILS (n=768):")
            report_content.append(f"Champion (LogReg) Missed Cases: {global_counts['FN']}")
            report_content.append(f"Champion (LogReg) False Alarms: {global_counts['FP']}")
            report_content.append(f"Final NN Loss (ReLU): {l_relu:.4f}")
            report_content.append(f"Final NN Loss (Tanh): {l_tanh:.4f}")
            report_content.append(f"Final NN Loss (Logistic): {l_log:.4f}")
            report_content.append(f"Top 3 Features: {', '.join(importance_df['Feature'].head(3).tolist())}")
            
            stats_df.to_csv(os.path.join(FINDINGS_PATH, 'clinical_significance.tsv'), sep='\t', index=False)

    # 5. Report Persistence
    report_path = os.path.join(FINDINGS_PATH, 'final_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_content))

    print(f"\n{'='*45}\nPROJECT COMPLETE. Summary saved to: {report_path}\n{'='*45}")

if __name__ == "__main__":
    main()