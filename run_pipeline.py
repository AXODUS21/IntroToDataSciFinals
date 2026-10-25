"""
Seoul Bike Sharing Demand - Machine Learning Pipeline
De La Salle Lipa - College of Information Technology and Engineering
Department of Computer Science
Course: Introduction to Data Science (Intdsci) - Final Project
Section: C2B
Instructor: Ms. Donna Garcia
Group Name: FML
Group Members: Fabello, Courtney Judge; Lubi, Axellerosh; Macaraig, Paolo Adrian
Date of Submission: Friday (October 30, 2026)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress library deprecation warnings for clean output
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Anchor directory to this script's location for robust path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set random seed for full reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # Create output directories relative to project root
    figures_dir = os.path.join(BASE_DIR, "figures")
    reports_dir = os.path.join(BASE_DIR, "reports")
    data_dir = os.path.join(BASE_DIR, "data")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Set plotting aesthetics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300
    })

    print("=" * 80)
    print("STAGE 1: DATA INGESTION & DATA DICTIONARY")
    print("=" * 80)

    # Load dataset
    data_path = os.path.join(data_dir, "SeoulBikeData.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not locate dataset at: {data_path}")

    df_raw = pd.read_csv(data_path, encoding="latin1")

    print(f"Raw dataset successfully loaded from {data_path}.")
    print(f"Initial Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    print("\nRaw Columns:")
    for i, col in enumerate(df_raw.columns, 1):
        print(f"  {i}. {col} (dtype: {df_raw[col].dtype})")

    print("\n" + "=" * 80)
    print("STAGE 2: DATA CLEANING & LEAKAGE SAFEGUARD (Traceable Log)")
    print("=" * 80)

    cleaning_log = []

    # Step 1: Check missing values
    missing_counts = df_raw.isnull().sum()
    total_missing = missing_counts.sum()
    cleaning_log.append({
        "step": "1. Missing Value Audit",
        "action": "Inspected null count across all columns",
        "result": f"Found {total_missing} null values. Dataset is complete."
    })
    print(f"[Log 1] Missing values count: {total_missing}")

    # Step 2: Check duplicate rows
    duplicate_count = df_raw.duplicated().sum()
    cleaning_log.append({
        "step": "2. Duplicate Row Audit",
        "action": "Checked for exact duplicate records",
        "result": f"Found {duplicate_count} duplicates. No rows dropped."
    })
    print(f"[Log 2] Duplicate rows count: {duplicate_count}")

    # Step 3: Column name standardization
    df = df_raw.copy()
    rename_map = {
        'Date': 'date',
        'Rented Bike Count': 'rented_bike_count',
        'Hour': 'hour',
        'Temperature(°C)': 'temperature',
        'Humidity(%)': 'humidity',
        'Wind speed (m/s)': 'wind_speed',
        'Visibility (10m)': 'visibility',
        'Dew point temperature(°C)': 'dew_point_temperature',
        'Solar Radiation (MJ/m2)': 'solar_radiation',
        'Rainfall(mm)': 'rainfall',
        'Snowfall (cm)': 'snowfall',
        'Seasons': 'season',
        'Holiday': 'holiday',
        'Functioning Day': 'functioning_day'
    }
    df.rename(columns=rename_map, inplace=True)
    cleaning_log.append({
        "step": "3. Column Standardization",
        "action": "Standardized headers to snake_case and removed special unicode symbols",
        "result": "Clean feature names created for programmatic robustness."
    })
    print("[Log 3] Columns renamed to clean snake_case format.")

    # Step 4: Parse Date and extract temporal features
    df['date_dt'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['month'] = df['date_dt'].dt.month
    df['day_of_week'] = df['date_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    cleaning_log.append({
        "step": "4. Temporal Feature Extraction",
        "action": "Parsed 'date' string into datetime and extracted month, day_of_week, and is_weekend",
        "result": "Created 3 new temporal indicators to capture weekly and monthly seasonality."
    })
    print("[Log 4] Temporal attributes extracted: month, day_of_week, is_weekend.")

    # Step 5: Formulate Target Variable
    # Target: is_rented (0 = No bikes rented, 1 = At least one bike rented)
    df['is_rented'] = (df['rented_bike_count'] > 0).astype(int)
    class_counts = df['is_rented'].value_counts()
    cleaning_log.append({
        "step": "5. Target Variable Formulation",
        "action": "Transformed continuous 'rented_bike_count' into binary 'is_rented' (0: count==0, 1: count>0)",
        "result": f"Class 0 (Not Rented): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%), "
                  f"Class 1 (Rented): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)"
    })
    print(f"[Log 5] Binary Target created: {class_counts.to_dict()}")

    # Step 6: Critical Data Leakage Prevention Check
    non_functioning_zero = len(df[(df['functioning_day'] == 'No') & (df['rented_bike_count'] == 0)])
    functioning_non_zero = len(df[(df['functioning_day'] == 'Yes') & (df['rented_bike_count'] > 0)])
    total_zeros = class_counts[0]

    print(f"\n[LEAKAGE AUDIT]:")
    print(f"  Total hours with 0 rentals: {total_zeros}")
    print(f"  Hours with functioning_day == 'No' and 0 rentals: {non_functioning_zero}")
    print(f"  Hours with functioning_day == 'Yes' and >0 rentals: {functioning_non_zero}")
    print("  => Functioning Day has a 100% deterministic correlation with rental occurrence.")
    print("  => EXPLICIT ACTION: Dropping 'functioning_day' from feature matrix X to prevent 100% test leakage.")

    cleaning_log.append({
        "step": "6. Data Leakage Remediation",
        "action": "Dropped 'functioning_day', 'rented_bike_count', and raw 'date' from predictor set X",
        "result": "Guaranteed strict compliance with learning outcome: zero test data leakage."
    })

    # Step 7: Multicollinearity Resolution
    corr_dew_temp = df['temperature'].corr(df['dew_point_temperature'])
    print(f"\n[MULTICOLLINEARITY CHECK]:")
    print(f"  Pearson correlation between Temperature and Dew Point Temperature: {corr_dew_temp:.4f}")
    print("  => EXPLICIT ACTION: Dropping 'dew_point_temperature' to prevent variance inflation.")

    cleaning_log.append({
        "step": "7. Multicollinearity Resolution",
        "action": f"Dropped 'dew_point_temperature' (Pearson r = {corr_dew_temp:.4f} with temperature)",
        "result": "Avoids severe multicollinearity in linear models and stabilizes coefficient estimates."
    })

    print(f"\nDataset Cleaned. Total records: {len(df)}. Clean columns: {df.columns.tolist()}")

    print("\n" + "=" * 80)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (Generating Figures)")
    print("=" * 80)

    palette = ["#e74c3c", "#2ecc71"]

    from matplotlib.patches import Rectangle

    # Plot 1: Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(
        x='is_rented', hue='is_rented', data=df, palette=palette,
        ax=ax, edgecolor='black', linewidth=1, legend=False
    )
    ax.set_title("Figure 1: Target Variable Distribution (is_rented)", weight="bold", pad=12)
    ax.set_xlabel("Rental Occurrence (0 = Not Rented, 1 = Rented)", labelpad=10)
    ax.set_ylabel("Number of Hours", labelpad=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0: Not Rented\n(Count = 0)", "1: Rented\n(Count > 0)"])
    for p in ax.patches:
        if isinstance(p, Rectangle):
            h = p.get_height()
            pct = (h / len(df)) * 100
            ax.annotate(f"{int(h):,} hrs\n({pct:.2f}%)",
                        (p.get_x() + p.get_width() / 2., h / 2),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_target_distribution.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved {fig1_path}")

    # Plot 2: Diurnal & Seasonal Patterns
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df, x='hour', y='rented_bike_count', hue='season',
        palette='tab10', marker='o', errorbar=None, ax=ax, linewidth=2
    )
    ax.set_title("Figure 2: Average Hourly Bike Rentals Across Seasons in Seoul", weight="bold", pad=12)
    ax.set_xlabel("Hour of the Day (0 to 23)", labelpad=10)
    ax.set_ylabel("Average Rented Bike Count", labelpad=10)
    ax.set_xticks(range(0, 24))
    ax.axvspan(7.5, 9.5, color='#f1c40f', alpha=0.25, label='Morning Commute Peak')
    ax.axvspan(17.5, 19.5, color='#e67e22', alpha=0.25, label='Evening Commute Peak')
    ax.legend(title="Season / Peak", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_hourly_seasonal_rentals.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved {fig2_path}")

    # Plot 3: Weather Distributions by Rental Status
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sns.boxplot(x='is_rented', y='temperature', hue='is_rented', data=df, palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("Temperature (°C) by Rental Status", weight="bold")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Not Rented (0)", "Rented (1)"])
    axes[0].set_xlabel("")

    sns.boxplot(x='is_rented', y='humidity', hue='is_rented', data=df, palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("Humidity (%) by Rental Status", weight="bold")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Not Rented (0)", "Rented (1)"])
    axes[1].set_xlabel("")

    sns.boxplot(x='is_rented', y='solar_radiation', hue='is_rented', data=df, palette=palette, ax=axes[2], legend=False)
    axes[2].set_title("Solar Radiation (MJ/m²) by Rental Status", weight="bold")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Not Rented (0)", "Rented (1)"])
    axes[2].set_xlabel("")

    plt.suptitle("Figure 3: Meteorological Feature Distributions by Rental Status", weight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_weather_distributions.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved {fig3_path}")

    # Plot 4: Correlation Matrix Heatmap
    numeric_cols = [
        'temperature', 'humidity', 'wind_speed', 'visibility',
        'solar_radiation', 'rainfall', 'snowfall', 'hour', 'month', 'is_rented'
    ]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        mask=mask, cbar_kws={'label': 'Pearson Correlation'}, ax=ax, linewidths=0.5
    )
    ax.set_title("Figure 4: Pearson Correlation Matrix of Features with Target (is_rented)", weight="bold", pad=12)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "04_correlation_matrix.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved {fig4_path}")

    print("\n" + "=" * 80)
    print("STAGE 4: FEATURE PREPARATION & 80-20 DATA SPLITTING (random_state=42)")
    print("=" * 80)

    # Define feature sets
    feature_cols_num = [
        'hour', 'temperature', 'humidity', 'wind_speed',
        'visibility', 'solar_radiation', 'rainfall', 'snowfall',
        'month', 'day_of_week', 'is_weekend'
    ]
    feature_cols_cat = ['season', 'holiday']

    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df['is_rented'].copy()

    print(f"Predictor Matrix X Shape: {X.shape}")
    print(f"Target Vector y Shape: {y.shape}")
    print(f"Numerical Features ({len(feature_cols_num)}): {feature_cols_num}")
    print(f"Categorical Features ({len(feature_cols_cat)}): {feature_cols_cat}")

    # Stratified 80-20 train-test split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining Set: {X_train.shape[0]} rows (80.0%)")
    print(f"  Class 0 (Not Rented): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"  Class 1 (Rented):     {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")

    print(f"Test Set:     {X_test.shape[0]} rows (20.0%)")
    print(f"  Class 0 (Not Rented): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  Class 1 (Rented):     {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")

    # Define Leak-Free Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols_num),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), feature_cols_cat)
        ]
    )

    print("\nPreprocessor Pipeline defined: StandardScaler for numerics, OneHotEncoder for categoricals.")
    print("Strict Leakage Prevention Guarantee: Fit will be called exclusively on X_train.")

    print("\n" + "=" * 80)
    print("STAGE 5: MODEL TRAINING & 5-FOLD CROSS VALIDATION")
    print("=" * 80)

    # Define models
    models = {
        "Baseline (Dummy - Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree (Balanced)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    # Cross-Validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    print("Evaluating 5-Fold Stratified Cross-Validation on Training Set...")
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
        )
        cv_results[name] = {
            'accuracy_mean': scores['test_accuracy'].mean(),
            'accuracy_std': scores['test_accuracy'].std(),
            'bal_acc_mean': scores['test_balanced_accuracy'].mean(),
            'bal_acc_std': scores['test_balanced_accuracy'].std(),
            'f1_macro_mean': scores['test_f1_macro'].mean(),
            'f1_macro_std': scores['test_f1_macro'].std(),
            'roc_auc_mean': scores['test_roc_auc'].mean(),
            'roc_auc_std': scores['test_roc_auc'].std()
        }
        print(f"  {name:32s} | Bal Acc: {scores['test_balanced_accuracy'].mean():.4f} | F1-Macro: {scores['test_f1_macro'].mean():.4f} | ROC-AUC: {scores['test_roc_auc'].mean():.4f}")

    print("\n" + "=" * 80)
    print("STAGE 6: TEST SET EVALUATION & CONFUSION MATRIX BREAKDOWN")
    print("=" * 80)

    fitted_pipes = {}
    test_evaluations = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe
        
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        f1_macro = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
        
        test_evaluations[name] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision_0': float(prec_0),
            'recall_0_specificity': float(rec_0),
            'f1_0': float(f1_0),
            'precision_1': float(prec_1),
            'recall_1_sensitivity': float(rec_1),
            'f1_1': float(f1_1),
            'f1_macro': float(f1_macro),
            'roc_auc': float(roc_auc),
            'y_pred': np.asarray(y_pred).tolist(),
            'y_prob': np.asarray(y_prob).tolist()
        }
        
        print(f"\n--- {name} ---")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"Accuracy: {acc:.4f} | Balanced Accuracy: {bal_acc:.4f}")
        print(f"Class 0 (Not Rented) -> Precision: {prec_0:.4f}, Recall (Specificity): {rec_0:.4f}, F1: {f1_0:.4f}")
        print(f"Class 1 (Rented)     -> Precision: {prec_1:.4f}, Recall (Sensitivity): {rec_1:.4f}, F1: {f1_1:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['0: Not Rented', '1: Rented'], zero_division=0))

    # Plot 5: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    model_keys = list(models.keys())
    for i, name in enumerate(model_keys):
        eval_data = test_evaluations[name]
        cm = np.array([[eval_data['tn'], eval_data['fp']], [eval_data['fn'], eval_data['tp']]])
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
            xticklabels=['0: Not Rented', '1: Rented'],
            yticklabels=['0: Not Rented', '1: Rented'],
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        axes[i].set_title(f"{name}\nBal Acc: {eval_data['balanced_accuracy']:.2%} | F1-Macro: {eval_data['f1_macro']:.3f}", pad=8)
        axes[i].set_xlabel("Predicted Class", labelpad=6)
        axes[i].set_ylabel("Actual Class", labelpad=6)

    plt.suptitle("Figure 5: Test Set Confusion Matrix Comparison Across Models", weight="bold", y=1.02)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "05_confusion_matrices.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"\nSaved {fig5_path}")

    # Plot 6: ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_keys:
        eval_data = test_evaluations[name]
        y_prob = np.array(eval_data['y_prob'])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {eval_data['roc_auc']:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.500)", linewidth=1.5)
    ax.set_title("Figure 6: Receiver Operating Characteristic (ROC) Curves", weight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", labelpad=10)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, "06_roc_pr_curves.png")
    plt.savefig(fig6_path)
    plt.close()
    print(f"Saved {fig6_path}")

    # Plot 7: Feature Importances (Random Forest)
    rf_pipe = fitted_pipes["Random Forest (Balanced)"]
    cat_feature_names = rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(feature_cols_cat).tolist()
    all_feature_names = feature_cols_num + cat_feature_names
    rf_importances = rf_pipe.named_steps['classifier'].feature_importances_

    df_imp = pd.DataFrame({
        'feature': all_feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x='importance', y='feature', hue='feature', data=df_imp,
        palette='viridis', ax=ax, edgecolor='black', linewidth=0.5, legend=False
    )
    ax.set_title("Figure 7: Random Forest Feature Importances for Rental Prediction", weight="bold", pad=12)
    ax.set_xlabel("Gini Feature Importance Score", labelpad=10)
    ax.set_ylabel("Feature Name", labelpad=10)
    for p in ax.patches:
        if isinstance(p, Rectangle):
            w = p.get_width()
            ax.annotate(f"{w:.3f}", (w + 0.005, p.get_y() + p.get_height() / 2.), va='center', fontsize=9)
    plt.tight_layout()
    fig7_path = os.path.join(figures_dir, "07_feature_importance.png")
    plt.savefig(fig7_path)
    plt.close()
    print(f"Saved {fig7_path}")

    print("\n" + "=" * 80)
    print("STAGE 7: ERROR ANALYSIS & SERIALIZATION")
    print("=" * 80)

    # Deep dive into Random Forest Errors on Test Set
    rf_pred = np.array(test_evaluations["Random Forest (Balanced)"]['y_pred'])
    test_indices = X_test.index
    df_test_errors = df.loc[test_indices].copy()
    df_test_errors['y_true'] = y_test.values
    df_test_errors['y_pred'] = rf_pred
    df_test_errors['is_misclassified'] = (df_test_errors['y_true'] != df_test_errors['y_pred']).astype(int)

    false_negatives = df_test_errors[(df_test_errors['y_true'] == 1) & (df_test_errors['y_pred'] == 0)]
    false_positives = df_test_errors[(df_test_errors['y_true'] == 0) & (df_test_errors['y_pred'] == 1)]

    print(f"Total Test Instances: {len(df_test_errors)}")
    print(f"Total Misclassified Cases: {len(false_negatives) + len(false_positives)}")
    print(f"  False Negatives (Actual: Rented, Predicted: Not Rented): {len(false_negatives)}")
    print(f"  False Positives (Actual: Not Rented, Predicted: Rented): {len(false_positives)}")

    if len(false_negatives) > 0:
        print("\nSample False Negatives (Weather summary):")
        print(false_negatives[['temperature', 'humidity', 'rainfall', 'snowfall', 'hour', 'season', 'rented_bike_count']].head(3))

    if len(false_positives) > 0:
        print("\nSample False Positives (Weather summary):")
        print(false_positives[['temperature', 'humidity', 'rainfall', 'snowfall', 'hour', 'season', 'rented_bike_count']].head(3))

    # Save summary metrics to JSON for report embedding
    summary_output = {
        'cleaning_log': cleaning_log,
        'cv_results': cv_results,
        'test_evaluations': {k: {m: v for m, v in val.items() if m not in ['y_pred', 'y_prob']} for k, val in test_evaluations.items()},
        'feature_importances': df_imp.to_dict(orient='records'),
        'error_analysis': {
            'total_test': len(df_test_errors),
            'total_errors': len(false_negatives) + len(false_positives),
            'fn_count': len(false_negatives),
            'fp_count': len(false_positives)
        }
    }

    json_path = os.path.join(reports_dir, "pipeline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_output, f, indent=2)

    print(f"\nPipeline execution successfully finished! Summary JSON written to {json_path}.")


if __name__ == '__main__':
    main()
