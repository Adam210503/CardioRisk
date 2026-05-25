import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. DATA ACQUISITION (Bypassing the expired UCI server certificate block directly)
ssl._create_default_https_context = ssl._create_unverified_context

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=columns, na_values="?")

# Handle missing values natively with Pandas medians before splitting
df = df.fillna(df.median())

# Separate features (X) and raw labels (y)
X = df.drop(columns=['target'])
y_raw = df['target'].values.ravel()

# 2. RETAIN THE 5 ORIGINAL CLASSES (0: Absence, 1-4: Varying Severity Levels)
y_original = y_raw.copy()

print("\n--- Target Variable Distribution (5 Original Classes) ---")
unique, counts = np.unique(y_original, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u} (Severity Level {u}): {c} patients")

# 3. CROSS-VALIDATION LOOP (BASELINE MODELS)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler() 

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000), # Increased max_iter for 5-class convergence
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), 
    "Gradient Boosting": GradientBoostingClassifier(random_state=42), 
    "Hist Gradient Boosting (XGB Alternative)": HistGradientBoostingClassifier(random_state=42) 
}

print("\n--- Stratified 5-Fold Cross-Validation Results ---")
print(f"{'Model Name':<42} | {'Avg Accuracy':<14}| {'Avg Precision':<12} | {'Avg Recall':<11} | {'Avg F1-Score':<11}")
print("-" * 120)

for name, model in models.items():
    fold_accuracies, fold_recalls, fold_f1s, fold_precisions = [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y_original):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] 
        y_tr, y_te = y_original[train_idx], y_original[test_idx]
        
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_te_scaled)
        
        fold_accuracies.append(accuracy_score(y_te, preds))
        fold_recalls.append(recall_score(y_te, preds, average='macro'))
        fold_f1s.append(f1_score(y_te, preds, average='macro'))
        fold_precisions.append(precision_score(y_te, preds, average='macro'))
        
    avg_acc = np.mean(fold_accuracies)
    avg_recall = np.mean(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    avg_precision = np.mean(fold_precisions)
    
    print(f"{name:<42} | {avg_f1:<14.2%} | {avg_acc:<13.2%} |  {avg_precision:<11.2%} | {avg_recall:<11.2%}") 

# 3. EXHAUSTIVE HYPERPARAMETER TUNING (CRITICAL FIX: Guiding GridSearch via Macro F1)
X_scaled = scaler.fit_transform(X)

# SMOTE adjustment to handle minority class size constraint (Class 4 has only 13 samples)
smote = SMOTE(k_neighbors=2, random_state=42) #13 class 4 divi by 5 folds, so k_neighbors must be less than 13 to avoid errors during resampling
X_balanced_train, y_balanced_train = smote.fit_resample(X_scaled, y_original) #each class will have 164 samples

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=skf,
    scoring='f1_macro', # primary metric for optimizing the best possible Macro F1 score
    n_jobs=-1
)
grid_search.fit(X_balanced_train, y_balanced_train)

print(f" Best Random Forest Parameters: {grid_search.best_params_}")
print(f" Best Cross-Validated Macro F1-Score: {grid_search.best_score_:.2%}")

# 4. MULTI-CLASS SMOTE & CALIBRATED RISK PIPELINE USING OPTIMAL ESTIMATORS
print("\n--- Running Balanced & Calibrated Risk Pipeline ---")

# Extract optimized parameters dynamically from the grid search results
best_params = grid_search.best_params_

native_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('calibrated_model', CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        ),
        method='sigmoid', 
        cv=5               
    ))
])

# Generate out-of-fold predictions on balanced space
X_balanced, y_balanced = smote.fit_resample(X, y_original)
y_pred = cross_val_predict(native_pipeline, X_balanced, y_balanced, cv=skf)

print(" Final Classification Report (5 Original Severity Classes After Tuning):")
print(classification_report(y_balanced, y_pred, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))

final_pipeline = native_pipeline.fit(X_balanced, y_balanced)
joblib.dump(final_pipeline, 'optimized_heart_disease_model.pkl')