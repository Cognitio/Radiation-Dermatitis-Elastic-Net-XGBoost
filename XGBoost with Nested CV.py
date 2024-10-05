import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (make_scorer, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
                             f1_score, matthews_corrcoef, cohen_kappa_score, log_loss,
                             balanced_accuracy_score, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import joblib
from scipy.stats import uniform, randint
import shap

# Load data from spreadsheet
data = pd.read_excel('filepath.xlsx')
X = data.iloc[:, 2:-1].values # excludes first two identifier columns that do not contain features
y = data.iloc[:, -1].values # target in last column

# Normalize the features between 0 and 1
X_max = X.max(axis=0)
X = X / X_max

# Define the parameter space for random search
param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc_roc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    kappa = cohen_kappa_score(y_true, y_pred_binary)
    logloss = log_loss(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    
    youdens_j = sensitivity + specificity - 1
    diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf
    lift = (tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn)) if (tp + fp) > 0 else np.inf
    gini = 2 * auc_roc - 1

    return (sensitivity, specificity, ppv, npv, accuracy, auc_roc, brier,
            f1, mcc, kappa, logloss, balanced_acc, auc_pr, youdens_j,
            diagnostic_odds_ratio, lift, gini)

# Nested Cross-Validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Lists to store results
outer_results = []
outer_predictions = np.zeros_like(y, dtype=float)

for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    print(f'Outer Fold {fold+1}/5')
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Inner cross-validation for hyperparameter tuning
    model = XGBClassifier(random_state=42, eval_metric='auc')
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=500,  
        cv=inner_cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring=make_scorer(roc_auc_score)
    )
    
    random_search.fit(X_train, y_train)
    
    print(f'Best parameters for outer fold {fold+1}: {random_search.best_params_}')
    
    # Train model with best parameters on entire training set
    best_model = XGBClassifier(**random_search.best_params_, random_state=42, eval_metric='auc')
    best_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = best_model.predict_proba(X_test)[:, 1]
    outer_predictions[test_index] = y_pred
    
    # Calculate and store metrics
    metrics = calculate_metrics(y_test, y_pred)
    outer_results.append(metrics)

# Calculate overall metrics
overall_metrics = calculate_metrics(y, outer_predictions)

# Print results
metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUC ROC', 'Brier Score',
                'F1 Score', 'Matthews Correlation Coefficient', 'Cohen\'s Kappa', 'Log Loss',
                'Balanced Accuracy', 'AUC Precision-Recall', 'Youden\'s J', 'Diagnostic Odds Ratio',
                'Lift', 'Gini Coefficient']

print("\nOverall Performance (Nested CV):")
for name, value in zip(metric_names, overall_metrics):
    print(f'{name}: {value:.4f}')

# Plot the AUC curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')
fpr, tpr, _ = roc_curve(y, outer_predictions)
plt.plot(fpr, tpr, label=f'AUC = {overall_metrics[5]:.4f}', linewidth=3.0)
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('XGBoost ROC Curve (Nested CV)', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best', fontsize=18)
plt.show()

# Calibration plot
plt.figure(figsize=(10, 8))
fraction_of_positives, mean_predicted_value = calibration_curve(y, outer_predictions, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value", fontsize=20)
plt.ylabel("Fraction of positives", fontsize=20)
plt.title("XGBoost Calibration Plot (Nested CV)", fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best', fontsize=18)
plt.show()

# Train final model on all data using the most common best parameters
all_best_params = [random_search.best_params_ for _ in outer_cv.split(X, y)]
most_common_params = {k: max(set(d[k] for d in all_best_params), key=[d[k] for d in all_best_params].count) 
                      for k in all_best_params[0]}

final_model = XGBClassifier(**most_common_params, random_state=42, eval_metric='auc')
final_model.fit(X, y)

# Save the model and the maximum values used for normalization
joblib.dump({
    'model': final_model,
    'X_max': X_max
}, 'model_xgboost_nested_cv.pkl')

print("Model saved to model_xgboost_nested_cv.pkl")

# SHAP explanations
explainer = shap.Explainer(final_model)
shap_values = explainer(X)

# Plot SHAP summary
shap.summary_plot(shap_values, features=X, feature_names=data.columns[2:-1], plot_size=(8, 8), show=False, color_bar=False)
plt.yticks(fontsize=17, color='black')
plt.xticks(fontsize=16)
plt.xlabel('SHAP value (impact on model output)', fontsize=18)
cbar = plt.colorbar(aspect=40)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Relative Feature Value', fontsize=16)
plt.savefig('XGBoost SHAP Nested CV.png', dpi=1000)
plt.show()
