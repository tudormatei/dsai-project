"""
Model Comparison Visualization
Comparing performance of different models for concept-to-activity prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available, skipping...")
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
df_windows = pd.read_csv('./data/final_window_labels.csv')

discrete_columns = ['periodicity', 'temporal_stability', 'coordination']
continuous_columns = ['movement_variability', 'movement_consistency']

# Check which columns exist
available_continuous = [col for col in continuous_columns if col in df_windows.columns]
feature_columns = discrete_columns + available_continuous

X_final = df_windows[feature_columns].values

y_label_str = df_windows['activity'].values
activity_mapping = {act: i for i, act in enumerate(np.unique(y_label_str))}
reverse_mapping = {v: k for k, v in activity_mapping.items()}
y_label = np.array([activity_mapping[act] for act in y_label_str])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_label, test_size=0.25, random_state=42, stratify=y_label
)

# Helper function to calculate AUROC
def calculate_auroc(y_true, y_pred_proba, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if y_true_bin.shape[1] == 1:
        return roc_auc_score(y_true_bin, y_pred_proba[:, 1])
    else:
        return roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')

num_classes = len(activity_mapping)
activity_names = [reverse_mapping[i] for i in range(num_classes)]

print(f"Activities: {activity_names}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# Initialize results storage
results = []

# Train all models and store results
print("Training models...\n")

# 1. Decision Tree
print("1/5 Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)
results.append({
    'model': 'Decision Tree',
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'auroc': calculate_auroc(y_test, y_pred_proba_dt, num_classes),
    'predictions': y_pred_dt,
    'probabilities': y_pred_proba_dt
})

# 2. Logistic Regression
print("2/5 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)
results.append({
    'model': 'Logistic Regression',
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'auroc': calculate_auroc(y_test, y_pred_proba_lr, num_classes),
    'predictions': y_pred_lr,
    'probabilities': y_pred_proba_lr
})

# 3. Random Forest
print("3/5 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2,
                           min_samples_leaf=1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)
results.append({
    'model': 'Random Forest',
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'auroc': calculate_auroc(y_test, y_pred_proba_rf, num_classes),
    'predictions': y_pred_rf,
    'probabilities': y_pred_proba_rf
})

# 4. XGBoost
if HAS_XGBOOST:
    print("4/5 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                       subsample=0.9, colsample_bytree=0.9, random_state=42,
                       use_label_encoder=False, eval_metric='mlogloss', verbose=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_proba_xgb = xgb.predict_proba(X_test)
    results.append({
        'model': 'XGBoost',
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'auroc': calculate_auroc(y_test, y_pred_proba_xgb, num_classes),
        'predictions': y_pred_xgb,
        'probabilities': y_pred_proba_xgb
    })
else:
    xgb = None

# 5. SVM
if HAS_XGBOOST:
    print("5/5 Training SVM...")
else:
    print("4/4 Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_pred_proba_svm = svm.predict_proba(X_test)
results.append({
    'model': 'SVM (RBF)',
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'auroc': calculate_auroc(y_test, y_pred_proba_svm, num_classes),
    'predictions': y_pred_svm,
    'probabilities': y_pred_proba_svm
})

print("\nModel training complete!\n")

# Convert results to DataFrame for easier analysis
df_results = pd.DataFrame(results)

# Create comprehensive visualization
print("Creating visualizations...")

fig = plt.figure(figsize=(20, 16))

# 1. Accuracy Comparison (Top Left)
ax1 = plt.subplot(3, 3, 1)
df_sorted = df_results.sort_values('accuracy', ascending=False)
colors_accuracy = ['#2ecc71' if x == df_sorted['accuracy'].max() else '#3498db' for x in df_sorted['accuracy']]
bars = ax1.barh(df_sorted['model'], df_sorted['accuracy'], color=colors_accuracy)
ax1.set_xlabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 0.7)
ax1.grid(True, alpha=0.3, axis='x')
# Add value labels
for i, v in enumerate(df_sorted['accuracy']):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

# 2. AUROC Comparison (Top Middle)
ax2 = plt.subplot(3, 3, 2)
df_sorted_auroc = df_results.sort_values('auroc', ascending=False)
colors_auroc = ['#2ecc71' if x == df_sorted_auroc['auroc'].max() else '#9b59b6' for x in df_sorted_auroc['auroc']]
bars = ax2.barh(df_sorted_auroc['model'], df_sorted_auroc['auroc'], color=colors_auroc)
ax2.set_xlabel('AUROC', fontsize=12)
ax2.set_title('AUROC Comparison', fontsize=14, fontweight='bold')
ax2.set_xlim(0.65, 0.9)
ax2.grid(True, alpha=0.3, axis='x')
# Add value labels
for i, v in enumerate(df_sorted_auroc['auroc']):
    ax2.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=10)

# 3. Accuracy vs AUROC Scatter (Top Right)
ax3 = plt.subplot(3, 3, 3)
for idx, row in df_results.iterrows():
    ax3.scatter(row['accuracy'], row['auroc'], s=150, alpha=0.7, label=row['model'])
ax3.set_xlabel('Accuracy', fontsize=12)
ax3.set_ylabel('AUROC', fontsize=12)
ax3.set_title('Accuracy vs AUROC', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# 4. Confusion Matrix - Random Forest (Best Accuracy) (Middle Left)
ax4 = plt.subplot(3, 3, 4)
best_idx = df_results['accuracy'].idxmax()
best_model_name = df_results.loc[best_idx, 'model']
best_predictions = df_results.loc[best_idx, 'predictions']
cm_best = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=activity_names,
           yticklabels=activity_names, ax=ax4)
ax4.set_title(f'{best_model_name} Confusion Matrix\n(Best Accuracy: {df_results.loc[best_idx, "accuracy"]:.3f})',
             fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=11)
ax4.set_ylabel('Actual', fontsize=11)

# 5. Confusion Matrix - Logistic Regression (Best AUROC) (Middle Middle)
ax5 = plt.subplot(3, 3, 5)
best_auroc_idx = df_results['auroc'].idxmax()
best_auroc_model = df_results.loc[best_auroc_idx, 'model']
best_auroc_predictions = df_results.loc[best_auroc_idx, 'predictions']
cm_auroc = confusion_matrix(y_test, best_auroc_predictions)
sns.heatmap(cm_auroc, annot=True, fmt='d', cmap='Greens', xticklabels=activity_names,
           yticklabels=activity_names, ax=ax5)
ax5.set_title(f'{best_auroc_model} Confusion Matrix\n(Best AUROC: {df_results.loc[best_auroc_idx, "auroc"]:.3f})',
             fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=11)
ax5.set_ylabel('Actual', fontsize=11)

# 6. Per-Class Precision-Recall (Middle Right)
ax6 = plt.subplot(3, 3, 6)
best_model_preds = df_results.loc[best_idx, 'predictions']
report = classification_report(y_test, best_model_preds, target_names=activity_names,
                              output_dict=True, zero_division=0)
precisions = [report[name]['precision'] for name in activity_names]
recalls = [report[name]['recall'] for name in activity_names]
f1_scores = [report[name]['f1-score'] for name in activity_names]

x_pos = np.arange(len(activity_names))
width = 0.25
ax6.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
ax6.bar(x_pos, recalls, width, label='Recall', alpha=0.8)
ax6.bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
ax6.set_xlabel('Activity Class', fontsize=11)
ax6.set_ylabel('Score', fontsize=11)
ax6.set_title(f'{best_model_name} - Per-Class Performance', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(activity_names, rotation=45, ha='right', fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim(0, 1.1)

# 7. Training vs Test Accuracy (Bottom Left)
ax7 = plt.subplot(3, 3, 7)
# Calculate training accuracies
model_objects = {'Decision Tree': dt, 'Logistic Regression': lr, 'Random Forest': rf,
                'SVM (RBF)': svm}
if HAS_XGBOOST:
    model_objects['XGBoost'] = xgb
train_accs = []
for idx, row in df_results.iterrows():
    model_name = row['model']
    model_obj = model_objects[model_name]
    train_preds = model_obj.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_accs.append(train_acc)

models_names_short = [r['model'] for r in results]
x_pos = np.arange(len(models_names_short))
width = 0.35
ax7.bar(x_pos - width/2, df_results['accuracy'], width, label='Test Accuracy', alpha=0.8)
ax7.bar(x_pos + width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
ax7.set_xticks(x_pos)
ax7.set_xticklabels([m.split()[0] for m in models_names_short], rotation=45, ha='right', fontsize=9)
ax7.set_ylabel('Accuracy', fontsize=11)
ax7.set_title('Training vs Test Accuracy', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Model Rankings (Bottom Middle)
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')
# Create ranking table
ranking_data = []
for idx, row in df_results.iterrows():
    ranking_data.append([row['model'], f"{row['accuracy']:.3f}", f"{row['auroc']:.3f}"])

df_sorted_both = df_results.sort_values('accuracy', ascending=False)
table_data = []
for idx, row in df_sorted_both.iterrows():
    table_data.append([row['model'], f"{row['accuracy']:.3f}", f"{row['auroc']:.3f}"])

table = ax8.table(cellText=table_data, colLabels=['Model', 'Accuracy', 'AUROC'],
                 cellLoc='center', loc='center', colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Highlight best
for i in range(len(table_data)):
    if i == 0:  # Best accuracy
        for j in range(3):
            table[(i+1, j)].set_facecolor('#2ecc71')
            table[(i+1, j)].set_text_props(weight='bold', color='white')

ax8.set_title('Model Ranking (by Accuracy)', fontsize=12, fontweight='bold', pad=20)

# 9. Feature Importance (Bottom Right) - from Random Forest
ax9 = plt.subplot(3, 3, 9)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

ax9.barh(range(len(feature_columns)), importances[indices], color='#e74c3c', alpha=0.8)
ax9.set_yticks(range(len(feature_columns)))
ax9.set_yticklabels([feature_columns[i] for i in indices], fontsize=10)
ax9.set_xlabel('Importance Score', fontsize=11)
ax9.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='x')
# Add value labels
for i, v in enumerate(importances[indices]):
    ax9.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.suptitle('Model Comparison: Concept to Activity Classification', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save the figure
output_path = './model_comparison_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"\n{'Model':<25} {'Accuracy':<12} {'AUROC':<12}")
print("-"*50)
for _, row in df_results.sort_values('accuracy', ascending=False).iterrows():
    print(f"{row['model']:<25} {row['accuracy']:<12.3f} {row['auroc']:<12.3f}")

print(f"\n🏆 Best Accuracy: {df_results.loc[best_idx, 'model']} ({df_results.loc[best_idx, 'accuracy']:.3f})")
print(f"🏆 Best AUROC: {df_results.loc[best_auroc_idx, 'model']} ({df_results.loc[best_auroc_idx, 'auroc']:.3f})")
print("\n" + "="*80)

plt.show()

