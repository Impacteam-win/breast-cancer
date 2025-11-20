import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Create a comprehensive dataset overview
print("="*60)
print("BREAST CANCER CLASSIFICATION PROJECT")
print("="*60)
print(f"\nDataset Shape: {X.shape}")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Samples: {X.shape[0]}")
print(f"Target Classes: {target_names}")
print(f"Class Distribution:\n{pd.Series(y).value_counts()}")
print("="*60)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Store results
results = {}
conf_matrices = {}
classification_reports = {}
roc_data = {}

# Train, predict, evaluate
print("\nTraining Models...")
for name, model in models.items():
    print(f"  - {name}")
    
    # Use scaled data for LR and KNN, original for DT
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation score
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        'Accuracy': acc,
        'F1 Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'ROC AUC': roc_auc
    }
    conf_matrices[name] = cm
    classification_reports[name] = classification_report(y_test, y_pred, target_names=target_names)
    roc_data[name] = (fpr, tpr, roc_auc)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("\n", results_df.round(4))

# ============================================================================
# VISUALIZATION 1: Comprehensive Model Comparison Dashboard
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=1.00)

# Plot 1: Accuracy and F1 Score
ax1 = axes[0, 0]
results_df[['Accuracy', 'F1 Score']].plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'], width=0.7)
ax1.set_title("Accuracy & F1 Score Comparison", fontweight='bold')
ax1.set_ylabel("Score")
ax1.set_ylim(0.85, 1.0)
ax1.set_xticklabels(results_df.index, rotation=45, ha='right')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Cross-Validation Scores with Error Bars
ax2 = axes[0, 1]
cv_means = results_df['CV Mean']
cv_stds = results_df['CV Std']
x_pos = np.arange(len(results_df))
ax2.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color='#2ecc71', alpha=0.7, edgecolor='black')
ax2.set_title("Cross-Validation Scores (5-Fold)", fontweight='bold')
ax2.set_ylabel("Mean Accuracy")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df.index, rotation=45, ha='right')
ax2.set_ylim(0.85, 1.0)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: ROC AUC Comparison
ax3 = axes[1, 0]
roc_aucs = results_df['ROC AUC']
colors = ['#9b59b6', '#f39c12', '#1abc9c']
bars = ax3.barh(results_df.index, roc_aucs, color=colors, edgecolor='black')
ax3.set_title("ROC AUC Score Comparison", fontweight='bold')
ax3.set_xlabel("ROC AUC Score")
ax3.set_xlim(0.85, 1.0)
ax3.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width - 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', ha='right', va='center', fontweight='bold', color='white')

# Plot 4: Performance Heatmap
ax4 = axes[1, 1]
heatmap_data = results_df[['Accuracy', 'F1 Score', 'ROC AUC']].T
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax4, 
            cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
ax4.set_title("Performance Metrics Heatmap", fontweight='bold')
ax4.set_xlabel("Models")
ax4.set_ylabel("Metrics")

plt.tight_layout()
plt.savefig("./01_model_performance_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 2: Confusion Matrices
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

for idx, (name, cm) in enumerate(conf_matrices.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    ax.set_title(f"{name}", fontweight='bold')
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    
    # Calculate and display metrics on the plot
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    ax.text(0.02, 0.98, f'Precision: {precision:.3f}\nRecall: {recall:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("./02_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 3: ROC Curves
# ============================================================================
plt.figure(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71']
for idx, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
    plt.plot(fpr, tpr, color=colors[idx], lw=2.5,
             label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./03_roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 4: Feature Importance (Decision Tree)
# ============================================================================
dt_model = models["Decision Tree"]
importances = dt_model.feature_importances_
indices = importances.argsort()[::-1][:15]  # Top 15 features

plt.figure(figsize=(12, 6))
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
bars = plt.barh(range(len(indices)), importances[indices], color=colors_gradient, edgecolor='black')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
plt.title("Top 15 Feature Importances - Decision Tree Classifier", fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("./04_feature_importance_decision_tree.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 5: Feature Correlation Heatmap
# ============================================================================
# Create correlation matrix for top features
df = pd.DataFrame(X, columns=feature_names)
top_features_idx = importances.argsort()[::-1][:10]
top_features = [feature_names[i] for i in top_features_idx]
corr_matrix = df[top_features].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap (Top 10 Important Features)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("./05_feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 6: Class Distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Dataset Class Distribution', fontsize=16, fontweight='bold')

# Pie chart
ax1 = axes[0]
class_counts = pd.Series(y).value_counts()
colors_pie = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax1.pie(class_counts, labels=target_names, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title("Overall Distribution", fontweight='bold')

# Bar chart with train/test split
ax2 = axes[1]
train_dist = pd.Series(y_train).value_counts()
test_dist = pd.Series(y_test).value_counts()
x = np.arange(len(target_names))
width = 0.35
bars1 = ax2.bar(x - width/2, train_dist, width, label='Train', color='#2ecc71', edgecolor='black')
bars2 = ax2.bar(x + width/2, test_dist, width, label='Test', color='#f39c12', edgecolor='black')
ax2.set_xlabel('Class', fontweight='bold')
ax2.set_ylabel('Count', fontweight='bold')
ax2.set_title('Train/Test Split Distribution', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(target_names)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("./06_class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Print Classification Reports
# ============================================================================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)
for name, report in classification_reports.items():
    print(f"\n{name}:")
    print("-" * 60)
    print(report)

print("\n" + "="*60)
print("VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*60)
print("\nGenerated Files:")
print("  1. 01_model_performance_dashboard.png")
print("  2. 02_confusion_matrices.png")
print("  3. 03_roc_curves.png")
print("  4. 04_feature_importance_decision_tree.png")
print("  5. 05_feature_correlation_heatmap.png")
print("  6. 06_class_distribution.png")
print("="*60)

# Display results dataframe if ace_tools is available
try:
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Model Performance Summary", dataframe=results_df)
except ImportError:
    pass
