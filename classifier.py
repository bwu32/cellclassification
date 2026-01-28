import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn import __version__ as sklearn_version
from packaging import version

# -------------------------
# Load dataset
# -------------------------
csv_path = "Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"
df = pd.read_csv(csv_path)

X_df = df.drop(["Cell_ID", "Cell_Type", "Disease_Status"], axis=1)
y_cell = df["Cell_Type"].astype(str)
y_status = df["Disease_Status"].astype(str)

X_train_df, X_test_df, y_cell_train, y_cell_test, y_status_train, y_status_test = train_test_split(
    X_df, y_cell, y_status, test_size=0.38,stratify=y_cell
)

X_train = X_train_df.values
X_test = X_test_df.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Base learners
# -------------------------
base_learners = {
    "Decision Stump": DecisionTreeClassifier(max_depth=1),
    "Weak Tree": DecisionTreeClassifier(max_depth=3),
    "Logistic Regression": LogisticRegression(max_iter=600),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

# -------------------------
# AdaBoost wrapper
# -------------------------
def train_adaboost(base_model, X_tr, y_tr, n_estimators=50, lr=0.8):
    try:
        if version.parse(sklearn_version) >= version.parse("1.2"):
            ada = AdaBoostClassifier(
                estimator=base_model, n_estimators=n_estimators,
                learning_rate=lr
            )
        else:
            ada = AdaBoostClassifier(
                base_estimator=base_model, n_estimators=n_estimators,
                learning_rate=lr
            )
        ada.fit(X_tr, y_tr)
        return ada
    except Exception:
        base_model.fit(X_tr, y_tr)
        return base_model

# -------------------------
# Layer-1 meta-features
# -------------------------
meta_train = pd.DataFrame(index=X_train_df.index)
meta_test = pd.DataFrame(index=X_test_df.index)

for name, model in base_learners.items():
    use_scaled = name in ["Logistic Regression", "Naive Bayes", "Linear SVM"]
    Xt = X_train_scaled if use_scaled else X_train
    Xs = X_test_scaled if use_scaled else X_test

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    oof = cross_val_predict(model, Xt, y_cell_train, cv=cv, method="predict")
    test_pred = model.fit(Xt, y_cell_train).predict(Xs)

    meta_train[f"l1_{name}"] = pd.Categorical(oof)
    meta_test[f"l1_{name}"] = pd.Categorical(test_pred)

# Majority vote
def maj(row):
    vals, counts = np.unique(row.values, return_counts=True)
    return vals[np.argmax(counts)]

meta_train["maj_pred"] = meta_train.astype(str).apply(maj, axis=1)
meta_test["maj_pred"] = meta_test.astype(str).apply(maj, axis=1)

# Encode meta-features
for col in meta_train.columns:
    le = LabelEncoder()
    combined = pd.concat([meta_train[col].astype(str), meta_test[col].astype(str)])
    le.fit(combined)
    meta_train[col + "_enc"] = le.transform(meta_train[col].astype(str))
    meta_test[col + "_enc"] = le.transform(meta_test[col].astype(str))

encoded_cols = [c for c in meta_train.columns if c.endswith("_enc")]
meta_train_enc = meta_train[encoded_cols].to_numpy()
meta_test_enc = meta_test[encoded_cols].to_numpy()

X_train_L2 = np.hstack([X_train_scaled, meta_train_enc])
X_test_L2  = np.hstack([X_test_scaled,  meta_test_enc])

# -------------------------
# Accuracy storage
# -------------------------
results = []

# Layer-1 accuracy
layer1_acc = accuracy_score(y_cell_test, meta_test["maj_pred"])
results.append({"Layer": "Layer-1", "Model": "MajorityVote", "Accuracy": layer1_acc})

mask_train_nc = (y_cell_train != "Cancer")
X_train_nc = X_train_L2[mask_train_nc.values]
y_train_nc = y_status_train[mask_train_nc]

mask_test_nc = (meta_test["maj_pred"] != "Cancer")
X_test_nc = X_test_L2[mask_test_nc.values]
y_test_nc = y_status_test[mask_test_nc]

final_pred = pd.Series(index=y_status_test.index, dtype=object)
final_pred.loc[~mask_test_nc] = "Tumor"

# Store Layer-2 predictions for majority vote
layer2_preds = pd.DataFrame(index=y_test_nc.index)

for name, model in base_learners.items():
    if name == "Linear SVM":
        n_est = 20
        lr = 0.3
    else:
        n_est = 50
        lr = 0.5

    clf = train_adaboost(model, X_train_nc, y_train_nc, n_estimators=n_est, lr=lr)
    preds_nc = clf.predict(X_test_nc)

    # Store predictions for majority vote
    layer2_preds[name] = preds_nc

    # Individual classifier results
    test_nc_indices = y_status_test.index[mask_test_nc.values]
    temp_final_pred = final_pred.copy()
    temp_final_pred.loc[test_nc_indices] = preds_nc

    acc = accuracy_score(y_status_test, temp_final_pred)
    acc_nc = accuracy_score(y_test_nc, preds_nc)
    results.append({"Layer": "Layer-2", "Model": name, "Accuracy": acc, "Accuracy_NonCancer": acc_nc})

# -------------------------
# Layer-2 majority vote
# -------------------------
def maj(row):
    vals, counts = np.unique(row.values, return_counts=True)
    return vals[np.argmax(counts)]

final_pred_nc_majority = layer2_preds.apply(maj, axis=1)
final_pred_majority = final_pred.copy()
final_pred_majority.loc[mask_test_nc] = final_pred_nc_majority

# Accuracy of Layer-2 majority vote
acc_majority = accuracy_score(y_status_test, final_pred_majority)
acc_nc_majority = accuracy_score(y_test_nc, final_pred_nc_majority)
results.append({"Layer": "Layer-2", "Model": "MajorityVote",
                "Accuracy": acc_majority, "Accuracy_NonCancer": acc_nc_majority})
results_df = pd.DataFrame(results)

# -------------------------
# Print accuracy table
# -------------------------
print("\nAccuracy Summary:")
print(results_df[["Layer", "Model", "Accuracy", "Accuracy_NonCancer"]])

# -------------------------
# Plot accuracies
# -------------------------
plt.figure(figsize=(10,6))
sns.barplot(data=results_df[results_df['Layer']=='Layer-2'],
            x='Model', y='Accuracy', palette="viridis")
plt.title("Layer-2 Disease-Status Accuracy by Base Learner")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.xlabel("Base Learner")
plt.xticks(rotation=15)
plt.show()
