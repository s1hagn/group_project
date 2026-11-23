import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# 1. Load the cleaned data and select upload-time features
df = pd.read_csv("cleaned_viral_videos.csv")

categorical_features = ["category", "upload_day"]
numeric_features = [
    "duration_sec",
    "title_length",
    "thumbnail_face",
    "upload_hour",
    "tags_count",
    "description_length",
]

X = df[categorical_features + numeric_features]
y = df["viral"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=0)

models = {"Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)}

results = []
predictions = pd.DataFrame({"video_id": df.loc[X_test.index, "video_id"].to_numpy(),"actual_viral": y_test.to_numpy()})

matrix_fig, matrix_axes = plt.subplots(1, 2, figsize=(10, 4.5))
roc_fig, roc_ax = plt.subplots(figsize=(7, 5.5))


# 2. Train and evaluate both models on the same split
for (model_name, model), matrix_ax in zip(models.items(), matrix_axes):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    predicted_class = pipeline.predict(X_test)
    viral_probability = pipeline.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predicted_class, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    roc_auc = roc_auc_score(y_test, viral_probability)

    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, predicted_class),
        "precision": precision_score(y_test, predicted_class, zero_division=0),
        "recall": recall_score(y_test, predicted_class, zero_division=0),
        "f1": f1_score(y_test, predicted_class, zero_division=0),
        "average_precision": average_precision_score(y_test, viral_probability),
        "roc_auc": roc_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    })

    prefix = model_name.lower().replace(" ", "_")
    predictions[f"{prefix}_prediction"] = predicted_class
    predictions[f"{prefix}_viral_probability"] = viral_probability

    # Confusion matrix using predicted classes
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Non-viral", "Viral"],
    ).plot(ax=matrix_ax, cmap="Blues", colorbar=False, values_format="d")

    matrix_ax.set_title(model_name)

    # ROC curve using probabilities
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, viral_probability
    )
    roc_ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )


# 3. Save the two comparison figures
matrix_fig.suptitle("Confusion matrices on the test set")
matrix_fig.tight_layout()
matrix_fig.savefig("ml_confusion_matrices.png", dpi=150)
plt.close(matrix_fig)

roc_ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
roc_ax.set(
    title="ROC curves on the test set",
    xlabel="False positive rate",
    ylabel="True positive rate",
    xlim=(0, 1),
    ylim=(0, 1.02),
)
roc_ax.legend(loc="lower right")
roc_fig.tight_layout()
roc_fig.savefig("ml_roc_comparison.png", dpi=150)
plt.close(roc_fig)


# 4. Export results for Dash and print the model comparison
metrics = pd.DataFrame(results).sort_values(
    "average_precision", ascending=False
)

comparison_columns = [
    "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"
]

print(f"Test-set viral rate: {y_test.mean():.1%}")
print(f"Always predict non-viral accuracy: {1 - y_test.mean():.1%}")
print(metrics.set_index("model")[comparison_columns].round(3).to_string())