"""
Diabetic Retinopathy Detection using GLCM, HOG, SVM, and PSO
Dataset: APTOS 2019 Blindness Detection

Set APTOS_DATA_DIR to the directory containing:
    train.csv
    train_images/
Example:
    Windows PowerShell:
        $env:APTOS_DATA_DIR = "C:\datasets\aptos2019-blindness-detection"
    Linux/macOS:
        export APTOS_DATA_DIR="/path/to/aptos2019-blindness-detection"
"""

import os
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
N_PER_CLASS = 150
IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.20

# Set this environment variable to your local APTOS dataset directory.
BASE_DIR = Path(
    os.environ.get(
        "APTOS_DATA_DIR",
        "/kaggle/input/competitions/aptos2019-blindness-detection",
    )
)
CSV_PATH = BASE_DIR / "train.csv"
IMAGE_DIR = BASE_DIR / "train_images"

LOG_C_RANGE = (-2, 2)
LOG_GAMMA_RANGE = (-4, 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("diabetic-retinopathy")


# ---------------------------------------------------------------------
# 1. Load labels and create a balanced binary dataset
# ---------------------------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {CSV_PATH}. Set APTOS_DATA_DIR to the folder "
        "containing train.csv and train_images/."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {"id_code", "diagnosis"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"train.csv is missing required columns: {missing_columns}")

# Diagnosis 0 -> No DR; diagnosis 1-4 -> DR.
df["label"] = (df["diagnosis"] > 0).astype(int)

parts = []
for _, group in df.groupby("label"):
    parts.append(
        group.sample(
            n=min(len(group), N_PER_CLASS),
            random_state=RANDOM_STATE,
        )
    )

df = (
    pd.concat(parts)
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

log.info("Selected %d images from the CSV.", len(df))


# ---------------------------------------------------------------------
# 2. Load and preprocess images
# ---------------------------------------------------------------------

images = []
labels = []
missing_images = []

for _, row in df.iterrows():
    image_path = IMAGE_DIR / f"{row['id_code']}.png"
    image = cv2.imread(str(image_path))

    if image is None:
        missing_images.append(str(image_path))
        continue

    # OpenCV loads BGR; convert to RGB before rgb2gray.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)

    images.append(image)
    labels.append(int(row["label"]))

if missing_images:
    log.warning("%d image files could not be loaded.", len(missing_images))
    for missing_path in missing_images[:5]:
        log.warning("Missing image: %s", missing_path)

if not images:
    raise RuntimeError(
        f"No images were loaded. Check that image files exist in {IMAGE_DIR}."
    )

images = np.asarray(images)
labels = np.asarray(labels)

log.info(
    "Loaded %d images. Class balance [no DR, DR]: %s",
    len(images),
    np.bincount(labels, minlength=2),
)


# ---------------------------------------------------------------------
# 3. Feature extraction: GLCM + HOG
# ---------------------------------------------------------------------

feature_list = []

for img in images:
    gray = rgb2gray(img)
    gray = (gray * 255).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_features = [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
    ]

    hog_features = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    feature_list.append(glcm_features + hog_features.tolist())

X = np.asarray(feature_list)
y = labels

log.info("Feature matrix shape: %s", X.shape)


# ---------------------------------------------------------------------
# 4. Train/test split and feature scaling
# ---------------------------------------------------------------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Fit scaler only on training data to avoid test-set leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

log.info("Train class balance: %s", np.bincount(y_train, minlength=2))
log.info("Test class balance: %s", np.bincount(y_test, minlength=2))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------------------
# 5. Baseline SVM with GridSearchCV
# ---------------------------------------------------------------------

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"],
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

baseline_model = grid.best_estimator_
y_pred_base = baseline_model.predict(X_test)
y_score_base = baseline_model.decision_function(X_test)

acc_base = accuracy_score(y_test, y_pred_base)

print("\nBASELINE SVM")
print("Best parameters:", grid.best_params_)
print(f"Accuracy: {acc_base:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_base))


# ---------------------------------------------------------------------
# 6. Particle Swarm Optimization (PSO)
# ---------------------------------------------------------------------

def sigmoid(x):
    """Convert real-valued particle positions to probabilities."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def decode_particle(position, n_features):
    """
    Decode a particle into:
      - binary feature mask
      - SVM C
      - SVM gamma
      - SVM kernel
    """
    mask_raw = position[:n_features]
    log_c = position[n_features]
    log_gamma = position[n_features + 1]
    kernel_val = position[n_features + 2]

    mask = sigmoid(mask_raw) > 0.5

    # Ensure that at least one feature is selected.
    if not mask.any():
        mask = np.zeros(n_features, dtype=bool)
        mask[np.argmax(mask_raw)] = True

    C = 10 ** np.clip(log_c, *LOG_C_RANGE)
    gamma = 10 ** np.clip(log_gamma, *LOG_GAMMA_RANGE)
    kernel = "rbf" if kernel_val >= 0.5 else "linear"

    return mask, C, gamma, kernel


def fitness_function(swarm_positions, feature_penalty=0.001):
    """
    Evaluate particles using cross-validation on training data only.

    The test set is not used during optimization.
    Lower fitness is better.
    """
    n_features = X_train.shape[1]
    costs = np.zeros(swarm_positions.shape[0])

    for i, position in enumerate(swarm_positions):
        mask, C, gamma, kernel = decode_particle(position, n_features)
        X_sub = X_train[:, mask]

        try:
            scores = cross_val_score(
                SVC(C=C, gamma=gamma, kernel=kernel),
                X_sub,
                y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
            )
            cv_accuracy = scores.mean()
        except Exception as exc:
            log.warning("Particle evaluation failed: %s", exc)
            cv_accuracy = 0.0

        fraction_used = mask.sum() / n_features
        costs[i] = (1 - cv_accuracy) + feature_penalty * fraction_used

    return costs


n_features = X_train.shape[1]
dimensions = n_features + 3  # Feature-mask dimensions + C + gamma + kernel.

lower_bounds = np.concatenate(
    [
        np.full(n_features, -3.0),
        [LOG_C_RANGE[0]],
        [LOG_GAMMA_RANGE[0]],
        [0.0],
    ]
)
upper_bounds = np.concatenate(
    [
        np.full(n_features, 3.0),
        [LOG_C_RANGE[1]],
        [LOG_GAMMA_RANGE[1]],
        [1.0],
    ]
)

options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

optimizer = ps.single.GlobalBestPSO(
    n_particles=15,
    dimensions=dimensions,
    options=options,
    bounds=(lower_bounds, upper_bounds),
)

best_cost, best_position = optimizer.optimize(
    fitness_function,
    iters=20,
)

mask, best_C, best_gamma, best_kernel = decode_particle(
    best_position,
    n_features,
)

print("\nPSO OPTIMIZATION")
print(f"Best fitness: {best_cost:.6f}")
print(f"Best C: {best_C:.6f}")
print(f"Best gamma: {best_gamma:.6f}")
print(f"Best kernel: {best_kernel}")
print(f"Selected features: {mask.sum()} / {n_features}")


# ---------------------------------------------------------------------
# 7. Train and evaluate final PSO-optimized SVM
# ---------------------------------------------------------------------

final_model = SVC(
    C=best_C,
    gamma=best_gamma,
    kernel=best_kernel,
)

final_model.fit(X_train[:, mask], y_train)

y_pred_final = final_model.predict(X_test[:, mask])
y_score_final = final_model.decision_function(X_test[:, mask])

acc_final = accuracy_score(y_test, y_pred_final)

print("\nPSO-OPTIMIZED SVM")
print(f"Accuracy: {acc_final:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred_final,
        target_names=["No DR", "DR"],
        zero_division=0,
    )
)


# ---------------------------------------------------------------------
# 8. Sensitivity and specificity
# ---------------------------------------------------------------------

def sensitivity_specificity(y_true, y_pred):
    """Calculate sensitivity and specificity for binary classification."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


sens_base, spec_base = sensitivity_specificity(y_test, y_pred_base)
sens_pso, spec_pso = sensitivity_specificity(y_test, y_pred_final)

print("\nMODEL COMPARISON")
print(f"{'Metric':<20}{'Baseline':>15}{'PSO-Optimized':>18}")
print("-" * 53)
print(f"{'Accuracy':<20}{acc_base:>15.4f}{acc_final:>18.4f}")
print(f"{'Sensitivity':<20}{sens_base:>15.4f}{sens_pso:>18.4f}")
print(f"{'Specificity':<20}{spec_base:>15.4f}{spec_pso:>18.4f}")
print(f"{'Features used':<20}{n_features:>15d}{int(mask.sum()):>18d}")


# ---------------------------------------------------------------------
# 9. PSO convergence graph
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(optimizer.cost_history)
plt.title("PSO Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost (1 - CV accuracy + feature penalty)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 10. ROC curves: baseline vs PSO-optimized SVM
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 6))

for model_name, y_score in [
    ("Baseline SVM", y_score_base),
    ("PSO-Optimized SVM", y_score_final),
]:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    color="gray",
    label="Chance",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Baseline vs PSO-Optimized SVM")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
