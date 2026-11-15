# Diabetic Retinopathy Detection Using GLCM, HOG, SVM and PSO

## Project Overview

This project implements a machine-learning pipeline for binary diabetic retinopathy (DR) classification using retinal fundus images from the **APTOS 2019 Blindness Detection** dataset.

The pipeline combines:
- **GLCM (Gray-Level Co-occurrence Matrix)** for texture feature extraction.
- **HOG (Histogram of Oriented Gradients)** for gradient and edge feature extraction.
- **SVM (Support Vector Machine)** for classification.
- **PSO (Particle Swarm Optimization)** for feature selection and SVM hyperparameter/kernel optimization.

The labels are converted into two classes:
- `0`: No diabetic retinopathy (original diagnosis = 0)
- `1`: Diabetic retinopathy (original diagnosis > 0)

> **Disclaimer:** This is an educational/research project, not a medical device. It must not be used to diagnose patients or guide clinical decisions.

## Dataset

**APTOS 2019 Blindness Detection**  
[Kaggle competition and dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

Download the dataset from Kaggle. The code expects this directory structure:

```text
aptos2019-blindness-detection/
├── train.csv
└── train_images/
    ├── image_id_1.png
    ├── image_id_2.png
    └── ...
```

The script samples up to 150 images from each binary class, giving a maximum of 300 images. The actual number may be lower if the dataset contains fewer examples in a class or image files are missing.

## Methodology

### 1. Image preprocessing
- Read retinal images with OpenCV.
- Convert images from BGR to RGB.
- Resize images to `128 × 128`.
- Convert images to grayscale for feature extraction.

### 2. Feature extraction

**GLCM:** Four texture features are extracted:
- Contrast
- Energy
- Homogeneity
- Correlation

**HOG:** Captures local gradient and edge patterns using:
- Pixels per cell: `(16, 16)`
- Cells per block: `(2, 2)`

The GLCM and HOG features are concatenated into one feature vector. In the reported experiment, this produced **1,768 features per image**.

### 3. Data splitting and scaling
- Training/testing split: 80% / 20%.
- Stratified splitting preserves the class proportions.
- `StandardScaler` is fitted on the training set only, then applied to the test set.

### 4. Baseline SVM
`GridSearchCV` with 5-fold stratified cross-validation searches over:
- `C`: `[0.1, 1, 10, 100]`
- `gamma`: `[0.0001, 0.001, 0.01, 0.1, 1]`
- `kernel`: `['rbf', 'linear']`

### 5. PSO optimization
Particle Swarm Optimization searches for:
- A binary feature-selection mask.
- The SVM `C` parameter.
- The SVM `gamma` parameter.
- The SVM kernel (`linear` or `rbf`).

PSO configuration:
- Number of particles: `15`
- Iterations: `20`
- Cognitive coefficient (`c1`): `1.5`
- Social coefficient (`c2`): `1.5`
- Inertia weight (`w`): `0.7`

The fitness function uses cross-validation on the training set:

```text
Fitness = (1 - CV accuracy) + (feature penalty × fraction of selected features)
```

The test set is held out from the PSO fitness evaluation and is used for final evaluation.

## Results

The following values are from the recorded run included with this project:

| Metric | Baseline SVM | PSO-Optimized SVM |
|---|---:|---:|
| Accuracy | 93.33% | **95.00%** |
| Sensitivity | 90.00% | 90.00% |
| Specificity | 96.67% | **100.00%** |
| Number of features | 1,768 | **874** |

### Confusion matrices

**Baseline SVM**

| | Predicted: No DR | Predicted: DR |
|---|---:|---:|
| Actual: No DR | 29 | 1 |
| Actual: DR | 3 | 27 |

**PSO-Optimized SVM**

| | Predicted: No DR | Predicted: DR |
|---|---:|---:|
| Actual: No DR | 30 | 0 |
| Actual: DR | 3 | 27 |

In this run, PSO increased accuracy from 93.33% to 95.00% and reduced the selected feature count from 1,768 to 874 (approximately 50.6% fewer features). Sensitivity remained 90%.

**Important evaluation note:** These results come from a balanced subset of 300 images and a single stratified 80/20 train-test split (60 test images). They should not be interpreted as evidence of clinical performance or generalization to other datasets. Further validation on larger, independent datasets is needed.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
pandas
opencv-python
scikit-image
scikit-learn
matplotlib
pyswarms
```

## Configure the dataset path

The script reads the dataset directory from the `APTOS_DATA_DIR` environment variable. Set it to the folder containing `train.csv` and `train_images/`.

**Windows PowerShell:**

```powershell
$env:APTOS_DATA_DIR = "C:\path\to\aptos2019-blindness-detection"
```

**Linux/macOS:**

```bash
export APTOS_DATA_DIR="/path/to/aptos2019-blindness-detection"
```

If the variable is not set, the script uses the default Kaggle input path:

```text
/kaggle/input/competitions/aptos2019-blindness-detection
```

## Run the project

```bash
python diabetic_retinopathy_detection_pso.py
```

The script prints the baseline and PSO-optimized evaluation metrics and displays:
- PSO convergence graph.
- ROC curves for baseline and PSO-optimized SVM.

## Repository Structure

```text
diabetic-retinopathy-detection/
├── README.md
├── diabetic_retinopathy_detection_pso.py
├── requirements.txt
└── results/                 # Optional: add generated plots here
```

## Limitations

- The experiment uses a small, balanced subset (up to 300 images), not the full dataset.
- The task is binary classification; it does not predict the five original DR severity grades.
- Results are based on one train-test split and may vary with the selected samples.
- The reported metrics require validation on larger and independent datasets.
- This project is for learning and experimentation, not clinical use.

## Future Improvements

- Train and evaluate using a larger portion of the APTOS dataset.
- Extend the task to five-class DR severity classification.
- Evaluate with repeated stratified cross-validation and independent test datasets.
- Compare handcrafted features with CNN-based approaches such as ResNet or EfficientNet.
- Investigate class imbalance, image quality, and clinically relevant sensitivity.

## Author

**Navya V M**

- GitHub: [navyavm123-sudo](https://github.com/navyavm123-sudo)
- LinkedIn: [Navya V M](https://linkedin.com/in/navya-v-m-3526912a2/)

---

If you find this project useful, feel free to ⭐ the repository.
