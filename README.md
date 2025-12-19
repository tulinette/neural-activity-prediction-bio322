# [BIO-322] Machine Learning for Neural Decoding
### Predicting Mouse Behavior from Large-Scale Brain Activity

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2-yellow.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)

**Course:** BIO-322: Machine Learning for Biology  
**Authors:**
- Tuline Dachraoui (`tuline.dachraoui@epfl.ch`)
- Sasha Ghanipour (`sasha.ghanipour@epfl.ch`)

---

## Abstract
This project presents a comprehensive machine learning pipeline designed to decode a mouse's behavior and perception from high-dimensional neural recordings. We systematically address the core challenges of this dataset: **non-stationary noise** (session drift), **high dimensionality**, and **severe class imbalance**. Our final model, a custom weighted ensemble named **"Captain v2,"** achieves a final Kaggle Leaderboard score of **0.653**, demonstrating high predictive accuracy. The key breakthrough was the identification and engineering of a feature to model the subject's **internal state (fatigue)**, which, when combined with a diverse set of non-linear models, proved essential for generalizing across different subjects.

## Key Findings & Insights
Our iterative process of hypothesis testing and modeling yielded several key insights:

1.  **Non-Linearity is Essential:** A baseline L1-regularized Logistic Regression achieved a local `GroupKFold` CV score of only **45.86%**. In contrast, our final non-linear ensemble reached **59.8%** on the same metric, proving the neuro-behavioral relationships are too complex for linear models alone.

2.  **The "Fatigue" Breakthrough:** The single most impactful discovery was the strong negative correlation between `trial_number` and licking probability (~30% -> ~18%). Modeling this "fatigue" effect by including `trial_number` as a feature was the key to breaking our initial performance plateau and jumping from a score of 0.639 to **0.653**.

3.  **The Importance of a "Geometric" Model (SVM):** A standalone SVM, trained on the neural data plus the fatigue feature, achieved a solo CV score of **~60%**, outperforming our standalone Random Forest (~58%). This validates that the fatigue effect is best modeled as a **smooth, continuous curve** (an RBF kernel manifold) rather than the rigid, step-like splits of decision trees.

4.  **The "Diamond Seed" & Limits of Predictability:** Our final model represents a highly optimized but brittle configuration. Numerous attempts to improve the 0.653 score—including advanced feature engineering (K-Means, RFE), domain adaptation (Pseudo-Labeling), and model blending—all failed. This suggests we reached the effective limit of the signal-to-noise ratio, where the remaining error is likely irreducible **aleatoric uncertainty** from the inherent randomness of biological behavior.

## Champion Model Architecture: "Captain v2"
Our final model is a weighted `VotingClassifier` that establishes a clear hierarchy based on our findings:

-   **The Captain (Random Forest, Weight 3):** Serves as the primary **"Neural Decoder,"** tasked with interpreting the ~600 high-dimensional firing patterns selected by our Lasso filter.
-   **The Specialist (SVM, Weight 1):** Serves as the **"State Modulator,"** using its geometric kernel to model the smooth, non-linear curve of subject fatigue.
-   **The Scouts (XGBoost, ExtraTrees, etc., Weight 1):** Provide architectural diversity, ensuring the final prediction is robust and not over-reliant on the quirks of a single algorithm.

## Repository Structure
```
├── 📂 data/
│   ├── train.csv
│   └── test.csv
├── 📂 notebooks/
│   └── solution.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducibility Guide

To ensure this project is fully reproducible, please follow these steps.

### 1. Clone the Repository
```bash
git clone https://github.com/sashaghanipour/Tuline-Sasha-Miniproject-BIO-322.git
cd Tuline-Sasha-Miniproject-BIO-322
```

### 2. Create and Activate the Conda Environment
We use Python 3.12 for this project.
```bash
conda create --name epfl_project python=3.12 -y
conda activate epfl_project
```

### 3. Install Dependencies
This installs the exact library versions used in our analysis.
```bash
pip install -r requirements.txt
```

### 4. Run the Notebook
1.  Place the competition data (`train.csv` and `test.csv`) inside the `data/` folder.
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter lab
    ```
3.  Open `notebooks/solution.ipynb`.
4.  Ensure the kernel is set to **epfl_project**.
5.  Run all cells to reproduce the figures, analyses, and the final submission file.

---
```
