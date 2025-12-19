# [BIO-322] Machine Learning for Neural Decoding
### Predicting Mouse Behavior from Large-Scale Brain Activity
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2-lightgrey.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26-blueviolet.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A project by **Tuline** & **Sasha**

**Course:** BIO-322: Machine Learning for Life Sciences Engineering  

**Authors:**
- Tuline Dachraoui (`tuline.dachraoui@epfl.ch`), **SCIPER**: 361774
- Sasha Ghanipour (`sasha.ghanipour@epfl.ch`), **SCIPER**: 364009

---

## Table of Contents
1.  [**Abstract**](#1-abstract)
2.  [**Our Approach: A Narrative of Discovery**](#2-our-approach-a-narrative-of-discovery)
3.  [**Champion Model Architecture: "Captain v2"**](#3-champion-model-architecture-captain-v2)
4.  [**Repository Structure**](#4-repository-structure)
5.  [**Reproducibility Guide**](#5-reproducibility-guide)
6.  [**AI Disclosure Statement**](#6-ai-disclosure-statement)

---

## 1. Abstract
This project presents a comprehensive machine learning pipeline designed to decode a mouse's behavior and perception from high-dimensional neural recordings, culminating in a **first-place finish on the private Kaggle Leaderboard with a final score of 0.62977**. Our final selected model, a custom weighted ensemble named **"Captain v2,"** demonstrated superior robustness and generalization on the hidden 80% of the test data, successfully navigating the leaderboard "shake-up."

Our development process systematically addressed the core challenges of this dataset: non-stationary noise (session drift), high dimensionality, and severe class imbalance. While our best public leaderboard score reached **0.653**, our key breakthrough was identifying that a model's true value lay in its ability to **generalize**. The winning architecture's success is attributed to the identification and engineering of a feature to model the subject's **internal state (fatigue)**. By combining this biologically-inspired feature with a diverse set of non-linear models, our ensemble learned a universal rule that proved highly effective and robust when evaluated against the final, unseen test set.

## 2. Our Approach: A Narrative of Discovery

Our path to the final model was an iterative process of hypothesis testing, where "failed" experiments were as informative as successes.

### I. The Linear Baseline & The Non-Linear Necessity
We began by establishing a baseline with an **L1-Regularized Logistic Regression**. This achieved a local `GroupKFold` CV score of only **45.86%**, confirming that a purely linear model is insufficient. This led us to explore non-linear models, with a **Random Forest** on the same data achieving a much stronger **57.1%**, proving that the decision boundaries are complex and interactive.

### II. The First Plateau: The Limits of the Neural Signal
After extensive feature engineering (Temporal Gradients) and noise reduction (Lasso Filtering), our best neural-only ensemble plateaued at a Kaggle score of **0.639**. This "glass ceiling" suggested we had exhausted the information available in the *firing rates alone*. The plots "Predictive Power of Brain Regions" and "Biological Timing" confirmed the signal was concentrated in **L5 of ALM/wS1** and in the **150-250ms time window**, but this knowledge was not enough to surpass the plateau.

### III. The Breakthrough: Modeling Internal State ("Fatigue")
The crucial insight came from investigating the "non-stationarity" mentioned in the project description.
-   **Hypothesis:** The mouse's internal state (satiety, fatigue) changes over a session, acting as a confounding variable.
-   **Evidence:** Our "Mouse Fatigue" plot provided the smoking gun, showing a **~40% drop in licking probability** from the start to the end of a session.
-   **Solution:** We engineered a `trial_number` feature to model this drift. This required adding a **Support Vector Machine (SVM)** to our ensemble, as its geometric RBF kernel could model the smooth, non-linear curve of fatigue in a way that our tree-based models could not. This single change shattered the plateau, boosting our score to **0.653**.

### IV. The Final Plateau: Defining the Limits of Predictability
With a new peak of 0.653, we launched a final series of advanced experiments to capture the #1 spot. Every attempt failed, but provided critical insights:
-   **Domain Adaptation (Pseudo-Labeling) failed** because our model's confidence on the "new mice" in the test set was too low (mean: 0.37), causing it to learn from its own hesitant predictions.
-   **Advanced Features (K-Means, RFE, etc.) failed**, proving that our simple feature set (Gradients + Lasso + Fatigue) had already captured the maximum available signal.
-   **Blending & Averaging failed**, confirming our final model is a **"Diamond Seed"**—a highly optimized but brittle configuration where any change dilutes its performance.

This proves that the remaining ~35% error is likely irreducible **aleatoric uncertainty** from the inherent randomness of biological behavior.

## 3. Champion Model Architecture: "Captain v2"
Our final model is a weighted `VotingClassifier` that embodies a hierarchical understanding of the decoding problem, achieving a local `GroupKFold` CV score of **59.8%** and a Kaggle score of **0.653**.

-   **The Captain (Random Forest, Weight 3):** The primary **"Neural Decoder,"** interpreting the ~600 high-dimensional firing patterns selected by our Lasso filter.
-   **The Specialist (SVM, Weight 1):** The **"State Modulator,"** using its geometric kernel to model the smooth curve of subject fatigue. Our weight ablation studies proved that giving the SVM more power hurt performance, confirming the hierarchy: **Neural Signal > Internal State**.
-   **The Scouts (XGBoost, ExtraTrees, etc.):** Provide architectural diversity to stabilize the ensemble.


## 4. Repository Structure
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

## 5. Reproducibility Guide

To ensure this project is fully reproducible, please follow these steps.

### I. Clone the Repository
```bash
git clone https://github.com/sashaghanipour/Tuline-Sasha-Miniproject-BIO-322.git
cd Tuline-Sasha-Miniproject-BIO-322
```

### II. Create and Activate the Conda Environment
We use Python 3.12 for this project.
```bash
conda create --name epfl_project python=3.12 -y
conda activate epfl_project
```

### III. Install Dependencies
This installs the exact library versions used in our analysis.
```bash
pip install -r requirements.txt
```

### IV. Run the Notebook
1.  Place the competition data (`train.csv` and `test.csv`) inside the `data/` folder.
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter lab
    ```
3.  Open `notebooks/solution.ipynb`.
4.  Ensure the kernel is set to **epfl_project**.
5.  Run all cells to reproduce the figures, analyses, and the final submission file.

## 6. AI Disclosure Statement

In accordance with EPFL academic policies **(LEX 1.3.3, Article 4)**, we disclose the use of a generative AI tool in the development of this project.

*   **LLM Used:** Google's Gemini.

*   **Role and Manner of Use:**
    The AI was utilized as an interactive tool for **debugging, code clarification, and scientific cross-referencing**. Its role was strictly that of a technical assistant, analogous to a compiler's error checker or a library's documentation, and was not used for generating core logic or scientific hypotheses. The primary uses were:
    1.  **Syntax and Debugging:** The AI was instrumental in resolving specific runtime errors (e.g., `ValueError` from Scikit-learn's metadata routing, `TypeError` in NumPy data casting). It helped identify and correct Python syntax for advanced library functions, accelerating our development cycle.
    2.  **Biological Cross-Referencing:** After we independently generated diagnostic plots and formed our own hypotheses (for example, identifying the "Fatigue" trend in licking probability), we queried the AI for established biological concepts that could explain our findings. This allowed us to connect our data-driven insights to correct scientific terminology, such as "homeostasis" and "neuromodulatory gain control," enriching our written analysis. The AI served as a dynamic textbook, to help us link our biology classes to observed mechanisms.
    3.  **Code Readability and Documentation:** We used the AI to improve the overall quality and readability of our codebase. This included assistance in generating clear docstrings for our functions and adding explanatory comments to complex logical blocks, making the project more understandable and maintainable.
    4.  **Language and Redaction:** The AI served as a writing assistant to refine the English phrasing, grammar, and structure of non-code text, such as markdown cells, comments, and this `README.md`. This ensured our scientific narrative was communicated clearly and professionally, using proper markdown syntax.

*   **Adherence to Guidelines and Academic Integrity:**
    *   **Original Work:** The core intellectual work, including the formulation of hypotheses, the design of experiments, the choice of models, and the final scientific conclusions, is entirely our own. No core project logic or complete analytical code blocks were generated by the AI.
    *   **Data Privacy:** No private or sensitive data, nor any copyrighted course materials, were shared with the AI tool. All inputs were limited to our own code snippets, error messages, and general scientific inquiries.
    *   **Verification:** All information provided by the AI, especially regarding biological concepts, was treated as a starting point and cross-verified with credible sources to ensure scientific accuracy.
