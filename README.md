# [BIO-322] EPFL Machine Learning Miniproject: Mouse Neural Activity Analysis

**Course:** BIO-322: Machine Learning for Biology  

**Team Name:** : Tuline & Sasha 

**Members:**

- Tuline Dachraoui (**tuline.dachraoui@epfl.ch, SCIPER: 361774**)
- Sasha Ghanipour (**sasha.ghanipour@epfl.ch, SCIPER: 364009**)

---

## 1. Project Overview
In this project, we analyze neural activity data recorded from mice to predict their behavior and perception. The goal is to classify trial types (visual vs. auditory stimuli) and the mouse's response (lick vs. no lick) using various machine learning models.

The project involves:

1.  **Data Inspection:** Visualization and exploration of neural time-series data.
2.  **Preprocessing:** Feature engineering and data cleaning.
3.  **Modeling:** Training Linear (Logistic Regression) and Non-Linear (Random Forest, SVM, etc.) models.
4.  **Evaluation:** Assessing model performance and generalization across different subjects.

## 2. Repository Structure

```text
├── data/                 # Raw CSV files (Not tracked by Git)
│   ├── train.csv         # Training dataset
│   └── test.csv          # Testing dataset
├── notebooks/            # Jupyter Notebooks
│   └── solution.ipynb    # Main project notebook containing all code and analysis
├── .gitignore            # Files to exclude from Git (e.g., large data files)
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies and versions
```

## 3. Installation & Setup (Reproducibility)

To ensure this project is fully reproducible, please follow these steps to set up the environment.

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your machine.
- Git.

### Step-by-Step Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sashaghanipour/Tuline-Sasha-Miniproject-BIO-322.git
    ```

2.  **Create the Python environment:**
    We use Python 3.12 for this project.
    ```bash
    conda create --name epfl_project python=3.12 -y
    conda activate epfl_project
    ```

3.  **Install dependencies:**
    This installs the exact library versions used in our analysis.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Register the Kernel (Optional but recommended):**
    If using Jupyter Notebook or Lab, register the kernel so it appears in the kernel list:
    ```bash
    python -m ipykernel install --user --name=epfl_project
    ```

## 4. How to Run

1.  Place the competition data (`train.csv` and `test.csv`) inside the `data/` folder.
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `notebooks/solution.ipynb`.
4.  Ensure the kernel is set to **epfl_project** (or "Python 3.12 (epfl_project)").
5.  Run all cells to reproduce the figures and results.

## 5. System Information
*   **Operating System:** [e.g., macOS Sequoia / Windows 11 / Ubuntu 22.04]
*   **Python Version:** 3.12