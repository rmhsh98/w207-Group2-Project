# Predicting Chagas Disease from ECG Features Using Machine Learning

## Project Overview
This repository contains the code, exploratory data analysis, and modeling experiments for predicting Chagas disease using engineered electrocardiogram (ECG) features. 
The primary goal of this project is to provide a computationally lightweight and highly interpretable machine learning approach to distinguish Chagas-positive from Chagas-negative exams. 
By utilizing traditional machine learning algorithms on defined cardiac features, this project serves as a screening-style tool to prioritize cases for follow-up testing in resource-constrained environments.

## Team Members
*   **Rohan Maheshwari**
*   **Khushleen Kaur**
*   **Michelle Ly**

## Dataset
The project utilizes the ECG Chagas Dataset from Kaggle, specifically working with merged demographic data and extracted physiological measurements (e.g., Heart Rate Variability, P-wave duration, PR intervals).
To prevent data leakage, a strict patient-level split (approx. 60/20/20) was maintained throughout all experiments. 

Processing included removing duplicate exams, rows without targets, and 161 exams from 66 patients with conflicting labels. Infinite values were converted to missing. We imputed 2,467 missing values using only training set medians. 
Features were then standardized using StandardScaler to account for differing numerical scales. No one-hot encoding was required. The 49 predictors were reduced to 18 features. 
We removed redundant min/max summaries and highly skewed ST_slope features lacking class separation, while retaining four core HRV metrics (MeanNN, SDNN, RMSSD, and pNN50).


## Repository Structure & Notebooks
The project workflow is divided into the following Jupyter Notebooks / Folders:

*   **`final_model_results`**: Contains results from the final model selected for best performance
*   **`model_results`**: Contains results from hyperparemeter tuning and feature importance for all 3 models: Logistics Regression, KNN, and Random Forest Classifier.
*   **`processed_data`**: intermediary and final version of the data with the train, validation, and test splits as csv files
*   **`khushleen_207_eda.ipynb`**: Maintained by Khushleen Kaur. Contains the expanded Exploratory Data Analysis (EDA). This notebook covers data cleaning, visualization of class balances, pairwise feature interactions, outlier analysis, and the final feature selection process that reduced the dataset to 18 key predictors.
*   **`rohan_logistic_regression.ipynb`**: Maintained by Rohan Maheshwari. Details the implementation, training, and tuning of the Logistic Regression model. It includes hyperparameter tuning for L2 regularization and evaluates feature weights to establish an interpretable linear boundary.
*   **`decision_tree.ipynb`**: Maintained by Michelle Li. Explores a standalone Decision Tree classifier to evaluate basic non-linear splits before progressing to more complex ensemble methods.
*   **`knn.ipynb`**: Maintained by Michelle Li. Contains the K-Nearest Neighbors (KNN) model experiments. This notebook includes tuning for the optimal number of neighbors (K), neighbor weighting strategies, and distance metrics (e.g., Manhattan distance).
*   **`khushleen_207_randomforest.ipynb`**: Maintained by Khushleen Kaur. Covers the implementation and extensive hyperparameter tuning (using `RandomizedSearchCV`) of the Random Forest classifier. This notebook focuses on managing tree complexity to capture non-linear interactions between ECG features without overfitting.
*   **`final_random_forest_test_set.ipynb`**: Maintained by Rohan Maheshwari. Contains the final, definitive evaluation. The best-performing model from the validation phase (Random Forest) is retrained on the combined train/validation data and evaluated exactly once on the locked test set to generate the final performance metrics (Accuracy, Recall, ROC-AUC, etc.).

The repository also contains 3 different branches, one for each team members and contains individually each members contributions and notebooks. 



