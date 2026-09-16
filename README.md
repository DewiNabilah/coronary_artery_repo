Coronary Artery Disease Survival Classification

Project objective

This project develops an end-to-end binary classification pipeline to predict the survival class of coronary artery disease patients. The results are intended to demonstrate a machine-learning workflow that could help doctors formulate pre-emptive medical treatments. They are not a clinically validated decision system.

Three classification models are evaluated:

Logistic Regression

K-Nearest Neighbours (KNN)

Decision Tree

The models are compared using accuracy, precision, recall, F1-score and ROC-AUC. F1-score is used for hyperparameter optimisation because the target has a moderate class imbalance.


Data preparation

The workflow performs the following operations:

Standardises Survive values from 0, 1, No and Yes into binary integers.

Standardises the capitalisation of the Smoke values.

Maps L to Low and N to Normal in Ejection Fraction.

Converts negative ages to their absolute values.

Drops ID because it is an identifier.

Drops Favorite color because it is a synthetic feature without supported clinical relevance.

Replaces missing numerical values with the training-set median.

Replaces missing categorical values with the most frequent training category.

Standardises numerical features.

One-hot encodes categorical features.

The data is divided using stratified sampling:

60% training data

20% validation data

20% test data

The test set remains untouched until final evaluation.

Assumptions

1 and Yes in the Survive column are treated as class 1, meaning survival.

0 and No are treated as class 0, meaning non-survival.

Negative ages are treated as synthetic sign errors because a patient's age cannot be negative and their absolute values fall within the observed age range.

Favorite color is treated as an irrelevant synthetic feature.

Repeated IDs are not removed because the repeated records are not exact duplicates and some have conflicting target values.

The target interpretation should be changed if the assignment's official data dictionary defines class 1 differently.

Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

python -m venv venv

Activate it in Windows PowerShell:

.\venv\Scripts\Activate.ps1

Install the required packages:

python -m pip install --upgrade pip
pip install -r requirements.txt

Running the project

Run the following command from the repository root:

python main.py

The default configuration file is src/config.yaml. A different configuration file can be supplied with:

python main.py --config-path path/to/config.yaml

Model development

Baseline models

Logistic Regression

0.8290 (Vaidation Accuracy)

0.7685 (Validation Precision)

0.6687 (Validation Recall)

0.7152 (Validation F1-score)

0.8730 (Validation ROC_AUC)

K-Nearest Neighbours

0.9883 (Validation Accuracy)

0.9735 (Validation Precision)

0.9907 (Validation Recall)

0.9820 (Validation F1-score)

0.9990 (Validation ROC_AUC)

Decision Tree

0.9910 (Validation Accuracy)

0.9845 (Validation Precision)

0.9875 (Validation Recall)

0.9860 (Validation F1-score)

0.9901 (Validation ROC_AUC)

Hyperparameter tuning

Stratified five-fold cross-validation is used with F1-score as the optimisation metric.

The selected KNN parameters are:

n_neighbors = 3
p = 1
weights = distance

The tuned validation results are:

Model

K-Nearest Neighbours

0.9960 (Best CV F1-score)

0.9990 (Validaiton F1 score)

1.0000 (Validation ROC_AUC)

Decision Tree

0.9873 (Best CV F1-score)

0.9933 (Validation F1 score)

0.9952 (Validation ROC-AUC score)

Logistic Regression

0.7178 (Best CV F1-Score)

0.7148 (Validation CV F1-score)

0.8701 (Validation ROC-AUC score)

Imbalanced data

Class 0 represents approximately 67.9% of the records and class 1 represents approximately 32.1%. Random oversampling, random undersampling and no resampling are compared using the tuned KNN model.

Random oversampling produces the highest validation F1-score of 0.9995 and is selected for the final pipeline.

Final result

The final KNN pipeline is fitted using the combined training and validation data, then evaluated once on the untouched test set.



Accuracy: 0.9990

Precision: 0.9969

Recall: 1.0000

F1-score: 0.9984

ROC-AUC: 1.0000

KNN is selected as the most suitable of the three evaluated models because it produces the highest cross-validation, validation and test performance. Random oversampling improves its minority-class recall without materially reducing precision.

Limitations

The dataset may contain synthetic patterns that make the classes unusually easy to separate.

The near-perfect results do not establish clinical validity.

Repeated IDs exist in the source data. If these identify repeated observations from the same patients, a random split could place related observations in different subsets and inflate model performance.

The target mapping must be verified against the official data dictionary.

The final pipeline requires independent validation using real-world hospital data before it could support clinical decisions.