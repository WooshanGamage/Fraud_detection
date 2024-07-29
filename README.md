# Fraud_detection
This project uses logistic regression to detect fraudulent credit card transactions with a dataset from Kaggle. It preprocesses and balances the data, scales features, and evaluates model accuracy. The code effectively predicts fraud in credit card transactions.
## Libraries Imported

- **pandas**: For data manipulation and analysis.
  
- **scikit-learn (sklearn)**: For machine learning models and utilities.
  - **train_test_split**: Splits the dataset into random train and test subsets.
  - **LogisticRegression**: The class used for binary and multi-class classification.
  - **StandardScaler**: Used for feature scaling.
  - **accuracy_score**: Calculates the accuracy of a model's predictions.
## Steps Performed in the Script

1. **Read the Dataset**:
    The credit card dataset is read using pandas.

2. **Separate the Dataset**:
    Legitimate transactions are separated from fraudulent transactions.

3. **Sampling**:
    A sample of legitimate transactions is taken, equal to the number of fraudulent transactions.

4. **Combine Data**:
    A new dataset is created combining the legitimate sample and all fraudulent transactions.

5. **Split Data into Features and Targets**:
    Features and targets are separated for model training.

6. **Split Data into Training and Testing Sets**:
    The data is split into training and testing sets using `train_test_split`.

7. **Scale Features**:
    Features are scaled using `StandardScaler` for better model performance.

8. **Train the Model**:
    A logistic regression model is trained using the scaled training data.

9. **Make Predictions**:
    Predictions are made on the test data.

10. **Evaluate the Model**:
    The accuracy of the model is evaluated using `accuracy_score` for both test and training data.

The project showcases how to handle imbalanced datasets, perform feature scaling, and evaluate a logistic regression model using scikit-learn. The careful use of library functions ensures a robust and accurate fraud detection system.
