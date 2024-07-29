# using a pandas library for data manipulation and analysis
import pandas as panda
# Use to splits matrices into random train and test subsets in the script
from sklearn.model_selection import train_test_split
# Import LogisticRegression class from Sklearn for binary and multi-class classification
from sklearn.linear_model import LogisticRegression
# Import this module from module used for feature scaling
from sklearn.preprocessing import StandardScaler
# Import this module for calculates the accuracy of a model's predictions
from sklearn.metrics import accuracy_score



# Read the credit card dataset ( Downloaded from kaggle.com )
credit_card = panda.read_csv("creditcard.csv")

# Separate the dataset into legitimate and fraudulent transactions
# getting class 0 as a legit transactions
legit_Transactions = credit_card[credit_card.Class == 0]
# getting class 1 as a fraud transactions
fraud_Transactions = credit_card[credit_card.Class == 1]

# Take a sample of legitimate transactions equal to the number of fraudulent transactions
# In here 492 are a fraud transactions
legit_sample = legit_Transactions.sample(n=492)

# Create a new dataset combining the legit sample and all fraudulent transactions
new_data_set = panda.concat([legit_sample, fraud_Transactions], axis=0)

# Splitting data into Features and Targets
# In here axis used for drop a Class column
X = new_data_set.drop('Class', axis=1)
Y = new_data_set['Class']

# Split data into training data and Testing data
# test_size=0.2 = 20% of the data will be used as the test set
# random_state=2 =  Ensures the train/test split is the same across different runs
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Using scaler as StandardScaler for easy to use in coding
scaler = StandardScaler()
# Scale the X features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training with increased max_iter and alternative solver
# max_iter=2000 - Allows up to 2000 iterations for convergence
# Used a saga as a solver because it is particularly efficient for large datasets
# Applies L2 regularization to avoid overfitting
# Sets the inverse of regularization strength to 1.0
model = LogisticRegression(max_iter=2000, solver='saga', penalty='l2', C=1.0)

# Training the logistic regression model with training data
model.fit(X_train_scaled, Y_train)

# Make predictions on test data
Y_test_prediction = model.predict(X_test_scaled)

# Evaluate the accuracy of the model for test data
accuracy_for_test_data = accuracy_score(Y_test, Y_test_prediction)
# Display the Test Data Accuracy just using format option and used 6 decimal numbers to present that
print(f"\n Test Data Accuracy: {accuracy_for_test_data:.6f}")

# Make predictions on training data
Y_train_prediction = model.predict(X_train_scaled)

# Evaluate the accuracy of the model for training data
accuracy_for_training_data = accuracy_score(Y_train, Y_train_prediction)
# Display the Training Data Accuracy just using format option and used 6 decimal numbers to present that
print(f"\n Training Data Accuracy: {accuracy_for_training_data:.6f}")
