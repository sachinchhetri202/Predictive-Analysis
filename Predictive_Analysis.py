import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report

print("\nAssignment 6 : Sachin Chhetri\n")

df_train = pd.read_csv('Titanic_crew_train.csv')
df_test = pd.read_csv('Titanic_crew_test.csv')
df_train.dropna(subset=['Survived?'], inplace=True)

# survival prediction
X_train_survival = df_train[['Age']]
y_train_survival = df_train['Survived?'].map({'LOST': 0, 'SAVED': 1})
X_test_survival = df_test[['Age']]
y_test_survival = (df_test['Survived?'] == 'SAVED').astype(int)
test_df_survival = df_test.dropna(subset=['Survived?'])
X_test_survival = test_df_survival[['Age']]
y_test_survival = (test_df_survival['Survived?'] == 'SAVED').astype(int)

# DT for Survival Prediction
dt_survival = DecisionTreeClassifier()
dt_survival.fit(X_train_survival, y_train_survival)
survival_predictions_dt = dt_survival.predict(X_test_survival)
correct_predictions_dt = sum(survival_predictions_dt == y_test_survival)
correct_predictions_dt = min(correct_predictions_dt, len(y_test_survival))  # Cap at the total number of crew members

# NN for survival prediction
survival_neural_network = MLPClassifier()
survival_neural_network.fit(X_train_survival, y_train_survival)
survival_predictions_nn = survival_neural_network.predict(X_test_survival)
correct_predictions_nn = sum(survival_predictions_nn == y_test_survival)
correct_predictions_nn = min(correct_predictions_nn, len(y_test_survival))

# Age prediction
X_train_age = df_train[['Age']]  # Assuming only age is used as a feature
y_train_age = df_train['Age'].astype(float)
X_test_age = df_test[['Age']]

# LR for Age Prediction
lr_age = LinearRegression()
lr_age.fit(X_train_age, y_train_age)
age_predictions_lr = lr_age.predict(X_test_age)
mse_linear_regression = mean_squared_error(df_test['Age'], age_predictions_lr)

# DT for Age Prediction
dt_age = DecisionTreeRegressor()
dt_age.fit(X_train_age, y_train_age)
age_predictions_dt = dt_age.predict(X_test_age)
mse_decision_tree = mean_squared_error(df_test['Age'], age_predictions_dt)

# NN for Age Prediction
nn_age = MLPRegressor()
nn_age.fit(X_train_age, y_train_age)
age_predictions_nn = nn_age.predict(X_test_age)
mse_neural_network = mean_squared_error(df_test['Age'], age_predictions_nn)

print("Survived prediction:")
print(f"Decision trees: {correct_predictions_dt}/178")
print(f"Neural networks: {correct_predictions_nn}/178")

# Confussion Matrix & Classification Report
print("\nConfusion Matrix for Decision Trees:")
print(confusion_matrix(y_test_survival, survival_predictions_dt))
print("\nClassification Report for Decision Trees:")
print(classification_report(y_test_survival, survival_predictions_dt))
print("Confusion Matrix for Neural Networks:")
print(confusion_matrix(y_test_survival, survival_predictions_nn))
#print("\nClassification Report for Neural Networks:")
#print(classification_report(y_test_survival, survival_predictions_nn))

print("\nAge Prediction: ")
print(f"Linear Regression for Age Prediction: MSE: {mse_linear_regression}")
print(f"Decision Tree for Age Prediction: MSE: {mse_decision_tree}")
print(f"Neural Network for Age Prediction: MSE: {mse_neural_network}")

