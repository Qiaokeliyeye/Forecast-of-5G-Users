import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')
X = data.drop(['id', 'target'], axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different solvers

# model = LogisticRegression(solver='lbfgs')
# Model AUC:0.4999

# model = LogisticRegression(solver='sag')
# Model AUC:0.5000

# model = LogisticRegression(solver='saga')
# Model AUC:0.5000

# model = LogisticRegression(solver='newton-cg')
# Model AUC:0.5002

model = LogisticRegression(solver='liblinear')
# Model AUC:0.5002

# Training model
model.fit(X_train, y_train)

# Evaluate the model using the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.4f}')

# Calculate the AUC score of the model
auc = roc_auc_score(y_test, y_pred)
print(f'Model AUC: {auc:.4f}')
