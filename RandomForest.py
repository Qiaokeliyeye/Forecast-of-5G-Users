import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')
X = data.drop(['id', 'target'], axis=1)
y = data['target']

# Feature standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Multiple experiments to determine the optimal parameters
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model using the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.4f}')

# Calculate the AUC score of the model
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f'Model AUC: {auc:.4f}')