import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#Încarcă datele procesate
data = pd.read_csv('data/label_depresie/data_cleaned_with_depresie.csv')

#Separă variabilele
X = data.drop(columns=['Depresie']) 
y = data['Depresie']  

#Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creăm modelul cu class_weight='balanced'
model = LogisticRegression(max_iter=8000, class_weight='balanced')

#Antrenăm modelul
model.fit(X_train, y_train)

#Preziceri
y_pred = model.predict(X_test)

#Evaluare pe test set
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Evaluare pe train set(overfitting)
y_train_pred = model.predict(X_train)
print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))

#Cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross validation
print("\nCross-validation scores:", scores)
print("Mean Cross-validation Accuracy:", scores.mean())
print("Standard Deviation of Cross-validation Accuracy:", scores.std())

# Confusion Matrix Heatmap ---
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Depressed', 'Depressed'], yticklabels=['Not Depressed', 'Depressed'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
