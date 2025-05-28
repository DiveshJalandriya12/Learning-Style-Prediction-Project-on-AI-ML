import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
df = pd.read_csv('learning_style_dataset.csv')

# Split data into features and target
X = df.drop(columns=['Learning Style'])
y = df['Learning Style']

# Encode the target labels (Auditory, Visual, Kinesthetic) into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the model (using Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# --- Save Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the plot
plt.show()

# --- Save Classification Report Heatmap ---
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues")
plt.title('Classification Report')
plt.savefig('classification_report.png')  # Save the plot
plt.show()

# --- Save Feature Importance Plot ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [f'Q{i+1}' for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig("feature_importance.png")  # Save the plot
plt.show()

# Save the trained model to a file
joblib.dump(model, 'learning_style_model.pkl')
print("Model saved as 'learning_style_model.pkl'")
