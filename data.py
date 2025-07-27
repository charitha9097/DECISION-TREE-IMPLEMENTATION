# Step 1: Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create and train Decision Tree model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
clf.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization - Iris Dataset")
plt.show()
