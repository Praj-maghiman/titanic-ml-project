import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
#Explore the Data
print(df.head())
print(df.info())
print(df.isnull().sum())
#Clean the Data
data = df.copy()
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
x = data['Age'].median()
data['Age'] = data['Age'].fillna(x)
data = data.dropna(subset = ['Embarked'])

#Convert Text to Numbers (Encoding)
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex']) #convert texts to numbers
le_embarked = LabelEncoder()
data['Embarked'] = le_embarked.fit_transform(data['Embarked'])
#print(data.head())

#Separate Features and Target
X = data.drop(['Survived'],axis = 1)
y = data['Survived']

#Split Data into Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(
    X,y,test_size=0.2, random_state= 39
                                                     )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 39, max_iter=1000)
model.fit(X_train, y_train) #trains the model on training data
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
pj = accuracy_score(y_test, y_pred)
print(f"Logistic Regression:\nAccuracy: {pj*100:.2f}%")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
modTree = DecisionTreeClassifier(random_state = 39)
modTree.fit(X_train, y_train)
dt_pred = modTree.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree:\nAccuracy: {dt_acc*100:.2f}%")

modRand =  RandomForestClassifier(random_state=39)
modRand.fit(X_train, y_train)
rf_pred = modRand.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest:\nAccuracy: {pj*100:.2f}%")

#Feature Importance
feature_importance = modRand.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})

#Create a Bar Chart for Model Comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['Logistic Regression','Decision Tree','Random Forest']
accuracies = [pj, dt_acc, rf_acc]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Model Comparison')
plt.savefig('model_comparison.png')
print("Chart saved as 'model_comparison.png'")

#Create a Bar Chart for Feature Importance
plt.figure(figsize=(6,8))
plt.barh(importance_df['Feature'],importance_df['Importance'])
plt.ylabel('Feature')
plt.xlabel('Importance Score')
plt.title('Feature Importance (Random Forest)')
plt.savefig('feature-importance.png')

#Create a Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm= confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot= True, fmt ='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

#Create One More Chart - Survival by Gender
survival_by_gender = data.groupby('Sex')['Survived'].mean()
plt.figure(figsize=(9,6))
plt.bar(['Female (0)','Male (1)'], survival_by_gender.values)
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.title('Survival Rate by Gender')
plt.ylim([0, 1])
plt.savefig('survival_by_gender.png')
print("Survival by gender chart saved!")