# titanic-ml-project
# Titanic Survival Prediction - Machine Learning Project

📌 Project Overview

This project uses Machine Learning to predict whether a passenger on the Titanic survived or not based on their characteristics (age, gender, ticket class, fare, etc.).

The goal is to build and compare multiple ML models to find the best predictor of survival.

---

📊 Dataset

- **Dataset**: Titanic Passenger Dataset
- **Total Records**: 891 passengers
- **Target Variable**: `Survived` (0 = Did not survive, 1 = Survived)
- **Features**: 7 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
- **Survival Rate**: 38.38%

---

## 🔧 Technologies Used

- **Python 3**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - Machine Learning algorithms
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization

---

## 📈 Project Steps

### 1. Data Exploration & Cleaning
- Loaded 891 passenger records
- Identified missing values:
  - Age: 177 missing
  - Cabin: 687 missing
  - Embarked: 2 missing
- Removed unhelpful columns (PassengerId, Name, Ticket, Cabin)
- Filled missing Age values with median (28.0)
- Dropped rows with missing Embarked values

### 2. Data Preprocessing
- Encoded categorical variables (Sex, Embarked) to numbers
- Final clean dataset: 889 rows × 8 columns

### 3. Feature Engineering
- Separated features (X) and target (y)
- X: 7 features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
- y: Survival outcome (0 or 1)

### 4. Train-Test Split
- Training set: 711 passengers (80%)
- Test set: 178 passengers (20%)
- Random seed: 42 (for reproducibility)

### 5. Model Training & Comparison
Trained 3 different ML models:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.21% |
| Decision Tree | 79.78% |
| **Random Forest** | **81.46%** ✓ |

**Winner**: Random Forest (81.46% accuracy)

### 6. Feature Importance Analysis
Using the Random Forest model, identified which features mattered most:

| Feature | Importance |
|---------|-----------|
| Fare | 27.4% |
| Age | 25.8% |
| Sex | 25.3% |
| Pclass | 8.1% |
| SibSp | 5.4% |
| Embarked | 4.0% |
| Parch | 3.9% |

**Key Insight**: Ticket fare (proxy for cabin location) was the strongest predictor of survival.

### 7. Model Evaluation
Confusion Matrix for Random Forest on test set:
```
                Predicted
              Did Not   Survived
              Survive
Actual
Did Not        93        17
Survive

Survived       16        52
```

- Correct predictions: 145 out of 178
- Accuracy: 81.46%

---

## 📊 Visualizations Created

1. **model_comparison.png** - Bar chart comparing all 3 models
2. **feature_importance.png** - Horizontal bar chart showing feature importance
3. **confusion_matrix.png** - Heatmap showing prediction accuracy
4. **survival_by_gender.png** - Bar chart showing survival rate by gender

---

## 🎯 Key Findings

1. **Women had higher survival rate** (~74%) compared to men (~19%)
   - Reflects historical "women and children first" policy

2. **Ticket fare was the strongest predictor** (27.4% importance)
   - Higher fare = better cabin location = closer to lifeboats

3. **Age mattered** (25.8% importance)
   - Younger passengers had better survival chances

4. **Passenger class was important** (8.1% importance)
   - 1st class had better access to lifeboats than 3rd class

---

## 💡 What I Learned

- How to build a complete ML pipeline from scratch
- Data cleaning and preprocessing techniques
- Comparing multiple ML models to find the best one
- Feature engineering and importance analysis
- Model evaluation using accuracy and confusion matrix
- Data visualization for insights

---

## 📁 Project Structure

```
titanic-ml-project/
├── README.md                    # This file
├── diy.py                       # Main Python code
├── model_comparison.png         # Model accuracy comparison chart
├── feature_importance.png       # Feature importance chart
├── confusion_matrix.png         # Confusion matrix heatmap
└── survival_by_gender.png       # Survival by gender chart
```

---

## 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

2. Run the Python script:
   ```bash
   python diy.py
   ```

3. Charts will be saved as PNG files in the same directory

---

📚 Future Improvements

- Feature scaling (StandardScaler) for better model performance
- Hyperparameter tuning to optimize Random Forest
- Cross-validation for more robust evaluation
- Deep learning models (Neural Networks)
- Handling class imbalance (more non-survivors than survivors)

---

👤 Author

Praj Maghiman V
BTech - CSE (Data Science)  
Date: October 2025

---

📝 Notes

This project was built step-by-step as a learning exercise to understand:
- ML pipeline workflow
- Data preprocessing
- Model training and evaluation
- Python libraries (pandas, scikit-learn, matplotlib)

---
Thank You!
