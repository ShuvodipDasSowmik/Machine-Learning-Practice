# Machine Learning Practice Repository

This repository contains my journey through machine learning fundamentals, following Kaggle's comprehensive learning paths. It includes implementations of various ML concepts, competition solutions, and practical examples using real-world datasets.

## 📚 Machine Learning Tutorial Guideline

This project follows **Kaggle's Learn** curriculum in a structured progression:

### **Step 1: [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)**
Fundamental concepts and basic model building covering:
- Basic model building and evaluation
- Decision trees and their limitations  
- Random forests for improved performance
- Train-validation-test splits
- Overfitting and underfitting concepts

### **Step 2: [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)**
Advanced techniques including:
- Missing value strategies and handling
- Categorical variable encoding techniques
- Pipeline construction for clean ML workflows
- Cross-validation for robust model evaluation
- Data leakage prevention

## 🎯 Practical Applications

### 🏠 Housing Prices Prediction
**Competition**: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

- **Objective**: Predict house prices using regression techniques
- **Models**: Random Forest Regressor with feature engineering
- **Files**:
  - `Model.ipynb` - Initial model implementation
  - `2ndModel.ipynb` - Improved model with advanced techniques

### 🚢 Titanic Survival Prediction
**Competition**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

- **Objective**: Predict passenger survival using classification
- **Models**: Random Forest with missing value handling
- **Files**:
  - `Model.ipynb` - Basic survival prediction model
  - `ModelConsideringCategoricalVars.ipynb` - Enhanced model with categorical variables

## 🔧 Machine Learning Concepts Implementation

### 1. Missing Values Handling (`Models/MissingValues/`)
- **File**: `HandlingMissingValues.ipynb`
- **Techniques**: Dropping columns, imputation strategies, comparison of approaches
- **Purpose**: Learn how to handle incomplete datasets effectively

### 2. Categorical Variables (`Models/CategoricalVariables/`)
- **File**: `HandlingCategoricalVariables.ipynb`
- **Techniques**: One-Hot Encoding, Label Encoding, Ordinal Encoding
- **Purpose**: Convert categorical data for machine learning algorithms

### 3. Model Pipelines (`Models/Pipeline/`)
- **File**: `ModelWithPipeline.ipynb`
- **Benefits**: Cleaner code, reduced data leakage, easier deployment
- **Purpose**: Create systematic ML workflows

### 4. Cross-Validation (`Models/CrossValidation/`)
- **Concepts**: K-Fold Cross-Validation, model performance estimation
- **Purpose**: Robust model evaluation and hyperparameter tuning

### 5. Decision Tree Regressor (`Models/DecisionTreeRegressor/`)
- **File**: `melbourneHousing.ipynb`
- **Implementation**: Basic decision tree for regression tasks
- **Purpose**: Understand tree-based learning algorithms

### 6. Random Forest (`Models/RandomForest/`)
- **File**: `melbourneHousing.ipynb`
- **Advantages**: Reduced overfitting, better generalization
- **Purpose**: Ensemble method for improved performance

## 🔄 Project Structure

```
Machine Learning/
├── README.md                    # Project documentation
├── Competitions/                # Kaggle competition solutions
│   ├── HousingPrices/          # House price prediction
│   │   ├── Data/               # Dataset and descriptions
│   │   └── Model/              # Jupyter notebooks and submissions
│   └── Titanic/                # Titanic survival prediction
│       ├── Data/               # Training and test datasets
│       └── Model/              # Model implementations
├── Models/                     # ML concepts implementation
│   ├── CategoricalVariables/   # Handling categorical data
│   ├── CrossValidation/        # Model validation techniques
│   ├── DecisionTreeRegressor/  # Decision tree implementation
│   ├── MissingValues/          # Missing data strategies
│   ├── Pipeline/               # ML pipeline construction
│   └── RandomForest/           # Ensemble method implementation
```

## 🛠️ Core Technologies

```python
# Essential Libraries
import pandas as pd              # Data manipulation
import numpy as np               # Numerical computing

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model Evaluation
from sklearn.metrics import mean_absolute_error, accuracy_score
```