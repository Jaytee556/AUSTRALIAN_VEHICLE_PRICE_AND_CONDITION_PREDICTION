## ***AUSTRALIAN_VEHICLE_PRICE_AND_CONDITION_PREDICTION***
This project entails performing supervised machine learning tasks, such as regression and classification, on the Australian Vehicle Price Dataset. The objective is to predict prices, determine the condition of the car, analyze market trends and vehicle features, and propose innovative solutions to meet market demand and optimize pricing strategies. Subsequently, our machine learning model will be deployed and closely monitored.

### ***The goal is to build and train different machine-learning models to perform regression and classification tasks where we are expected to predict the price of vehicle and the condition of the Vehicle (either New or Used)***
This repository contains three different Jupyter notebooks consisting: 
>- In-depth exploratory data analysis on the Australian Vehicle Price dataset.
>- Supersived machine learning regression tasks to predict the price of a vehicle (building and training different machine learning models to perform regression tasks)
>- Supersived machine learning classification tasks to predict the condition of the vehicle (building and training different machine learning models to perform classification tasks)

### ***PROJECT OBJECTIVES***
1. `Price Prediction`: Develop and deploy machine learning models to predict the price of a car based on its features.
2. `Car Condition Classification`: Implement a car condition classification model (Used/New) for efficient categorization.
3. `Market Analysis`: Conduct comprehensive market analysis to identify trends and insights for different car types using descriptive statistics and visualizations.
4. `Feature Analysis`: Perform in-depth feature analysis to determine the most influential factors affecting car prices across various models/brands.

### ***DATASET DESCRIPTION***
This dataset contains the latest information on car prices in Australia for the year 2023. It covers various brands, models, types, and features of cars sold in the Australian market. It 
provides useful insights into the trends and factors influencing car prices in Australia. It contains `19 columns` and about `16,734 rows`.

`Dataset Source`: https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices/data

### STEPS TAKEN WHEN WORKING ON THIS PROJECT
---
1. Comprehensive and In-Depth Exploratory Data Analysis on Australian Vehicle Price Dataset:
   - Data loading: Loading in the dataset (it is in CSV format)
   - Data Inspection and Data Profiling report: Inspected the dataset, checking for missing values, duplicates, samples of the dataset, first five rows, shape, and displayed information of all        the data types of each column.
   - Data Cleaning and Preprocessing: Cleaned dirty and inconsistent columns to maintain consistency for accurate results. Handled missing values using Simple Imputer, handled outliers using          Interquartile Range Method (IQR), Checked for any column data type inconsistency, and formatted it to the right data type. Created new features (columns) where necessary.
   - Exploratory Analysis (Involving Market and Feature Analysis): Comprehensive EDA was carried out on the dataset and it was broken down into Univariate, Bivariate, and Multivariate analysis        to show market trends and feature analysis of different vehicles and their effect on the vehicle price. Also, it showed the correlation between the features.
   - Visualized different features (columns) of the dataset to show value count, correlation, relationship, feature, and market analysis to identify market trends as well.
   - Performed Univariate, Bivariate, and Multivariate analysis to explore further to uncover insight on how some features affect the prices of vehicles.
   - Made use of MatplotLib and Seaborn to visualize insights.
  
2. SUPERVISED LEARNING REGRESSION TASKS
   - Used Label encoder to transform the data from categorical to numerical columns.
   - Use Train Test Split to split the data to training and testing test. Standardized the X features with MinMax Scaler.
   - For regression tasks, five machine learning models (algorithms) were used to train so as to predict the vehicle price; they were:
     >- Linear Regression (lr)
     >- Lasso
     >- XGBRF Regressor
     >- Decision Tree Regressor (Dt)
     >- Random Forest regressor (Rf)
   - Feature selections was carried out. We defined the target variable (y) and independent variable (X)
   - The dataset was split into testing and training sets using train-test split
   - The X features were scaled using MinMax Scaler
   - Regression models building, and training commenced as well as the model evaluation.
   - Each model was fit to train the model so as to predict the value y (Price column) which is our target variable.
   - The performance was evaluated using MAE, r2_score, MSE, RMSE (Mean absolute error, R squared, Mean squared error, and root mean squared error) to show the difference between the actual           values and predicted values. The model was the lowest error and the highest r2_score was picked to be the best-performing model.
   - A scatter plot graph was plotted to visualize the performance of each model on how well it is predicting the vehicle price.
   - Feature Importance and Engineering were carried out to see the features that contribute to the performance of the model.

3. SUPERVISED LEARNNG CLASSIFICATION TASKS
   - For classification tasks, five machine learning models were used train so as to predict the condition of the vehicle (either USED/NEW); they were:
     >- Logistics Regression (log)
     >- Decision Tree Classifier(Dc)
     >- Random Forest Classifier(Rc)
     >- Support Vector Classifier (svc)
     >- KNN Classifer (Knn)
   - Feature selections was carried out. We defined the target variable (y) and independent variable (X)
   - The dataset was imablanced so SMOTE was used to balanced the dataset so that we wont get biased result.
   - The dataset was splited into testing and training sets using train-test splitThe X features were scaled using MinMax Scaler
   - Classification models building and training commenced as well as the model evaluation.
   - Each model was fit to train the model as to predict the probability of the condition of the vehicle (either USED or NEW) which is our target variable.
   - The performance was evaluated using confusion matrix, classification report(fi-score, precision, accuracy, recall). The model with the highest accuracy and other metrics was picked to be         the best-performing model.
   - AUC-ROC was plotted to also show model performance
   - Feature Engineering was carried out to see the features that contribute to the performance of the model and the features that contributed more was used to re train the model.
  
4. MODEL DEPLOYMENT
   - Streamlit was used to deploy the model. Two enviroment was created to deploy the models for the regression and classification prediction
  
### TOOLS USED
---
1. Data Visualization (Matplotlib, Seaborn, Missingno)
2. Python 3, Jupyter notebook
3. Handling missing values (Simple Imputer) and Data manipulation (NumPy, Pandas)
4. Data Preprocessing (Label Encoder, Train Test Split, Min-Max Scaler)
5. Model Building for regression tasks (Linear Regression, Lasso, XGBRFRegressor, Decision Tree Regressor, Random Forest Regressor)
6. Model Metrics for regression (mean_squared_error, r2_score, mean_absolute_error)
7. Model Building for classification tasks (Logistic regression, SVC, KNN Classifier, Decision Tree Classifier, Random Forest Classifier)
8. Model Metrics for classification (classification_report, confusion_matrix, roc_auc_score, accuracy_score)
9. Model Selection validation (cross_val_score, cross_val_predict, roc_curve, auc).
10. For Balancing imbalanced dataset (SMOTE and counter)






