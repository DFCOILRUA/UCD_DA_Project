# Importing all necessary and potential libraries
import requests
import json
from datetime import datetime
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skl
import xgboost as xgb
from xgboost import XGBClassifier
sns.set()


data = pd.read_csv("Tweets.csv") # Reading the Data and importing it into a Pandas DataFrame
print(data.head())

 # dropping all non-numeric value fields, leaving only airline_sentiment as we will be replacing neutral, positive and negative with 0, 1 and 2 respectively
print(data[["airline_sentiment", "airline_sentiment_confidence", "negativereason_confidence", "retweet_count"]])

data_clean= data[["airline_sentiment", "airline_sentiment_confidence", "negativereason_confidence", "retweet_count"]]
print(data_clean.head())


# replacing airline_sentiment string values with numeric values for the purposes of carrying logisitic regression on this multi-class classification issue
print(data_clean['airline_sentiment'].replace(['neutral', 'positive', 'negative'], [0, 1, 2], inplace=True))

print(data_clean.head(5))

print(data_clean.describe())

print(data_clean.dropna())

print(data_clean_2 =data_clean.dropna())    #dropping null values from the dataframe data_clean and assigning it to a new dateframe data_clean_2

print(data_clean_2.head())

print(data_clean_2.describe())   #displaying descriptive statistics for the data_clean_2 dataframe

#Plotting a distrubtion of data_clean_2 through seaborn
plt.figure(figsize=(25,20), facecolor='pink')
plotnumber = 2

for column in data_clean_2:
    if plotnumber<=4 and column!='airline_sentiment' :
        ax = plt.subplot(2,2, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=2
plt.show()


#plotting distribution plot of airline_sentiment
sns.distplot(data_clean_2.airline_sentiment)
plt.show()


#plotting distribution plot of airline_sentiment_confidence
sns.distplot(data_clean_2.airline_sentiment_confidence)
plt.show()


#plotting distribution plot of negativereason_confidence
sns.distplot(data_clean_2.negativereason_confidence)
plt.show()

#plotting distribution plot of retweet_count
sns.distplot(data_clean_2.retweet_count)
plt.show()


X = data_clean_2.drop(columns = ['airline_sentiment'])
y = data_clean_2['airline_sentiment']

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

print(X_scaled)


#checking for multicolinearity i.e. feature with similar meanings
# vif = variance inflation factor
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns
print(vif)


# None of the above values are above 5, rather they are all very low which is good, therefore we can keep them
#splitting my data into training and testing subsets
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 217)


#run the Logistic regression function from the scikitlearn library that was imported
#loading the LogisticRegression algorithm and saving it to a variable named log_reg
log_reg = LogisticRegression()

#train the algorithm with the training values that I created
print(log_reg.fit(x_train,y_train))

y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


#Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print(Accuracy)


# Precison
Precision = true_positive/(true_positive+false_positive)
print(Precision)


# Recall
Recall = true_positive/(true_positive+false_negative)
print(Recall)

# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(F1_Score)

plt.figure(figsize = (20,15))
sns.heatmap(conf_mat, annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# The above heat map shows us an impressive degreee of accuracy whereby a True Positive was detected 240 times for the value
# of 0 (neutral sentiment) and a total of 2,300 times for the the value of 2 (negative sentiment).
# The only False Positives recorded were 83 cases where the value 0 was falsely predicted when in fact the true value was
# 1 (positive sentiment)


###### BOOSTING  ######################



# fit model no training data
model = XGBClassifier(objective='binary:logistic')
print(model.fit(x_train, y_train))


# cheking training accuracy
y_pred_1 = model.predict(x_train)
predictions = [round(value) for value in y_pred_1]
accuracy_1 = accuracy_score(y_train,predictions)
print(accuracy_1)


# cheking initial test accuracy
y_pred_1 = model.predict(x_test)
predictions = [round(value) for value in y_pred_1]
accuracy_1 = accuracy_score(y_test,predictions)
print(accuracy_1)


print(x_test[0])

# In order to increase the accuracy of the model, we'll do hyperparameter tuning using grid search
param_grid = {

    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    ## The max depth of each weak learner that I want. i.e. choosing the amount of layers to apply to each decision tree.
    'n_estimators': [10, 50, 100, 200]  ## The number of decision trees that I want as weak learners

}


grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid, verbose=3)

print(grid.fit(x_train,y_train))

# To  find the parameters giving maximum accuracy
print(grid.best_params_)