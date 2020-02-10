import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10
import seaborn as sns

#1. Looking at the data
#read a comma-separated values (.csv) file into DataFrames
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#features explained:
    #Pclass - passenger's class (1, 2, 3)
    #SibSp - no, of siblings(rodzeństwo)/spouses(małżonków);
    #Parch - no. of parents/children
    #Fare - cost of the ticket
    #Embarked - where the passenger got on the ship

#let's drop not useful features
train = train.drop(columns = ["PassengerId", "Name", "Ticket"])
test = test.drop(columns = ["PassengerId", "Name", "Ticket"])
print(train.head()) #returns first 5 rows on defafult

categorical_features = ["Pclass", "Sex", "Embarked", "Cabin"]
numerical_features = ["Age", "SibSp", "Parch", "Fare"]

#2. Filling missing values
print("\n=========== Missing values =============")
print(train.isnull().sum()) #counts True for each column
train = train.fillna(np.nan) #nulls to NaN
test = test.fillna(np.nan)

#lots of cabins missing - let's change missing ones to X and leave the first letter only for the rest
def fix_cabins(data):
    data["Cabin"] = [i[0] if not pd.isna(i) else 'X' for i in data["Cabin"]]

fix_cabins(train)
fix_cabins(test)

#for the rest of the features let's pursue a following strategy to not dump lots of data:
#a) for numerical features - change NaN to median,
#b) for categorical features - change NaN to mode.

def fill_missing(data):
    #isnull() returns True if value is NaN or None etc.
    #sum() returns an array (column, count)
    missing = data.isnull().sum()
    missing = missing[missing > 0] #using a mask

    for column in list(missing.index):
        if data[column].dtype == "object":
            data[column].fillna(data[column].value_counts().index[0], inplace = True)
        else:
            data[column].fillna(data[column].median(), inplace = True)

    #inplace = True returns None, inplace = False returns a copy of the object with the operation performed

fill_missing(train)
fill_missing(test)

print("\n=========== After filling missing values =============")
print(train.isnull().sum())

#3. Plotting data
for cat in categorical_features:
    sns.countplot(train[cat])
    plt.show()

for num in numerical_features:
    sns.distplot(train[num])
    plt.show()

#4. Handling categorical data
for category in categorical_features:
    dummy_features = pd.get_dummies(train[category])
    #get_dummies splits categorical column into multiple binary columns, each one entitled by the category
    dummy_features.columns = [(category + ": "+ str(column)) for column in dummy_features.columns] #naming
    train = pd.concat([train, dummy_features], axis = "columns")

train = train.drop(columns = categorical_features)

for category in categorical_features:
    dummy_features = pd.get_dummies(test[category])
    dummy_features.columns = [(category + ": "+ str(column)) for column in dummy_features.columns]
    test = pd.concat([test, dummy_features], axis = "columns")

test = test.drop(columns = categorical_features)

print("\n=========== After handling categorical data =============")
print(train.head())

# 4. Fixing outliers & standarization & log transformation
#let's see Age one more time
sns.distplot(train["Age"]).set_title("Before standarization") #feature distribution
plt.show()

#almost normal distribution - standarization will be the most effective
from sklearn import preprocessing
train["Age"] = preprocessing.scale(train["Age"].reshape(-1, 1)) #need to convert to a list
test["Age"] = preprocessing.scale(test["Age"].reshape(-1, 1))

#dumping outliers
train.drop(train[(train["Age"] < -3) | (train["Age"] > 3)].index, inplace = True)
test.drop(test[(test["Age"] < -3) | (test["Age"] > 3)].index, inplace = True)
#pandas uses different logical operators than Python
"""
OGARNĄĆ
https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
"""

sns.distplot(train["Age"]).set_title("After standarization")
plt.show()

sns.distplot(train["Fare"]).set_title("Before norm & log transform") #skewed data
plt.show()
"""
OGARNĄĆ
https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
"""
train["Fare"] = preprocessing.normalize(list(train["Fare"]))
test["Fare"] = preprocessing.normalize(list(test["Fare"]))

train["Fare"] = (train["Fare"] + 1).transform(np.log)
test["Fare"] = (test["Fare"] + 1).transform(np.log)

#zobaczmy, co się zmieniło - chyba wygląda lepiej!
sns.distplot(train["Fare"]).set_title("After norm & log transform")
plt.show()

#w teście w ogóle nie ma kolumny T! Trzeba ją usunąć z traina
train = train.drop(columns = ["Cabin: T"])

print("\n=========== Finally =============") #chyba już dużo zrobiłem - spróbujmy coś przewidzieć!
print(train.head())

print("\n=========== Preprocessed test =============")
print(test.head())

from sklearn.svm import SVC

train_y = train["Survived"]
train = train.drop(columns = ["Survived"])
clf = SVC(gamma = "auto")
clf.fit(train, train_y)
test_y = clf.predict(test)
