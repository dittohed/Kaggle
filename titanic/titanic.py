import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10
import seaborn as sns

# 1. Looking at data
#read a comma-separated values (.csv) file into DataFrame
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#features:
    #Pclass - passenger's class (1, 2, 3)
    #SibSp - no, of siblings(rodzeństwo)/spouses(małżonków);
    #Parch - no. of parents/children
    #Fare - cost of the ticket
    #Embarked - where the passenger got on the ship

#return the first n rows (n = 5 on default) with col names
    #print(train.head())

#print information of all columns
    #train.info()

#let's drop not useful features
train = train.drop(columns = ["PassengerId", "Name", "Ticket"])
test = test.drop(columns = ["PassengerId", "Name", "Ticket"])
print(train.head())

#draw plots (diagram liczebności - zobaczmy rozkłady poszczególnych cech)
"""
train.hist() #wywołuje plt.hist() na każdej kolumnie
plt.tight_layout() #zapewnia odpowiednie odstępy między wykresami (żeby nie były za małe)
plt.show()
"""
categorical_features = ["Pclass", "Sex", "Embarked"]
#for cat in categorical_features:
    #sns.countplot(train[cat])
    #plt.show()

# 2. Filling missing values
print("\n=========== Missing values =============")
print(train.isnull().sum()) #zlicza True dla każdej kolumny
train = train.fillna(np.nan) #zamiana nulli na numpy'owy NaN
test = test.fillna(np.nan)

#brakuje bardzo dużo kabin - zróbmy z braku kabiny nowe ID :)

#zamienimy ID kabin na pierwszą literę; X jeśli brak informacji
def fix_cabins(data):
    data["Cabin"] = [i[0] if not pd.isna(i) else 'X' for i in data["Cabin"]]

fix_cabins(train)
fix_cabins(test)

#sns.countplot(train["Cabin"])
#plt.show()

#dla reszty cech przyjmujemy następującą strategię, żeby nie wywalać danych:
#a) dla cech ilościowych zastępujemy medianą,
#b) dla cech jakościowych zastępujemy modą.

def fill_missing(data):
    missing = data.isnull().sum()
    missing = missing[missing > 0]

    for column in list(missing.index):
        if data[column].dtype == "object":
            data[column].fillna(data[column].value_counts().index[0], inplace = True)
        else:
            data[column].fillna(data[column].median(), inplace = True)

    #inplace = True wykonuje operację na obiekcie, a nie zwraca kopii

fill_missing(train)
fill_missing(test)

print("\n=========== After filling missing values =============")
print(train.isnull().sum())

# 3. Converting categorical data
categorical_features.append("Cabin") #przekształciliśmy już tę cechę do prostszej postaci

#tu powinna być funkcja, ale coś nie działało

for category in categorical_features:
    dummy_features = pd.get_dummies(train[category])
    #trzeba ładnie nazwać kolumny, bo przy tworzeniu wielu dummy features wszystko może się pomieszać
    dummy_features.columns = [(category + ": "+ str(column)) for column in dummy_features.columns]
    train = pd.concat([train, dummy_features], axis = "columns")

train = train.drop(columns = categorical_features)

for category in categorical_features:
    dummy_features = pd.get_dummies(test[category])
    #trzeba ładnie nazwać kolumny, bo przy tworzeniu wielu dummy features wszystko może się pomieszać
    dummy_features.columns = [(category + ": "+ str(column)) for column in dummy_features.columns]
    test = pd.concat([test, dummy_features], axis = "columns")

test = test.drop(columns = categorical_features)

print("\n=========== After handling categorical data =============")
print(train.head())

# 4. Fixing outliers & standarization & log transformation
#zobaczmy najpierw rozkład cech ilościowych
sns.distplot(train["Age"]) #rozkład cechy wraz estymatorem funkcji gęstości
plt.show()

#cecha ma rozkład zbliżony do normalnego - standaryzujemy
from sklearn import preprocessing
train["Age"] = preprocessing.scale(train["Age"])
test["Age"] = preprocessing.scale(test["Age"])

#usuwamy odpowiednie wiersze
train.drop(train[(train["Age"] < -3) | (train["Age"] > 3)].index, inplace = True)
test.drop(test[(test["Age"] < -3) | (test["Age"] > 3)].index, inplace = True)
"""
OGARNĄĆ
https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
"""
#pandas używa nieco innych operatorów logicznych niż Python

#zobaczmy, co się zmieniło - chyba wygląda lepiej!
sns.distplot(train["Age"])
plt.show()

sns.distplot(train["Fare"]) #skewed data
plt.show()
"""
OGARNĄĆ
https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
"""

#użyjmy transformacji logarytmicznej
train["Fare"] = (train["Fare"] + 1).transform(np.log)
test["Fare"] = (test["Fare"] + 1).transform(np.log)

#i na dokładkę standaryzujemy
train["Fare"] = preprocessing.scale(train["Fare"])
test["Fare"] = preprocessing.scale(test["Fare"])

#zobaczmy, co się zmieniło - chyba wygląda lepiej!
sns.distplot(train["Fare"])
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
