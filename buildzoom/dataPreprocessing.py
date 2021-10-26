import pandas as pd
import numpy as np
import timeit

from statistics import median, mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score

import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

def get_uniques(df, col_name):

    """Any instance of the license type may contain duplicate!! For example:
    the GENERAL CONTRACTOR LICENSE may appear twice, but in reality doesn't
    add any value to our model and need to be removed."""

    uniques = []
    for i in df[col_name]:
        uniques += i
    return list(set(uniques))


def get_pattern(df):

    """
    This function identifies all the different possible appearances
    for a license type. For example: GENERAL CONTRACTOR LICENSE has the
    following appearances:
    >>> ['GENERAL CO', 'GENERAL C', 'GENERA',
        'GENERAL CONTRACTOR LICENSE', 'GENERAL ', 'GENERAL CONT',
        'GENERAL CONTRA', 'GENERAL']
    """

    uniques = get_uniques(df, 'licensetype')
    unique = []
    for i in range(len(uniques)):

        pattern = re.compile('^'+uniques[i][:5]+'*')
        matches = []
        for index, license in enumerate(uniques):

            if pattern.search(license) is not None:
                matches.append(license)

        if matches not in unique:
            unique.append(matches)
        pass

    licenseList = []
    licenseDict = {}

    for licenses in unique:
        licenseList.append(licenses)
        licenseDict[max(licenses)] = licenses

    return licenseList


# Clean licensetype
def clean_license(inp_list, pattern):

    """
    This function takes a list of patterns generated by the get_pattern()
    function and replaces any unusual license type by the more general!!!
    For example: ['GENERAL CO', 'GENERAL C', 'GENERA',
                  'GENERAL CONTRACTOR LICENSE', 'GENERAL ',
                  'GENERAL CONT', 'GENERAL CONTRA', 'GENERAL']

    will be replaces with 'GENERAL CONTRACTOR LICENSE'
    """

    temp_list = []
    for i in inp_list:
        if i == 'None':
            temp_list.append(i)
        else:
            for j in pattern:
                temp = []
                if i in j:
                    temp_list.append(max(j).lower().replace(" ", "_"))
                    break
    return temp_list

# BusinessName
def get_businessname(data, n):
    """Set top N businessnames as factor"""
    temp = data['businessname'].value_counts().head(n).index.values
    top_n = [i.lower().replace(" ","_") if i in temp else 'Other' for i in data['businessname']]
    return top_n

# Remove stopwords, non alphabetic characters
def nltk_description(data):

    """
    This function takes in text data in the form of description
    which is first tokenized and later cleaned by removing all stopwords,
    removing special characters, stemming each character to its root form
    and returns a string.
    """

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    num_pattern = re.compile(r'\s*[\W0-9\s]\s*')

    sample = []

    for index, description in enumerate(data["description"]):

        words = word_tokenize(description)
        no_stops = [i for i in words if i.lower() not in stop_words]
        no_special_char = [ps.stem(num_pattern.sub("",i)) for i in no_stops if ps.stem(num_pattern.sub("",i)) != '']
        descrip = " ".join(i for i in no_special_char)
        sample.append(descrip)

    return sample


def encode_subtype(data):
    """Encode subtype using OneHotEncoding"""
    data.loc[:,'subtype'] = data.loc[:,'subtype'].fillna('None')
    z = data.loc[:,['subtype']].values
    y = OneHotEncoder().fit_transform(z).toarray()
    return y


def data_preprocessing(data):

    # license type
    data.loc[:,'licensetype'] = data.loc[:,'licensetype'].fillna('None')
    data.loc[:,'licensetype'] = data.loc[:,'licensetype'].apply(lambda x: x.split(', ')).apply(lambda x: list(set(x)))

    pattern = get_pattern(data)
    cleaned_license = [clean_license(item, pattern) for item in data['licensetype']]
    cleaned_license = ['-'.join(sorted(i)) for i in cleaned_license]
    data.loc[:,'licensetype'] = cleaned_license

    # Set top business names as factors
    data['businessname'].fillna('None', inplace=True)
    data.loc[:,'businessname'] = get_businessname(data, 100)

    # Set binary value for legal description
    data.loc[:,'legaldescription'] = data['legaldescription'].fillna('None')
    data.loc[:,'has_ld'] = [1 if i!='None' else 0 for i in data['legaldescription']]

    # tfidf for description
    data.loc[:,'description'] = data.loc[:,'description'].fillna('None')
    data.loc[:,'description'] = nltk_description(data)

    # Subtype
    data['subtype'] = data['subtype'].fillna('None')
    data['subtype'] = [i.lower().replace(" ", "_") for i in data['subtype']]

    # Job Value
    cleaned_job_value = data['job_value'].apply(lambda x: float(str(x).replace('$', '').replace(',','')))
    data.loc[:,'job_value'] = cleaned_job_value
    data.loc[:,'job_value'] = data['job_value'].fillna(0.0)

    return data.loc[:, ~data.columns.isin(['legaldescription'])]


def machine_learning_prep(train_x, test_x, test_y):

    # Apply data preprocessing
    train_X = train_x.loc[:,~train_x.columns.isin(['type'])].copy()
    train_y = train_x['type'].apply(lambda x: 1 if x=='ELECTRICAL' else 0)
    test_X = test_x.copy()
    test_y = test_y

    train_x = data_preprocessing(train_X)
    test_x = data_preprocessing(test_X)

    colNames = ['description', 'licensetype', 'businessname', 'subtype', 'job_value', 'has_ld']
    train_x = train_x[colNames]
    test_x = test_x[colNames]

    # Prep training data
    X = train_x.values
    y = train_y.values
    X_test = test_x.iloc[:25148,:].values
    y_test = test_y.values

    # Train Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.25,
                                                      random_state=1,
                                                      stratify=y)

    # print("Make sure stratification works and we have equal split across train-val-test sets")
    # print("Train split: ", y_train.sum()/len(y_train))
    # print("Validation split: ", y_val.sum()/len(y_val))
    # print("Test split: ", y_test.sum()/len(y_test))
    # print("\n")

    # # Transform licensetype, businessname and subtype using OneHotEncoding
    # column_trans = make_column_transformer((OneHotEncoder(sparse=False, handle_unknown='ignore'), [0, 1, 3]),
    #                                        remainder='passthrough')
    #
    # X_train = column_trans.fit_transform(X_train)
    # X_val = column_trans.transform(X_val)
    # X_test = column_trans.transform(X_test)
    #
    #
    # # Transform Description using tf-idf
    # tf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=False)
    #
    # # Sum features to single feature
    # X_train[:,-3] = tf.fit_transform(X_train[:,-3]).toarray().sum(axis=1)
    # X_val[:,-3] = tf.transform(X_val[:,-3]).toarray().sum(axis=1)
    # X_test[:,-3] = tf.transform(X_test[:,-3]).toarray().sum(axis=1)


    # Transform licensetype, businessname and subtype using OneHotEncoding
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    cat_X_train = ohe.fit_transform(X_train[:,1:4])
    cat_X_val = ohe.transform(X_val[:,1:4])
    cat_X_test = ohe.transform(X_test[:,1:4])


    # Transform Description using tf-idf
    tf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=False)
    text_X_train = tf.fit_transform(X_train[:,0]).toarray()
    text_X_val = tf.transform(X_val[:,0]).toarray()
    text_X_test = tf.transform(X_test[:,0]).toarray()

    # Concatenate all feature transformations
    X_train = np.concatenate((cat_X_train, text_X_train, X_train[:,4:]), axis=1)
    X_val = np.concatenate((cat_X_val, text_X_val, X_val[:,4:]), axis=1)
    X_test = np.concatenate((cat_X_test, text_X_test, X_test[:,4:]), axis=1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_scores(classifier, X_val, y_val, X_test, y_test):

    # Evaluating Model
    y_pred = classifier.predict(X_val)
    cm_val = confusion_matrix(y_val, y_pred)
    accuracy_val = accuracy_score(y_val, y_pred)

    # Evaluating model on test
    y_test_pred = classifier.predict(X_test)
    cm_test = confusion_matrix(y_test, y_test_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    print('Validation Set Metrics: ')
    print('-----------------------')
    print('Confusion matrix: \n', cm_val)
    print('\nF1 Score: ', 2*cm_val[1, 1]/(2*cm_val[1, 1]+cm_val[0, 1]+cm_val[1, 0]))
    print('\nPrecision: ', cm_val[1, 1]/(cm_val[1, 1]+cm_val[0, 1]))
    print('\nAccuracy: ', accuracy_val)
    print('\nRecall/Sensitivity: ', cm_val[1, 1]/(cm_val[1, 1]+cm_val[1, 0]))
    print('\nSpecificity: ', cm_val[0, 0]/(cm_val[0, 0]+cm_val[0, 1]))

    print('\n')
    print('Test Set Metrics: ')
    print('-----------------')
    print('Confusion matrix: \n', cm_test)
    print('\nF1 Score: ', 2*cm_test[1, 1]/(2*cm_test[1, 1]+cm_test[0, 1]+cm_test[1, 0]))
    print('\nPrecision: ', cm_test[1, 1]/(cm_test[1, 1]+cm_test[0, 1]))
    print('\nAccuracy: ', accuracy_test)
    print('\nRecall/Sensitivity: ', cm_test[1, 1]/(cm_test[1, 1]+cm_test[1, 0]))
    print('\nSpecificity: ', cm_test[0, 0]/(cm_test[0, 0]+cm_test[0, 1]))
