# -*- coding: utf-8 -*-
# program to detect gender given name

import codecs
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import timeit
from sklearn import preprocessing
from sklearn.externals import joblib


def read_file(file_path):
    dic_name_details = {}
    with codecs.open(file_path,'rU','utf-8') as fl:
        for line in fl.readlines():
            tokens = line.split('\t')
            tokens[4] = tokens[4].strip()
            if dic_name_details.has_key(tokens[4]):
                tuple = dic_name_details[tokens[4]] 
                new_tuple = (int(tuple[0])+int(tokens[1]), int(tuple[1])+int(tokens[2]), int(tuple[2])+int(tokens[3]))
                dic_name_details[tokens[4]] = new_tuple
            else:
                new_tuple = (tokens[1], tokens[2], tokens[3])
                dic_name_details[tokens[4]] = new_tuple
    return dic_name_details

def get_gender(dic_name_details):
    dic_name_gender = {}
    for name in dic_name_details.keys():
        if dic_name_details[name][1]>0 and dic_name_details[name][2] ==0 :
            dic_name_gender[name] = "male"
        if dic_name_details[name][2]>0 and dic_name_details[name][1] ==0 :
            dic_name_gender[name] = "female"
            
    return dic_name_gender

def extract_features(name):
        name=name.upper()
        if name[-1] in 'AEIOUY':
            a = 1
        else:
            a = 0
            
        return [ord(name[-1]) - 65, ord(name[-2]) - 65, ord(name[-3]) - 65, a, int(str(ord(name[-1]) - 65) + 
                                                                                   str(ord(name[-2]) - 65) + 
                                                                                   str(ord(name[-3]) - 65))]
    
def generate_training_data(dic_name_gender):
    training_data = []
    training_label = []
    le = preprocessing.LabelEncoder()
    le.fit(["male", "female"])
    
    joblib.dump(le, 'gender_lb.pkl')
    for name in dic_name_gender.keys():
        if len(name) >2:
            features = extract_features(name)
            training_data.append(features)
            
            training_label.append(le.transform([dic_name_gender[name]]))
#             training_label.append(a)
    return numpy.array(training_data), numpy.array(training_label)

def train_model(training_data, training_label):
   
#     print type(tfidf.toarray())
#     print type(label)
    X_train, X_test, y_train, y_test = train_test_split(training_data,training_label, test_size=0.3, random_state=0)
#     pprint.pprint(y_test[0])
#     print numpy.unique(y_train)
    print "X_test.shape ",X_test.shape
    print "X_train.shape ",X_train.shape
    print "y_train.shape ", y_train.shape
    print "y_test.shape ", y_test.shape
    print '\nTraining model through sklearn...'


    start = timeit.default_timer()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
#     print clf.predict(X_test[10])
    print "Accuracy RF ",clf.score(X_test, y_test)
    stop = timeit.default_timer()
    print "Time RF ", stop - start, '\n'
    joblib.dump(clf, 'gender_model.pkl') 
    
    start = timeit.default_timer()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
#     print gnb.predict(X_test[10])
    print "Accuracy NB ",gnb.score(X_test, y_test)
    stop = timeit.default_timer()
    print "Time NB ", stop - start, '\n'
    return clf
 
 
def predict_gender(test_str): 
    
    clf = joblib.load('gender_model.pkl') 
    lb = joblib.load('gender_lb.pkl') 
    x_test = numpy.array(extract_features(test_str))
    predicted = clf.predict(x_test)
    label = lb.inverse_transform(predicted)
    print '%s == %s' % (test_str, label)
    return label
                        
if __name__ == '__main__':
    dic_name_details = read_file("name_stat.txt")
    dic_name_gender = get_gender(dic_name_details)
    training_data, training_label = generate_training_data(dic_name_gender)
    clf = train_model(training_data, training_label)
    predict_gender("asadd")
