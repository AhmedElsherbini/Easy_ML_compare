#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:36:18 2020

@author: ahmed elsherbini
"""
##########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,), dtype={int, float}
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

"""
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
"""
#########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import preprocessing, svm
#from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import fnmatch
#######################################################################
print("make sure that all of your files are csv files and the first column start with class")
print("Ahmed Elsherbini")
#####################################################################################################
#import you multidimesion data
#lets do 2D PCA
fg = input ("1-do you want to do a 2D PCA? y/b/n: ")
if (fg == "y"):
    zed = input("what is the name of you .csv file?")
    
    data = pd.read_csv(zed)
    data = data.fillna(data.mean())
    
    y = data.loc[:,['class']].values #the first column values
    
    targets = list(np.unique(y))
    
    features = list(data.iloc[:0]) #the first raw which are the variables
    features.pop(0) #remove month
    
    x = data.loc[:, features].values #extract all values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, data[['class']]], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PCA1=%f'% pca.explained_variance_ratio_[0], fontsize = 15)
    ax.set_ylabel('PCA2=%f'% pca.explained_variance_ratio_[1], fontsize = 15)
    ax.set_title('2D PCA %s' %(zed), fontsize = 20)
    colors = ['r','g','y','b','c','k','#f48ec5','#f48ec5','#c670c0','#a71db2','#f40efa','#4b34ca','#fa7d66','#37b0a4','#beda3f','#c52553','#ac6895','#7ca77b','#66d0ee','#06c405','#2b97c2','#292927','#1d9375','#e3ec51','#7dfe45','#f48f84','#6df17a','#2470d0','#a0f4a0','#7380b9','#c31bd0','#9be737','#445397','#12c693','#9a853f','#070740','#a65530','#e637a5','#b23399','#a29316','#a0ebe0','#ef955c','#f7b512','#a062a4','#ce6a1c','#e2e7fa','#8a26d5','#e3fe76','#adf8e6','#12dfde','#ae653b','#82a36c','#416d7e','#a5d12e','#e4815c','#d95602','#9a6616','#3029d2','#532c7f','#528525','#51eff3','#fb234f','#d13750','#cd0135','#e24c95','#c507b6','#e20241','#2f90df','#36f5d0','#cfb087','#970826','#fa5265','#4241f1','#346331','#b79c2e','#d56f1d','#fad7f5','#6d2f1b','#e0e8b1','#261d62','#00e899','#4e6cfb','#77bdc4','#aa1c5c','#ccd9a0','#fad2bc','#cd746b','#71bca7','#45c442','#68aa78','#046d99','#d0fd75','#0ca4d9','#73db67','#14401c','#17c915','#17bad5','#c3b9fb','#c7a7a5','#fd3ae7','#b17f7b','#174b9e','#a8a3fd','#f3a354','#587d92','#4d752b']
    
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['class'] == target
        color = np.random.rand(3,)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = [color]
                   , s = 50)
    ax.legend(targets) #bbox_to_anchor=(0.8, 0.8)
    ax.grid()
    plt.savefig("2D_PCA_of_%s.jpg" %(zed)) #save your file!
    plt.close()

elif (fg == "b"):
    print("make sure you have a batch of  .csv files")
    for f in os.listdir():
        if fnmatch.fnmatch(f, '*.csv'):
            data = pd.read_csv(f)
            data = data.fillna(data.mean())
            
            y = data.loc[:,['class']].values #the first column values
            
            targets = list(np.unique(y))
            
            features = list(data.iloc[:0]) #the first raw which are the variables
            features.pop(0) #remove month
            
            x = data.loc[:, features].values #extract all values
            x = StandardScaler().fit_transform(x)
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(x)
            principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
            finalDf = pd.concat([principalDf, data[['class']]], axis = 1)
            
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PCA1=%f'% pca.explained_variance_ratio_[0], fontsize = 15)
            ax.set_ylabel('PCA2=%f'% pca.explained_variance_ratio_[1], fontsize = 15)
            ax.set_title('2D PCA %s' %(f), fontsize = 20)
            colors = ["r","g","y","b","c","k","#D226D2", "#D2CF26", "#EAD66A"]
            
            for target, color in zip(targets,colors):
                indicesToKeep = finalDf['class'] == target
                color = np.random.rand(3,)
                ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                           , finalDf.loc[indicesToKeep, 'principal component 2']
                           , c = [color]
                           , s = 50)
            ax.legend(targets)
            ax.grid()
            plt.savefig("2D_PCA_of_%s.jpg" %(f)) #save your file!
            plt.close()
            
            
            
    

#################################################################################
#let' do the unsupervised data analysis
fj = input ("3- do you want to use supervised data analysis (b for batch mode)? y/b/n: ")
if (fj == "y"):
    elseh = input("what is the name of your .csv?")
    data = pd.read_csv(elseh)
    data = shuffle(data)
    data = data.fillna(data.mean())
    label = data['class']
    features = data.drop(columns =['class'])
    print('data shape (samples/columns, raws/variables) = ', data.shape)
    
    ###########################################
    
    Features = np.asarray(features, dtype= float)
    Labels = np.asarray(label) 
    
    X_train, X_val, y_train, y_val = train_test_split(Features, Labels, test_size=0.3, random_state=42, stratify= Labels)
    #X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify= y_val)
    
    #print("training_dat=",np.unique(y_train, return_counts= True))
    #print("validing_data =",np.unique(y_val, return_counts= True))
    #print(np.unique(y_test, return_counts= True))
    ###################################################
    def train_test(clf, X_train, y_train, X_val, y_val):
        clf.fit(X_train, y_train)
    
        y_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_performance = train_acc
        
        y_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        val_performance = val_acc
        
        return clf, train_performance, val_performance
    
    def represent_performance(clf_name, train_performance, val_performance):
        print(clf_name,'_train_accuracy =', train_performance)
        print(clf_name,'_valid_accuracy =', val_performance)
    ##############################################################
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=9, random_state = 0)#, class_weight=loss_weights)
    DT_clf, DT_train_performance, DT_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
    represent_performance('Decision_tree', DT_train_performance, DT_val_performance)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(clf, "D.tree",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
    fig.savefig("D.Tree_%s.jpg"%(elseh))
   
    
    clf = SVC(kernel = 'linear', random_state = 0) #should be used for binary classfications
    SVM_clf, SVM_train_performance, SVM_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
    represent_performance('SVM', SVM_train_performance, SVM_val_performance)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(clf, "SVM.linear",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
    fig.savefig("SVM_%s.jpg"%(elseh))
    
    clf = LogisticRegression(random_state = 0)
    LR_clf, LR_train_performance, LR_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
    represent_performance('Logistic_regression', LR_train_performance, LR_val_performance)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(clf, "L. regression",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
    fig.savefig("L.regression_%s.jpg"%(elseh))
    
    clf = GaussianNB()
    NB_clf, NB_train_performance, NB_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
    represent_performance('NB', NB_train_performance, NB_val_performance)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(clf, "N. Bayes",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
    fig.savefig("N.basyes_%s.jpg"%(elseh))
    
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    RF_clf, RF_train_performance, RF_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
    represent_performance('Random_forest', RF_train_performance, RF_val_performance)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(clf, "R. forest",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
    fig.savefig("R.forest_%s.jpg"%(elseh))
    
    # save the model to disk
    #filename = 'finalized_RF_model.sav'
    #pickle.dump(RF_clf, open(filename, 'wb'))
    ####################################################################
    Col = features.columns
    Features = np.asarray(features, dtype= float)
    Labels = np.asarray(label)
    X_train, X_valid, Y_train, Y_valid = train_test_split(Features, Labels, test_size=0.5, random_state=42, stratify= Labels)
    
    print(np.unique(Y_train, return_counts= True))
    print(np.unique(Y_valid, return_counts= True))
    
    
    
    
    clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=5, random_state = 0)
    clf.fit(X_train, Y_train)
    
    y_pred = clf.predict(X_train)
    train_acc = accuracy_score(Y_train, y_pred)
    print('train accuracy = ', train_acc)
    
    y_pred = clf.predict(X_valid)
    test_acc = accuracy_score(Y_valid, y_pred)
    print('test accuracy = ', test_acc)
    
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                       feature_names=Col,  
                       filled=True)
    
    fig.savefig("D.Tree_%s.jpg"%(elseh))
    plt.close()
    
    
    #lets' print out our tree as atext''
    f = open('DT_text%s.txt'%(elseh),'w')
    a = data.columns[1:].tolist()
    text_representation = tree.export_text(clf,  feature_names= a)
    print(text_representation,file=f)
    f.close()
    
    x=['SVM','L.Regression', 'N.Bayes','D.Tree', 'R.Forest']
    
    y=[SVM_train_performance,LR_train_performance, NB_train_performance, DT_train_performance,RF_train_performance]

    z=[SVM_val_performance,LR_val_performance, NB_val_performance, DT_val_performance,RF_val_performance]
    fig, ax = plt.subplots()
    ax.bar(x, y, label='train')
    ax.bar(x, z, label='valid')
    ax.set_ylabel('Accuracy' ,  color = "r")
    ax.set_title('models_comparisons_%s'%(elseh),  color = "r")
    fig.legend()
    plt.show()
    plt.savefig('ML_of_%s.jpg'%(elseh))
    plt.close()
    
    

    
#####################################
elif (fj== 'b'):
     for fathy in os.listdir():
        if fnmatch.fnmatch(fathy, '*.csv'):
                data = pd.read_csv(fathy)
                data = shuffle(data)
                data = data.fillna(data.mean())
                label = data['class']
                features = data.drop(columns =['class'])
                print('data shape', data.shape)
                
                ###########################################
                
                Features = np.asarray(features, dtype= float)
                Labels = np.asarray(label) 
                
                X_train, X_val, y_train, y_val = train_test_split(Features, Labels, test_size=0.3, random_state=42, stratify= Labels)
                #X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify= y_val)
                
                print(np.unique(y_train, return_counts= True))
                print(np.unique(y_val, return_counts= True))
                #print(np.unique(y_test, return_counts= True))
                ###################################################
                def train_test(clf, X_train, y_train, X_val, y_val):
                    clf.fit(X_train, y_train)
                
                    y_pred = clf.predict(X_train)
                    train_acc = accuracy_score(y_train, y_pred)
                    train_performance = train_acc
                    
                    y_pred = clf.predict(X_val)
                    val_acc = accuracy_score(y_val, y_pred)
                    val_performance = val_acc
                    
                    return clf, train_performance, val_performance
                
                def represent_performance(clf_name, train_performance, val_performance):
                    print(clf_name,'_train_accuracy =', train_performance)
                    print(clf_name,'_valid_accuracy =', val_performance)
                ##############################################################
                clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=9, random_state = 0)#, class_weight=loss_weights)
                DT_clf, DT_train_performance, DT_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
                represent_performance('Decision_tree', DT_train_performance, DT_val_performance)
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                plot_learning_curve(clf, "D.tree",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
                fig.savefig("D.Tree_%s.jpg"%(fathy))
               
                
                clf = SVC(kernel = 'linear', random_state = 0) #should be used for binary classfications
                SVM_clf, SVM_train_performance, SVM_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
                represent_performance('SVM', SVM_train_performance, SVM_val_performance)
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                plot_learning_curve(clf, "SVM.linear",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
                fig.savefig("SVM_%s.jpg"%(fathy))
                
                clf = LogisticRegression(random_state = 0)
                LR_clf, LR_train_performance, LR_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
                represent_performance('Logistic_regression', LR_train_performance, LR_val_performance)
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                plot_learning_curve(clf, "L. regression",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
                fig.savefig("L.regression_%s.jpg"%(fathy))
                
                clf = GaussianNB()
                NB_clf, NB_train_performance, NB_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
                represent_performance('NB', NB_train_performance, NB_val_performance)
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                plot_learning_curve(clf, "N. Bayes",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
                fig.savefig("N.basyes_%s.jpg"%(fathy))
                
                clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
                RF_clf, RF_train_performance, RF_val_performance = train_test(clf, X_train, y_train, X_val, y_val)
                represent_performance('Random_forest', RF_train_performance, RF_val_performance)
                cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                plot_learning_curve(clf, "R. forest",X_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv ,n_jobs=4)
                fig.savefig("R.forest_%s.jpg"%(fathy))
                # save the model to disk
                #filename = 'finalized_RF_model.sav'
                #pickle.dump(RF_clf, open(filename, 'wb'))
                ####################################################################
                Col = features.columns
                Features = np.asarray(features, dtype= float)
                Labels = np.asarray(label)
                X_train, X_valid, Y_train, Y_valid = train_test_split(Features, Labels, test_size=0.5, random_state=42, stratify= Labels)
                
                print(np.unique(Y_train, return_counts= True))
                print(np.unique(Y_valid, return_counts= True))
                
                
                clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=5, random_state = 0)
                clf.fit(X_train, Y_train)
                
                y_pred = clf.predict(X_train)
                train_acc = accuracy_score(Y_train, y_pred)
                print('train accuracy = ', train_acc)
                
                y_pred = clf.predict(X_valid)
                test_acc = accuracy_score(Y_valid, y_pred)
                print('test accuracy = ', test_acc)
                
                fig = plt.figure(figsize=(25,20))
                _ = tree.plot_tree(clf, 
                                   feature_names=Col,  
                                   filled=True)
                
                fig.savefig("DT_%s.jpg"%(fathy))
                plt.close()
                 #lets' print out our tree as atext''
                f = open('D.Tree_text_%s.txt'%(fathy),'w')
                a = data.columns[1:].tolist()
                text_representation = tree.export_text(clf,  feature_names= a)
                print(text_representation,file=f)
                f.close() 
                ###################
                x=['SVM','L.Regression', 'N.Bayes','D.Tree', 'R.Forest']
                y=[SVM_train_performance,LR_train_performance, NB_train_performance, DT_train_performance,RF_train_performance]
                z=[LR_val_performance, NB_val_performance, DT_val_performance,RF_val_performance]
                fig, ax = plt.subplots()
                ax.bar(x, y, label='train')
                ax.bar(x, z, label='valid')
                ax.set_ylabel('Accuracy' ,  color = "r")
                ax.set_title('models_comparisons_%s'%(fathy),  color = "r")
                fig.legend()
                plt.show()
                plt.savefig('ML_of_%s.jpg'%(fathy))
                plt.close()
                
            
                            
                       
