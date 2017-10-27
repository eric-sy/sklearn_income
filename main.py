#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:28:31 2017

Main file, import other modules.

@author: EricYang
"""
import graphviz 
import pandas
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import preparation



# =============================================================================
# Data Preparation and Transformation
# =============================================================================

dataframe = preparation.read_file()
preparation.prepare_data(dataframe)

# =============================================================================
# Select Features
# =============================================================================
# Get all the columns from the dataframe.
columns = dataframe.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["occupation","native-country","income"]]
# Store the variable we'll be predicting on.
target = "income"
target_names=["low-income","high-income"]


# =============================================================================
# Split into train and test sets
# =============================================================================
# Generate the training set.  Set random_state to be able to replicate results.
train = dataframe.sample(frac=0.80, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = dataframe.loc[~dataframe.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# =============================================================================
# Decision Tree
# =============================================================================
print("===========================Decision Tree==============================")
"""
# Training
"""
# Initial Tree
tree_model = tree.DecisionTreeClassifier(min_samples_split=0.002,
                                  min_samples_leaf=0.001,
                                  random_state=1,
                                  max_leaf_nodes=30)

tree_model = tree_model.fit(train[columns], train[target])

"""
# Visualization
"""
dot_data = tree.export_graphviz(tree_model, out_file=None,
                                feature_names=columns,  
                                class_names=target_names,  
                                label="all",
                                filled=True, rounded=True,  
                                special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("tree-result") 

"""
# Assessment
"""
predictions = tree_model.predict(test[columns])

actual=list(test[target])
predictions=list(predictions)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Decistion Tree ROC Curve')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

diff=0;
for i in range(len(actual)):
    if(actual[i] != predictions[i]):
        diff=diff+1
print("Accuracy:"+ str(1-diff/len(actual)))


# =============================================================================
# Neural Network
# =============================================================================
print("===========================Neural Network==============================")
"""
# Training
"""
MLP_model = MLPClassifier(solver='adam',alpha=1e-04, batch_size='auto',
                         learning_rate='adaptive',learning_rate_init=0.001,
                         random_state=1,beta_1=0.9, beta_2=0.999,
                         epsilon=1e-08, hidden_layer_sizes=(150,),
                         max_iter=200, momentum=0.9, nesterovs_momentum=True, 
                         power_t=0.5,shuffle=True,tol=0.0001, 
                         validation_fraction=0.1, verbose=False,
                         warm_start=False)
       
MLP_model=MLP_model.fit(train[columns], train[target])     

"""
# Assessment
"""
predictions = MLP_model.predict(test[columns])
actual=list(test[target])
predictions=list(predictions)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Neural Network ROC Curve')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

diff=0;
for i in range(len(actual)):
    if(actual[i] != predictions[i]):
        diff=diff+1
print("Accuracy:"+ str(1-diff/len(actual)))

# =============================================================================
# Support Vector Machines
# =============================================================================
print("======================Support Vector Machines===========================")
"""
# Training
"""
SVC_model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

SVC_model=SVC_model.fit(train[columns], train[target])     

"""
# Assessment
"""
predictions = SVC_model.predict(test[columns])
actual=list(test[target])
predictions=list(predictions)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Support Vector Machines ROC Curve')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

diff=0;
for i in range(len(actual)):
    if(actual[i] != predictions[i]):
        diff=diff+1
print("Accuracy:"+ str(1-diff/len(actual)))

# =============================================================================
# Support Vector Machines
# =============================================================================
#print("==================Stochastic Gradient Descent==========================")
#"""
## Training
#"""
#SGD_model = SGDClassifier(loss="hinge", penalty="l2")
#SGD_model.fit(train[columns], train[target])  
#
#"""
## Assessment
#"""
#predictions = SGD_model.predict(test[columns])
#actual=list(test[target])
#predictions=list(predictions)
#false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#plt.title('Stochastic Gradient Descent ROC Curve')
#plt.plot(false_positive_rate, true_positive_rate, 'b',
#label='AUC = %0.2f'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.1,1.2])
#plt.ylim([-0.1,1.2])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
#
#diff=0;
#for i in range(len(actual)):
#    if(actual[i] != predictions[i]):
#        diff=diff+1
#print("Accuracy:"+ str(1-diff/len(actual)))  
