#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:10:19 2017

Data Preparation and transformation modules

@author: SongYang

"""

import pandas

# =============================================================================
# Read data
# =============================================================================
def read_file():
    dataframe = pandas.read_csv("income.csv")
#    print(dataframe.columns)
#    print(dataframe.shape)
    return dataframe

# =============================================================================
# Data transformation
# =============================================================================
def prepare_data(dataframe):
    dataframe["income"].replace("<=50K",0,inplace=True)
    dataframe["income"].replace(">50K",1,inplace=True)
    
    dataframe["workclass"].replace("Private","1",inplace=True)
    dataframe["workclass"].replace("Self-emp-not-inc","2",inplace=True)
    dataframe["workclass"].replace("Self-emp-inc","2",inplace=True)
    dataframe["workclass"].replace("Federal-gov","3",inplace=True)
    dataframe["workclass"].replace("Local-gov","3",inplace=True)
    dataframe["workclass"].replace("State-gov","3",inplace=True)
    dataframe["workclass"].replace("Without-pay","4",inplace=True)
    dataframe["workclass"].replace("Never-worked","5",inplace=True)
    
    dataframe["marital-status"].replace("Married-civ-spouse","1",inplace=True)
    dataframe["marital-status"].replace("Divorced","2",inplace=True)
    dataframe["marital-status"].replace("Never-married","3",inplace=True)
    dataframe["marital-status"].replace("Separated","4",inplace=True)
    dataframe["marital-status"].replace("Widowed","5",inplace=True)
    dataframe["marital-status"].replace("Married-spouse-absent","1",inplace=True)
    dataframe["marital-status"].replace("Married-AF-spouse","1",inplace=True)
    
    dataframe["gender"].replace("Male","1",inplace=True)
    dataframe["gender"].replace("Female","2",inplace=True)
    
    dataframe["race"].replace("White","1",inplace=True)
    dataframe["race"].replace("Black","2",inplace=True)
    dataframe["race"].replace("Asian-Pac-Islander","3",inplace=True)
    dataframe["race"].replace("Amer-Indian-Eskimo","4",inplace=True)
    dataframe["race"].replace("Other","4",inplace=True)
    
    dataframe["relationship"].replace("Husband","1",inplace=True)
    dataframe["relationship"].replace("Wife","2",inplace=True)
    dataframe["relationship"].replace("Not-in-family","3",inplace=True)
    dataframe["relationship"].replace("Unmarried","3",inplace=True)
    dataframe["relationship"].replace("Own-child","4",inplace=True)
    dataframe["relationship"].replace("Other-relative","4",inplace=True)
    
    dataframe["education-level"].replace("Preschool","1",inplace=True)
    dataframe["education-level"].replace("Primary School","3",inplace=True)
    dataframe["education-level"].replace("Secondary School","9",inplace=True)
    dataframe["education-level"].replace("Undergraduate","13",inplace=True)
    dataframe["education-level"].replace("Master or higher","16",inplace=True)
    
    dataframe["occupation"].replace("Tech-support","1",inplace=True)
    dataframe["occupation"].replace("Craft-repair","2",inplace=True)
    dataframe["occupation"].replace("Sales","3",inplace=True)
    dataframe["occupation"].replace("Exec-managerial","4",inplace=True)
    dataframe["occupation"].replace("Prof-specialty","5",inplace=True)
    dataframe["occupation"].replace("Handlers-cleaners","6",inplace=True)
    dataframe["occupation"].replace("Machine-op-inspct","7",inplace=True)
    dataframe["occupation"].replace("Adm-clerical","8",inplace=True)
    dataframe["occupation"].replace("Farming-fishing","9",inplace=True)
    dataframe["occupation"].replace("Transport-moving","10",inplace=True)
    dataframe["occupation"].replace("Priv-house-serv","11",inplace=True)
    dataframe["occupation"].replace("Protective-serv","12",inplace=True)
    dataframe["occupation"].replace("Armed-Forces","13",inplace=True)
    dataframe["occupation"].replace("Other-service","14",inplace=True)
    
    return dataframe;
