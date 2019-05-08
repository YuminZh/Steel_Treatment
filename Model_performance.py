#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:04:10 2019

@author: admin
"""

import pandas as pd
import numpy
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
import io 
import pydotplus
from scipy import misc 

# =============================================================================
# Declaration of functions 
# =============================================================================
def AdaBoostedRegression(features, target, feature_names, plot_title):
    features_train, features_test, target_train, target_test = cross_validation.train_test_split(features, target, test_size=0.2)
    regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, random_state=0) #add random_state? 
    regressor.fit(features_train,target_train)

    score = regressor.score(features_train,target_train)
    
    #Evaluate performance of Adaboost regressor 
    target_pred = regressor.predict(features_test)
    mse = mean_squared_error(target_test,target_pred)
    evs = explained_variance_score(target_test,target_pred)
    print("\nADABOOST REGRESSOR")
    print("Mean squared error =", round(mse,2))
    print("Explained variance score =", round(evs,2))
    print("regressor score =", score)
    # extract features importances 
    feature_importances = regressor.feature_importances_
    # feature_names = comp_data.keys()

    #Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    #sort the values and flip them 
    index_sorted = np.flipud(np.argsort(feature_importances))

    #arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    #plot the bar graph 
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align = 'center')
    plt.xticks(pos,feature_names[index_sorted], rotation=45,ha='right',rotation_mode='anchor', fontsize = 32)
    indy=np.arange(0,120,step=20)
    plt.yticks(indy,fontsize=32)
    plt.ylabel ('Relative importance', fontsize = 34)
    #plt.title(plot_title)
    plt.savefig(plot_title)
    plt.show()

    return;

def show_tree(tree, features, path):
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=misc.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)
    return;

def f(x):
    return x;
#def Model_predict():
#    return;
# =============================================================================
# Evaluate the influence of composition on properties  
# =============================================================================

prop_data = pd.read_csv('properties.csv')
# in the property data, 'prop_data.keys()' gives creep rupture time CRT, creep temp CT, rupture stress RS
# and rupture temperature RT, here independent variable is creep rupture time, and rupture stress, CT and RT is the same 

CRT = prop_data.values[:,0] # creep rupture time
CRT = CRT[:,None]
RS = prop_data.values[:,2] # rupture stress
RS = RS[:,None]

comp_data = pd.read_csv('composition.csv')
comp_value = comp_data.values
feature_names = comp_data.keys()
plot_title = 'Feature importance of Composition for Creep Rupture Time using Adaboost regressor'
AdaBoostedRegression(comp_value, CRT, feature_names, plot_title)

plot_title = 'Feature importance of Composition for Rupture stress using Adaboost regressor'
AdaBoostedRegression(comp_value, RS, feature_names, plot_title)

# =============================================================================
# Evaluate the influence of preparation on properties 
# =============================================================================
prep_data = pd.read_csv('preparation.csv')
prep_data.replace('Furnace cool',0.03, inplace=True) 
prep_data.replace('Air cool',0.88, inplace=True)
prep_data.replace('Water quench',40.00, inplace=True)
prep_data.replace('Oil quench',16.67, inplace=True)

prep_value = prep_data.values
feature_names = prep_data.keys()
plot_title = 'Feature importance of Preparation for Creep Rupture Time using Adaboost regressor'
AdaBoostedRegression(prep_value, CRT, feature_names, plot_title)

plot_title = 'Feature importance of Preparation for Rupture Stress using Adaboost regressor'
AdaBoostedRegression(prep_value, RS, feature_names, plot_title)
# =============================================================================
# Evaluate the influence of all on properties 
# =============================================================================
temp_data = prop_data.values[:,1]
temp_data = temp_data[:,None]
all_ind = np.concatenate([comp_data.values, prep_data.values, temp_data], axis=1)

#feature_names = np.concatenate([comp_data.keys(), prep_data.keys()])
#feature_names = numpy.append(feature_names,'Creep Temp/Rupture Temp')
feature_names = ['C','Si','Mn','P','S','Cr','Mo','W','Ni','Cu','V','Nb','N','Al','B','Co','Ta','O','Re','Fe','NTE','NTI','TTE','TTI','TR','ATE','ATI','CT/RT']
feature_names = array(feature_names)
plot_title = 'All Features for Creep Rupture Time'
AdaBoostedRegression(all_ind, CRT, feature_names, plot_title)

plot_title = 'All Features for Rupture Stress'
AdaBoostedRegression(all_ind, RS, feature_names, plot_title)

# =============================================================================
# Prediction using the produced models
# =============================================================================

# model fitted for predicting the rupture stress 

#
#all_train, all_test, target_train, target_test = cross_validation.train_test_split(all_ind, RS, test_size=0.2)
#regr = DecisionTreeRegressor(max_depth=6)
#dt=regr.fit(all_train,target_train)
##show_tree(dt, feature_names,'dt_RS.png')
#
#regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, random_state=0) #add random_state? 
#regressor.fit(all_train,target_train)
##data_test = all_ind[0]
##data_test = data_test[None,:]
##RS_pred = regressor.predict(data_test)
##print("<<<Predicted Rupture stress>>> = ", RS_pred)
##RS_pred = regressor.predict(all_ind)
#RS_pred = regressor.predict(all_test)
##RS_measure = RS
#RS_measure = target_test
#R2 = r2_score(RS_measure,RS_pred)
#MSE = mean_squared_error(RS_measure,RS_pred)
#print("\THE MEAN SQUARE ERROR OF PREDICTION AND MEASUREMENTS = ", MSE)
#print("\THE R2 OF PREDICTION AND MEASUREMENTS = ", R2)
#X_axis = np.linspace(1,413,413)
#plt.plot(X_axis,RS_measure,color = 'k')
#plt.plot(X_axis,RS_pred, 'g')
#plt.show()
#
#n=10
#RS_measure_selec=[]
#RS_pred_selec = []
#for i in range(0,413):
#    if i%n==0:
#        dum = RS_measure.item(i)
#        RS_measure_selec.append(dum)
#        dum1 = RS_pred.item(i)
#        RS_pred_selec.append(dum1)
#
#limit = n*(len(RS_measure_selec)-1)      
#X_axis = np.linspace(0,limit,len(RS_measure_selec))
#plt.plot(X_axis,RS_measure_selec,color = 'r', linewidth=3.0)
#plt.plot(X_axis,RS_pred_selec, 'b', linewidth=3.0)
#plt.ylabel ('Rupture Stress MPa', fontsize = 34)
#plt.xlabel('Sample Index', fontsize=34)
#plt.legend(['RS measured', 'RS predicted'], fontsize=32) 
#indx=np.arange(0,410,step=50)
#indy=np.arange(0,500,step=50)
##plt.title("Experimental Rupture Stress Compares to Predicticed with AdaBoosted Regression", fontsize=20)
#plt.xticks(indx, fontsize=32)
#plt.yticks(indy,fontsize=32)
# 
#plt.savefig('Rupture exp and predict.png')     
#plt.show()  


# =============================================================================
# make the new set of features 
# =============================================================================
prop_data_mod = prop_data.drop(columns = 'Rupture temp')
comp_data_mod = comp_data.drop(columns = ['Nb','S','B','O','Ta','Re','Co'])
prep_data_mod = prep_data.drop(columns = ['Annealing temperature','Annealing time', 'Tempering cooling rate'])
all_ind_mod = np.concatenate([comp_data_mod.values, prep_data_mod.values, temp_data], axis=1)

all_train_mod, all_test_mod, target_train_mod, target_test_mod = cross_validation.train_test_split(all_ind_mod, RS, test_size=0.2)
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, random_state=0)
regressor.fit(all_train_mod,target_train_mod)
RS_pred_mod = regressor.predict(all_test_mod)
RS_measure_mod = target_test_mod

R2 = r2_score(RS_measure_mod,RS_pred_mod)
MSE = mean_squared_error(RS_measure_mod,RS_pred_mod)
print("\THE MEAN SQUARE ERROR OF PREDICTION AND MEASUREMENTS = ", MSE)
print("\THE R2 OF PREDICTION AND MEASUREMENTS = ", R2)
X_axis = np.linspace(1,413,413)
plt.plot(X_axis,RS_measure_mod,color = 'k')
plt.plot(X_axis,RS_pred_mod, 'g')
plt.show()


x = np.arange(0.0,500.0,10.0)
plt.plot(RS_pred_mod, RS_measure_mod,'bo',markersize = 18)
plt.plot(x,f(x), 'k--', linewidth = 3.0)
plt.axis([0,500,0,500])
indx=np.arange(0,500,step=50)
plt.xticks(indx, fontsize=32)
plt.yticks(indx,fontsize=32)
plt.ylabel ('Predicted Rupture Stress MPa', fontsize = 34)
plt.xlabel('Experimental Rupture Stress MPa', fontsize=34)
plt.savefig('PVA_rupture_selec.png')
plt.show()

n=10
RS_measure_selec=[]
RS_pred_selec = []
for i in range(0,413):
    if i%n==0:
        dum = RS_measure_mod.item(i)
        RS_measure_selec.append(dum)
        dum1 = RS_pred_mod.item(i)
        RS_pred_selec.append(dum1)

limit = n*(len(RS_measure_selec)-1)      
X_axis = np.linspace(0,limit,len(RS_measure_selec))
plt.plot(X_axis,RS_measure_selec,color = 'r', linewidth=3.0)
plt.plot(X_axis,RS_pred_selec, 'b', linewidth=3.0)
plt.ylabel ('Rupture Stress MPa', fontsize = 34)
plt.xlabel('Sample Index', fontsize=34)
plt.legend(['RS measured', 'RS predicted'], fontsize=32) 
indx=np.arange(0,410,step=50)
indy=np.arange(0,500,step=50)
#plt.title("Experimental Rupture Stress Compares to Predicticed with AdaBoosted Regression with Modified Features", fontsize=20)
plt.xticks(indx, fontsize=32)
plt.yticks(indy,fontsize=32)
 
plt.savefig('Rupture exp and predict modify features.png')     
plt.show() 
