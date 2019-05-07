#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:28:45 2019

@author: Yumin Zhang 
"""

import pandas as pd
import numpy
import numpy as np
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
    plt.xticks(pos,feature_names[index_sorted], rotation=45,ha='right',rotation_mode='anchor')
    plt.ylabel ('relative importance')
    plt.title(plot_title)
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

feature_names = np.concatenate([comp_data.keys(), prep_data.keys()])
feature_names = numpy.append(feature_names,'Creep Temp/Rupture Temp')
plot_title = 'All Features for Creep Rupture Time'
AdaBoostedRegression(all_ind, CRT, feature_names, plot_title)

plot_title = 'All Features for Rupture Stress'
AdaBoostedRegression(all_ind, RS, feature_names, plot_title)

# =============================================================================
# Prediction using the produced models
# =============================================================================

# model fitted for predicting the rupture stress 


all_train, all_test, target_train, target_test = cross_validation.train_test_split(all_ind, RS, test_size=0.2)
regr = DecisionTreeRegressor(max_depth=6)
dt=regr.fit(all_train,target_train)
#show_tree(dt, feature_names,'dt_RS.png')

regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, random_state=0) #add random_state? 
regressor.fit(all_train,target_train)
#data_test = all_ind[0]
#data_test = data_test[None,:]
#RS_pred = regressor.predict(data_test)
#print("<<<Predicted Rupture stress>>> = ", RS_pred)
#RS_pred = regressor.predict(all_ind)
RS_pred = regressor.predict(all_test)
#RS_measure = RS
RS_measure = target_test
R2 = r2_score(RS_measure,RS_pred)
MSE = mean_squared_error(RS_measure,RS_pred)
print("\THE MEAN SQUARE ERROR OF PREDICTION AND MEASUREMENTS = ", MSE)
print("\THE R2 OF PREDICTION AND MEASUREMENTS = ", R2)
X_axis = np.linspace(1,413,413)
plt.plot(X_axis,RS_measure,color = 'k')
plt.plot(X_axis,RS_pred, 'g')
plt.show()

# plot the linearity line 
x = np.arange(0.0,500.0,10.0)
plt.plot(RS_measure, RS_pred,'bo',markersize = 18)
plt.plot(x,f(x), 'k--', linewidth = 3.0)
plt.axis([0,500,0,500])
indx=np.arange(0,500,step=50)
plt.xticks(indx, fontsize=32)
plt.yticks(indx,fontsize=32)
plt.ylabel ('Experimental Rupture Stress MPa', fontsize = 34)
plt.xlabel('Predicted Rupture Stress MPa', fontsize=34)
plt.savefig('PVA_rupture.png')
plt.show()


n=10
RS_measure_selec=[]
RS_pred_selec = []
for i in range(0,413):
    if i%n==0:
        dum = RS_measure.item(i)
        RS_measure_selec.append(dum)
        dum1 = RS_pred.item(i)
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
#plt.title("Experimental Rupture Stress Compares to Predicticed with AdaBoosted Regression", fontsize=20)
plt.xticks(indx, fontsize=32)
plt.yticks(indy,fontsize=32)
 
plt.savefig('Rupture exp and predict.png')     
plt.show()  



# =============================================================================
# Below is to plot figures 
# =============================================================================
#X_axis = np.linspace(1,2065,2065)
#plt.plot(X_axis,RS_measure,color = 'k')
#plt.plot(X_axis,RS_pred, 'g')
#plt.show()
#
#n=50
#RS_measure_selec=[]
#RS_pred_selec = []
#for i in range(0,2065):
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
#indx=np.arange(0,2050,step=200)
#indy=np.arange(0,450,step=50)
##plt.title("Experimental Rupture Stress Compares to Predicticed with AdaBoosted Regression", fontsize=20)
#plt.xticks(indx, fontsize=32)
#plt.yticks(indy,fontsize=32)
# 
#plt.savefig('Rupture exp and predict.png')     
#plt.show()      



#RS_pred = regressor.predict(all_ind.values)



# model fitted for predicting the creep rupture time 
#all_train, all_test, target_train, target_test = cross_validation.train_test_split(all_ind, CRT, test_size=0.2)
#regr = DecisionTreeRegressor(max_depth=6)
#dt=regr.fit(all_train,target_train)
#show_tree(dt, feature_names,'dt_CRT.png')
#
#regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, random_state=0) #add random_state? 

#print("\nFe comp min", comp_data['Fe'].min())
#print("\nC comp min", comp_data['C'].min())
#print("\nNi comp min", comp_data['Ni'].min())
#print("\nV comp min", comp_data['V'].min())
#print("\nW comp min", comp_data['W'].min())
#print("\nSi comp min", comp_data['Si'].min())
#print("\nN comp min", comp_data['N'].min())
#print("\nMn comp min", comp_data['Mn'].min())
#print("\nFe comp max", comp_data['Fe'].max())
#print("\nC comp max", comp_data['C'].max())
#print("\nNi comp max", comp_data['Ni'].max())
#print("\nV comp max", comp_data['V'].max())
#print("\nW comp max", comp_data['W'].max())
#print("\nSi comp max", comp_data['Si'].max())
#print("\nN comp max", comp_data['N'].max())
#print("\nMn comp max", comp_data['Mn'].max())


#print("\nP comp min", comp_data['P'].min())
#print("\nS comp min", comp_data['S'].min())
#print("\nCr comp min", comp_data['Cr'].min())
#print("\nMo comp min", comp_data['Mo'].min())
#print("\nCu comp min", comp_data['Cu'].min())
#print("\nNb comp min", comp_data['Nb'].min())
#print("\nAl comp min", comp_data['Al'].min())
#print("\nB comp min", comp_data['B'].min())
#print("\nCo comp min", comp_data['Co'].max())
#print("\nTa comp min", comp_data['Ta'].max())
#print("\nNi comp min", comp_data['Ni'].max())
#print("\nO comp min", comp_data['O'].max())
#print("\nRe comp min", comp_data['Re'].max())
#
#print("\nP comp max", comp_data['P'].max())
#print("\nS comp max", comp_data['S'].max())
#print("\nCr comp max", comp_data['Cr'].max())
#print("\nMo comp max", comp_data['Mo'].max())
#print("\nCu comp max", comp_data['Cu'].max())
#print("\nNb comp max", comp_data['Nb'].max())
#print("\nAl comp max", comp_data['Al'].max())
#print("\nB comp max", comp_data['B'].max())
#print("\nCo comp max", comp_data['Co'].max())
#print("\nTa comp max", comp_data['Ta'].max())
#print("\nNi comp max", comp_data['Ni'].max())
#print("\nO comp max", comp_data['O'].max())
#print("\nRe comp max", comp_data['Re'].max())











