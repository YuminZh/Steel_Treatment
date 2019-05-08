# =============================================================================
# Author: Yumin Zhang 
# Date: Mar 31 2019
# Project: bayesian machine learning steel treatment
# =============================================================================
import json
import csv
# =============================================================================
# Open json file for reading and csv file for writing
# =============================================================================
with open ('creep_output.json') as json_file:
    data_all = json.load(json_file)

data_len = len(data_all); # length of json data file 

data = data_all[0]
ele_name = []
treat_name = []
# =============================================================================
# write title row of each csv files 
# =============================================================================
for item in data['composition']:
    name = item['element']
    ele_name.append(name)
with open('composition.csv','w', newline='') as file_comp: 
    thewriter = csv.writer(file_comp)  
    thewriter.writerow(ele_name)       
# write to preparation file 
for item in data['preparation']:
    for subitem in item['details']:
        name = subitem['name']
        treat_name.append(name)
with open('preparation.csv','w',newline='') as file_prep:
    thewriter = csv.writer(file_prep)
    thewriter.writerow(treat_name)

prop_name = ['Creep rupture time','Creep temp','Rupture stress','Rupture temp']    
with open('properties.csv','w',newline='') as file_proper:
    thewriter = csv.writer(file_proper)
    thewriter.writerow(prop_name)  
# =============================================================================
# write value rows for each csv files    
# =============================================================================
i=0
for i in range(data_len): 
    data = data_all[i]
    ele_weight = []
    treat_temp = []
    prop = []
 
# compositions     
    for item in data['composition']:
        weight = item['idealWeightPercent']
        ele_weight.append(weight)
    j=0
    for j in range(19):
        ele_weight[j] = float(ele_weight[j])
     
    ele_weight = ele_weight[:-1]
    Fe_weight = 100-sum(ele_weight)
    ele_weight.append(Fe_weight)
     
    with open('composition.csv','a', newline='') as file_comp:
        thewriter = csv.writer(file_comp)  
        thewriter.writerow(ele_weight)

# preparations 
    for item in data['preparation']:
        for subitem in item['details']:
            temp = subitem['scalars']
            treat_temp.append(temp)
    with open('preparation.csv','a',newline='') as file_prep:
        thewriter = csv.writer(file_prep)
        thewriter.writerow(treat_temp)

# properties 
    for item in data['properties']:
        properties = item['scalars'] 
        cond = item['conditions'][0]['scalars']
        prop.append(properties)
        prop.append(cond) 
    with open('properties.csv','a',newline='') as file_proper:
        thewriter = csv.writer(file_proper)
        thewriter.writerow(prop)   
        
 


 



   