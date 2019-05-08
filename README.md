# Steel_Treatment
Codes developed for the final project of Spring 2019 24-786, Bayesian Machine Learning, in the Mechanical Engineering department at CMU. 

In the repository of steel_treatment by YuminZh, there are 4 python scripts. 

The script "Json_csv_yumin.py" read in json data file from citrine and parse, sort into three useful csv files: compositions, preparations, and properties. 

The second script "Project_Adaboost_regressor.py" perform the main jobs for this project, it output relative importance of features information, produce and plot definition trees, as well as plot predicted versus actual plot of testing dataset. 

The third script "Model_performance.py" does the same things as the "Project_adaboost_regressor", but using selected features as input. 

The forth one "correlation_map.py" does Pearson correlation analysis and produce colormap to explore relationship within features and relationships between features and property outputs. 
