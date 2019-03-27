# targetY columns to be predicted
target_y_col='acceptability'
# percentage of data kept for out of sample validation
out_of_sample_split=0.20
# No of components of PCA to be considered for decomposition
pca_components_output=6
# selected feature list that will be used to train ML models
selected_features=['buying_price','maintenance_price','number_of_doors','person_capacity','luggage_boot','safety']
# ML model whose hyper parameter is to be tunned
ML_model='RandomForestClassifier()'
# Hyper parameters to be tunned
model_hyper_parameters="dict(max_depth= [80, 90, 100, 110],max_features= [2, 3,4],min_samples_leaf= [3, 4, 5],n_estimators= [100, 200, 300, 1000])"
# Final ML model - this model is stored as a pickle for use of testing
ML_model_tunned='RandomForestClassifier(max_depth=80,max_features=4,min_samples_leaf=3,n_estimators=200)'
# no of random features to be generated into the dataset to check for feature relevance benchmark
no_rand=5
# dictionary used to map string features to numeric data (used in cleaning module)
feature_mapper={'buying_price':{'low':0,'med':1,'high':2,'vhigh':3},
                'maintenance_price':{'low':0,'med':1,'high':2,'vhigh':3},
                'number_of_doors':{'2':0,'3':1,'4':2,'5more':3},
                'person_capacity':{'2':0,'4':1,'more':2},
                'luggage_boot':{'small':0,'med':1,'big':2},
                'safety':{'low':0,'med':1,'high':2},
                'acceptability':{'unacc':0,'acc':1,'good':2,'vgood':3}}