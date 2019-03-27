from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn import metrics
from sklearn.metrics import roc_auc_score,precision_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from  sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
from pickle import dump
from pickle import load
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# function to compare different ML models after oversampling of data and to plot the score metrics
def compare_ml_algo_over_sampling(df,selected_features):
    X = df[selected_features]
    Y = df[config.target_y_col]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X=X,y=Y)
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('GBC', GradientBoostingClassifier()))
    # each model is evaluated in turn
    results = []
    names = []
    scorer = make_scorer(metrics.precision_score,
                         greater_is_better=True, average="weighted")
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=7)
        cv_results = cross_val_score(model, X_resampled, y_resampled, cv=kfold, scoring=scorer)
        results.append(cv_results)
        names.append(name)
        print(name, cv_results.mean(), cv_results.std())
    fig = pyplot.figure()
    fig.suptitle('ML model Comparison by over sampling')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()
    pyplot.savefig('graphs/ml_comparision_oversample.png')
    return

# function to compare different ML models and to plot the score metrics
def compare_ml_algo(df,selected_features):

    X=df[selected_features]
    Y=df[config.target_y_col]
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('GBC',GradientBoostingClassifier()))
    # each model is evaluated in turn
    results = []
    names = []
    scorer = make_scorer(metrics.precision_score,
                         greater_is_better=True, average="weighted")
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=7)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scorer)
        results.append(cv_results)
        names.append(name)
        print(name, cv_results.mean(), cv_results.std())
    fig = pyplot.figure()
    fig.suptitle('ML model Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()
    pyplot.savefig('graphs/ml_comparision.png')
    return

# function to perform hyper parameters tunning on a particular ML model - currently made for GradientBoostingClassifier
def grid_search(df,selected_features):

    print('Grid search for best hyper parameters has started')
    param_grid=eval(config.model_hyper_parameters)
    X = df[selected_features]
    Y = df[config.target_y_col]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X=X, y=Y)
    model=eval(config.ML_model)
    kfold = StratifiedKFold(n_splits=5, random_state=7)
    scorer = make_scorer(metrics.precision_score,
                         greater_is_better=True, average="weighted")
    grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring=scorer, cv=kfold)
    grid.fit(X_resampled, y_resampled)
    print("best score: ",grid.best_score_)
    print("The best model parameters : min_samples_leaf,max_depth, n_estimators,max_features")
    print(grid.best_estimator_.min_samples_leaf,grid.best_estimator_.max_depth,grid.best_estimator_.n_estimators,
          grid.best_estimator_.max_features)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return

# function to train the final ML model and to pickle it
def final_model_oversampled(training_df, selected_features, scoring):
    X = training_df[selected_features]
    Y = training_df[config.target_y_col]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X=X, y=Y)
    model = eval(config.ML_model_tunned)
    model.scoring=scoring
    model.fit(X_resampled, y_resampled)
    dump(model, open('saved_ML_model/finalized_model.sav', 'wb'))
    return


# function to return the confussion matrix and roc_auc of ML model on out_of_sample data
def validate_out_of_sample(out_sample, selected_features):

    X_test = out_sample[selected_features]
    Y_test = out_sample[config.target_y_col]
    loaded_model = load(open('saved_ML_model/finalized_model.sav', 'rb'))
    Y_predicted=loaded_model.predict(X_test)
    accuracy=precision_score(Y_test,Y_predicted,average='weighted')
    print("confusion matrix")
    print(confusion_matrix(Y_test,Y_predicted))
    print("precision",accuracy)
    return

