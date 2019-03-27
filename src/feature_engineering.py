from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# function to generate random features for setting a bench mark for other features to beat
def random_indicator(df_original):

    df = deepcopy(df_original)
    for i in range(0, config.no_rand):
        df['random_variable_' + str(i + 1)] = np.random.normal(size=df.shape[0])
        # shifting to positive region as some feature importance methods cannot handle negative values
        if df['random_variable_' + str(i + 1)].min()<0:
            df['random_variable_' + str(i + 1)] = df['random_variable_' + str(i + 1)]+ \
                                                  abs(df['random_variable_' + str(i + 1)].min())
    return df

# function to generate the chi squared scores to rank features
def chi_squared_score(df_original,label):

    df=deepcopy(df_original)
    df = random_indicator(df)
    Y=df[config.target_y_col]
    input_features=df.columns.values[np.in1d(df.columns.values,[Y.name],invert=True)].tolist()
    X=df[input_features]
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, Y)
    scores=fit.scores_
    score_df=pd.DataFrame(X.columns.values,columns=["feature_name"])
    score_df['scores']=scores
    score_df.sort_values('scores',ascending=False,inplace=True)
    score_df.reset_index(drop=True,inplace=True)
    score_df.to_csv('reports/chi_squared'+str(label)+'.csv',index=False)
    return score_df

# function to generate the rank of features using extaTrees - wrapper method
def feature_importance_extraTrees(df_original,label):

    df = deepcopy(df_original)
    Y = df[config.target_y_col]
    input_features = df.columns.values[np.in1d(df.columns.values, [Y.name], invert=True)].tolist()
    X = df[input_features]
    X = random_indicator(X)
    model = ExtraTreesClassifier()
    model.scoring='precision'
    model.fit(X, Y)
    scores=model.feature_importances_
    score_df = pd.DataFrame(X.columns.values, columns=["feature_name"])
    score_df['scores'] = scores
    score_df.sort_values('scores', ascending=False, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.to_csv('reports/feature_importance_extraTrees'+str(label)+'.csv', index=False)
    return score_df

# function to generate the MI scores to rank features
def mutual_info(df_original,label):

    df = deepcopy(df_original)
    Y = df[config.target_y_col]
    input_features = df.columns.values[
        np.in1d(df.columns.values, [Y.name], invert=True)].tolist()
    X = df[input_features]
    X = random_indicator(X)
    test = SelectKBest(score_func=mutual_info_classif, k=5)
    fit = test.fit(X, Y)
    scores = fit.scores_
    print(X.columns.values)
    print(scores)
    score_df = pd.DataFrame(X.columns.values, columns=["feature_name"])
    score_df['scores'] = scores
    score_df.sort_values('scores', ascending=False, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.to_csv('reports/mutual_info'+str(label)+'.csv', index=False)
    return score_df

# function to decompose the input features and check the variance observed
def pca_decomposition(df_original,n_components):

    non_calc_columns = [config.target_y_col]
    columns_to_decompose=df_original.columns.values[[np.in1d(df_original.columns.values,non_calc_columns,invert=True)]]
    df = deepcopy(df_original)
    X = df[columns_to_decompose]
    pca = PCA(n_components=n_components)
    fit=pca.fit(X)
    print("Explained Variance:", fit.explained_variance_ratio_)
    return

# recursive backward feature selection to rank the features
def rbfs(df_original,label=''):

    df=deepcopy(df_original)
    non_calc_columns = [config.target_y_col]
    features_to_select = df_original.columns.values[
        [np.in1d(df_original.columns.values, non_calc_columns, invert=True)]]
    X=df[features_to_select]
    Y=df[config.target_y_col]
    model=SVC(kernel='linear')
    rfe = RFE(model, 1)
    fit = rfe.fit(X, Y)
    print("Selected Features:", features_to_select[fit.support_])
    print("Feature Ranking: %s", fit.ranking_)
    score_df = pd.DataFrame(X.columns.values, columns=["feature_name"])
    score_df['ranks'] = fit.ranking_
    score_df.sort_values('ranks', ascending=True, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.to_csv('reports/rbfs_ranking_' + str(label) + '.csv', index=False)