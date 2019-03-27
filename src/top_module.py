import pandas as pd
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config
from copy import deepcopy
from src.data_cleaning_and_preprocessing import clean_column_names,make_columns_numeric
from src.feature_engineering import chi_squared_score,feature_importance_extraTrees,mutual_info,pca_decomposition,rbfs
from src.data_stats import feature_correlation,feature_statistics,targetY_count
from src.data_visualization import histograms_plot,correlation_matrix_plot
from src.machine_learning import compare_ml_algo,grid_search,validate_out_of_sample, \
    compare_ml_algo_over_sampling,final_model_oversampled


# function to draw different plots for the features in the data
def visualize_data(df_train,label='unknown_data'):

    histograms_plot(df_train,label)
    correlation_matrix_plot(df_train,label)

# function to rank the features in accordance to it's relevance to targetY
def feature_engineering(df_train,label="unkown"):

    print('Ranking features according to chi_squared test')
    chi_squared_score(df_train,label)
    print("Chi Square scores csv generated")
    print('Ranking features according to importance given by ExtraTrees')
    feature_importance_extraTrees(df_train,label)
    print("feature importance given by ExtraTree stored in csv file")
    mutual_info(df_train,label)
    rbfs(df_train,label)

# function to generate initial stats for raw data
def generate_data_stats(df_original):

    df=deepcopy(df_original)
    print('Generating data stats only for numeric features')
    feature_statistics(df)
    feature_correlation(df)
    targetY_count(df)

# function to handle end to end data cleaning of training data
def clean_dataset(original_df,label='unknown_data'):

    df=deepcopy(original_df)
    df=clean_column_names(df)
    df=make_columns_numeric(df)
    df.to_csv('clean_data/cleaned_'+str(label)+'.csv',index=False)
    print("cleaned csv files are generated")
    return df

# function which creates an out of sample data set from the training set keeping the proportion of class frequency same
def create_validation_set(df_original):

    df=deepcopy(df_original)
    df.dropna(subset=[config.target_y_col],inplace=True)
    percent_split=config.out_of_sample_split
    training_data = pd.DataFrame()
    validation_data = pd.DataFrame()

    for target_class in df[config.target_y_col].unique():

        df_subset=df[df[config.target_y_col]==target_class]
        df_validation_subset=df_subset.sample(frac=percent_split, random_state=200)
        df_training_subset=df_subset.drop(df_validation_subset.index)
        validation_data=pd.concat((validation_data,df_validation_subset))
        training_data=pd.concat((training_data,df_training_subset))

    training_data.reset_index(drop=True, inplace=True)
    training_data.to_csv('raw_data/training_data.csv',index=False)
    validation_data.reset_index(drop=True, inplace=True)
    validation_data.to_csv('raw_data/validation_data.csv',index=False)
    print("validation and training dataset generated")

# start the program from this function
def main():

    # ================================= Creation of out of sample validation data =====================================

    answer=input("Do you want to create validation data from the raw data? (y/n)")
    if answer.lower()=="y":
        try:
            dataset = pd.read_csv('raw_data/cars.csv')
        except:
            print("raw files not found")
        create_validation_set(dataset)
        del dataset

    # ================================== Generate statistics of raw data ==============================================

    answer=input("Do you want to generate the raw data statistics for new training data? (y/n)")
    if answer.lower()=="y":
        try:
            training_data = pd.read_csv('raw_data/training_data.csv')
        except:
            print("Please create an out of sample test data and new training data")
        generate_data_stats(training_data)

    # ========================== Cleaning of only new training data==================================================

    answer=input("Do you want to clean the raw data - only new_training_data? (y/n)")
    if answer.lower()=="y":
        try:
            training_data=pd.read_csv('raw_data/training_data.csv')
        except:
            print("Please create an out of sample test data and new training data")
        clean_dataset(training_data,label='training_data')

    # ========================== Visualization of variables of training data ===========================================

    answer = input("Do you want to visualize the clean data?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        visualize_data(cleaned_training_data,label='_clean_data')

    # ========================== Feature Engineering on training data ================================================

    answer = input("Do you want to check importance of individual feature of clean data?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        feature_engineering(cleaned_training_data,label='_cleaned_data')

    # ============================ PCA decomposition to see how much variance is shown ================================

    answer = input("Do you want to do PCA decompose of clean_standardized_data to check varinace?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        pca_decomposition(cleaned_training_data,n_components=config.pca_components_output)
    # ============================================ ML model comparision ===============================================

    answer = input("Do you want to compare various ML models on clean_data without sampling?(y/n)")
    if answer.lower() == "y":
        try:
            training_df =pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        selected_features=config.selected_features
        compare_ml_algo(training_df,selected_features=selected_features)

    # ============================================ ML model by over sampling =============================================

    answer = input("Do you want to compare various ML models with over sampling?(y/n)")
    if answer.lower() == "y":
        try:
            training_df = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        selected_features = config.selected_features
        compare_ml_algo_over_sampling(training_df, selected_features=selected_features)


    # ============================ Run Grid search on selected ML model ===============================================

    answer = input("Do you want to run grid search for the selected ML model?(y/n)")
    if answer.lower() == "y":
        try:
            training_df = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        selected_features = config.selected_features
        grid_search(training_df,selected_features=selected_features)


    # ============================ Create the final ML model by oversampling ==========================================

    answer = input("Do you want to create the final ML model on oversampling?(y/n)")
    if answer.lower() == "y":
        try:
            training_df = pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Clean data file is not present")
        selected_features = config.selected_features
        final_model_oversampled(training_df, selected_features=selected_features, scoring='precision')

    # ============================ Validate Trained ML model on out of sample data =====================================

    answer = input("Do you want to check the trained ML model accuracy on out of sample data?(y/n)")
    if answer.lower() == "y":
        try:
            out_sample_data = pd.read_csv('raw_data/validation_data.csv')
        except:
            print("raw out of sample data file is not present")
        new_df=clean_dataset(out_sample_data,label='validation_data')
        selected_features = config.selected_features
        validate_out_of_sample(new_df, selected_features=selected_features)

    # ============================ ML model as a service to the user =====================================

    answer = input("Do you want to use the trained model to classify a car?(y/n)")
    if answer.lower() == "y":
        print("please start the flask application by running the classify_car.py script in src folder")



if __name__ == '__main__':
    main()
