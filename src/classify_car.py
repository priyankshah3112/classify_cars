from flask import Flask, render_template,request
from wtforms import Form,SelectField
import pickle
import pandas as pd
import os
from config import config
from src.data_cleaning_and_preprocessing import make_columns_numeric

def classify(df):

    df=make_columns_numeric(df)
    clf=pickle.load(open('saved_ML_model/finalized_model.sav', 'rb'))
    X=df[config.selected_features]
    y = clf.predict(X)[0]
    feature_mapper=config.feature_mapper
    label=list(feature_mapper[config.target_y_col].keys())[list(feature_mapper[config.target_y_col].values()).index(y)]
    return label

app = Flask(__name__,template_folder=os.getcwd()+"\\template")

class CarForm(Form):

    buying_price=SelectField('buying_price')
    maintenance_price=SelectField('maintenance_price')
    number_of_doors=SelectField('number_of_doors')
    person_capacity=SelectField('person_capacity')
    luggage_boot=SelectField('luggage_boot')
    safety=SelectField('safety')


@app.route('/')
def index():

    form = CarForm(request.form)
    return render_template("car_features.html", form=form)

@app.route('/results', methods=['POST'])
def results():

    form = CarForm(request.form)
    if request.method == 'POST':
        buying_price = request.form['buying_price']
        maintenance_price = request.form['maintenance_price']
        number_of_doors = request.form['number_of_doors']
        person_capacity = request.form['person_capacity']
        luggage_boot = request.form['luggage_boot']
        safety = request.form['safety']
        data_df=pd.DataFrame(data=[[buying_price,maintenance_price,number_of_doors,person_capacity,luggage_boot,safety]],
                             columns=["buying_price","maintenance_price","number_of_doors","person_capacity",
                                      "luggage_boot","safety"])
        y = classify(data_df)
        return render_template('results.html',prediction=y,)
    return render_template('car_features.html', form=form)



if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)