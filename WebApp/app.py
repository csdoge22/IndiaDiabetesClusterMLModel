from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import FloatField, IntegerField, RadioField
from dotenv import load_dotenv
import os

import pandas as pd

app = Flask(__name__)

load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
BASE_DIR = os.getenv('BASE_DIR')
# DEBUG
def retrieve_base_csv():
    df = pd.read_csv(os.path.join(BASE_DIR,"diabetes_prediction_india.csv"))
    return df

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/user', methods=['POST'])
def process_input():
    pass


if __name__=="__main__":
    app.run(debug=True)