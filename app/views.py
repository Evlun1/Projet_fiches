from app import app
from flask import render_template, send_file
import pandas as pd
import os


@app.route("/")
def index():
    result_df = pd.read_csv("Output/scenario_detaille_20200206_14_39.csv", sep=";")
    return render_template("public/index.html",
                           tables=[result_df.to_html(classes='data')],
                           titles=result_df.columns.values)


@app.route("/download_output/")
def download_output():
    try:
        return send_file('../Output/scenario_detaille_20200206_14_39.csv',
                         mimetype="application/x-csv",
                         attachment_filename='Sortie.csv')
    except Exception as e:
        return str(e)
