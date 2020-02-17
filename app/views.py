from app import app
from flask import render_template, send_file, request
import pandas as pd
import os
from app.controller import optimization

file_name = ""


@app.route("/")
def index():
    return render_template("public/index.html", file_name=file_name)


@app.route("/download_output/")
def download_output():
    if file_name:
        try:
            return send_file("../"+file_name,
                             attachment_filename='Sortie.csv')
        except Exception as e:
            return str(e)
    else:
        return render_template("public/index.html", file_name=file_name)


@app.route("/convert", methods=["GET", "POST"])
def convert():
    global file_name
    if request.method == "POST":
        file_name = optimization()
    return render_template("public/index.html", file_name=file_name)
