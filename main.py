"""`main` is the top level module for your Flask application."""

# Import the Flask Framework
from flask import Flask
from flask import request
from flask import render_template
from flask import send_file
from model import style_transfer
import requests
import time

import os
app = Flask(__name__)
# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.


@app.route('/')
def my_form():
    """Return a friendly HTTP greeting."""
    return render_template("my-form.html")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    # imageName = text.split("/")[-1]
    # contentImagePath = "images/input/"+imageName
    # outputImagePath = "images/output/"+imageName
    # # print(file_path)
    # if not os.path.exists(outputImagePath):
    #     f = open(contentImagePath, 'wb')
    #     f.write(requests.get(text).content)
    #     f.close()
        #style_transfer("images/profile.jpg")
    style_transfer()

    # return send_file(outputImagePath,mimetype='image/jpg')


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404


@app.errorhandler(500)
def application_error(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: {}'.format(e), 500
