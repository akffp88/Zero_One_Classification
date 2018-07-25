# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:49:02 2018

@author: LEE
"""

from flask import Flask
from flask import request
from flask import jsonify
from flask import json
import myProcessing

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/keyboard")
def keyboard():
        content = {
            'type' : 'buttons',
            'buttons' : ['Label 0', 'Label 1']
            }
        return jsonify(content)

@app.route("/message",methods=['GET', 'POST'])
def message():
        data = json.loads(request.data)
        content = data["content"]

        if content == "Label 0":
                result = myProcessing._get_response_(0)
        else :
                result = myProcessing._get_response_(1)

        print(result)
        
        response ={
                "message" :{
                        "text" : result
                }
        }

        response = json.dumps(response)

        return response

if __name__ == "__main__":
    myProcessing._setup_()
    app.run(host="0.0.0.0", port=5000)