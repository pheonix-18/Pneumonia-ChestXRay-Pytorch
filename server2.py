#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, request, jsonify
from PIL import  Image
from inference_cloud import ChestXRay

app = Flask(__name__)
model = ChestXRay()

print("\n ======***")

@app.route('/', methods=['POST'])
def predict():
    record = request.get_json()
    output = model.predict(record['file'])
    return jsonify(output)


app.run(debug=True, host='0.0.0.0')

