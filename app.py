from flask import Flask, render_template, request, jsonify, make_response
from flask_restful import Api, Resource
import numpy as np
import requests

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

app = Flask(__name__)
api = Api(app)


pretrained_model = InceptionV3(weights="imagenet")

class Main(Resource):
    def get(self):
        return make_response(render_template('index.html'))

class Recognise(Resource):
    def post(self):
        try:
            postedData = request.get_json()
            image_url = postedData.get('image_url')
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((299, 299))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            preds = pretrained_model.predict(img_array)
            decoded = imagenet_utils.decode_predictions(preds, top=1)
            ret_json = {}
            for result in decoded[0]:
                ret_json[result[1]] = round(float(result[2] * 100),2)

            # desc_sorted = dict(sorted(ret_json.items(), key=lambda item: item[1], reverse=True))
            return jsonify({"status": True, "image_url":image_url, "message":'', "result": ret_json})
        except Exception as ex:
            ret_json = {
                'status' : False,
                "result" : {},
                "message": "The image could not be recognised or Please enter a valid image URL"
            }
            return jsonify(ret_json)


api.add_resource(Recognise, '/recognise')
api.add_resource(Main, '/')

if __name__=="__main__":
    app.run(debug=True)