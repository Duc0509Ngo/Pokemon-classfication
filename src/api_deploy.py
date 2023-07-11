from flask import Flask, jsonify, request
from test import  test


app = Flask(__name__)

def get_prediction(image_bytes):
    prediction = test(image_path=image_bytes, deploy=True)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction = get_prediction(image_bytes=img_bytes)
        return jsonify({'Pokemon': prediction})

if __name__ == '__main__':
    app.run()