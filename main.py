import base64
import cv2
import io
import numpy as np
import os
import random

from flask import Flask, render_template, request, jsonify
from google.cloud import storage, vision
from PIL import Image
from typing import Sequence
from uuid import uuid4

BUCKET_NAME = os.getenv('BUCKET_NAME')
USER = 'alice'
SAVED_WORDS = []


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def root():
    return render_template('index.html')

@app.route("/image", methods=['GET', 'POST'])
def show_image():
    if request.method == 'POST':
        # Retrieve uploaded file
        f = request.files['image-file']
        image_bytes = f.read()

        # Connect to Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Upload image to Cloud Storage
        filename = str(uuid4())
        blob = bucket.blob(os.path.join(USER, filename))
        blob.upload_from_string(image_bytes, content_type='image/jpeg')

        # Vision processing
        response = analyze_image_from_local(image_bytes=image_bytes, feature_types=[vision.Feature.Type.TEXT_DETECTION])
        result_bytes = draw_bounding_box(image_bytes=image_bytes, objects=response.text_annotations)
        
        words = {}
        annotations = response.text_annotations[1:]
        for v in annotations:
            words[v.description] = {'translation': None}

        # Save image as in-memory file
        img = Image.open(io.BytesIO(result_bytes))
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG')
        encoded_img_data = base64.b64encode(buffer.getvalue())

    return render_template('image.html', 
                           image=encoded_img_data.decode('utf-8'), 
                           words=words)

@app.route("/study", methods=['GET', 'POST'])
def study():
    card_order = generate_random(len(SAVED_WORDS))
    return render_template('study.html', title='Study', words=SAVED_WORDS, card_order=card_order)

@app.route("/translate", methods=['POST'])
def translate():
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    word = request.json.get('word')
    if isinstance(word, bytes):
        word = word.decode("utf-8")

    result = translate_client.translate(word) # Defaults to English

    return jsonify({'result': result["translatedText"]})

@app.route("/add", methods=['POST'])
def add_word():
    word = request.json.get('word')
    translation = request.json.get('translation')
    SAVED_WORDS.append({'word': word, 'translation': translation})

    return jsonify({'result': 'OK'})

@app.route("/library", methods=['GET'])
def show_library():
    images = []

    # Connect to Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Download images from Cloud Storage
    blobs = bucket.list_blobs(prefix=f'{USER}/')
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        image_bytes = blob.download_as_bytes()

        # Save image as in-memory file
        img = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG')
        encoded_img_data = base64.b64encode(buffer.getvalue())
        images.append(encoded_img_data.decode('utf-8'))

    return render_template('library.html', images=images)

def generate_random(n):
    numbers = list(range(0, n))
    random.shuffle(numbers)
    return numbers

def analyze_image_from_local(
    image_bytes: bytes,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=image_bytes)
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response

def draw_bounding_box(image_bytes, objects):
    # Draw bounding box on image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for o in objects:
        vertices = o.bounding_poly.vertices
        cv2.rectangle(image, (vertices[0].x, vertices[0].y), (vertices[2].x, vertices[2].y), (0, 255, 0), 2) 

    _, encoded_image = cv2.imencode('.jpg', image)
    return encoded_image.tobytes()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
