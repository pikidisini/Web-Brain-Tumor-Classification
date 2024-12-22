import pickle
import numpy as np
import io
from flask import Flask, render_template, request, send_file

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# Load model from pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
app = Flask(__name__)

app.config['STATIC_FOLDER'] = 'static'

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html')

@app.route('/', methods=['GET'])
def scan():
    return render_template('/scan.html')

@app.route('/scan', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./image/" + imagefile.filename
    imagefile.save(image_path)
    
    # Process the image
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Rescale the image

    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ['Tumor Glioma', 'Tumor Meningioma', 'Tidak ada Tumor', 'Tumor Pituitary']
    result = class_labels[predicted_class[0]]
    
    # Prepare filename for display in template
    filename = imagefile.filename
    
    return render_template('/scan.html', prediction=result, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file('./image/' + filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)