import pickle
import numpy as np
import io

# Data Processing
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, render_template, request

# Load model from pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define prediction function
def predict(uploaded_file):
    # Convert image_data (FileStorage) to BytesIO
    image_data = io.BytesIO(uploaded_file.read())
    
    # getting image to test output
    im = load_img(image_data, target_size=(150, 150))
    im = img_to_array(im)

    # Reshape it to (1, 150, 150, 3)
    im = np.expand_dims(im, axis=0)
    print(f'x reshaped: {im.shape}')

    # normalization tensor
    im /= np.max(im) # ~ np.max(img_tensor)

    # Make prediction using the normalized copy
    prediction = model.predict(im)
    return prediction

# Create a Flask app
app = Flask(__name__)

app.config['STATIC_FOLDER'] = 'static'

# Route for main page
@app.route('/')
def index():
    return render_template('/index.html')

# Route for scan page
@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'GET':
        return render_template('/scan.html')
    elif request.method == 'POST':
        # Upload image and get image data
        uploaded_file = request.files['image']
        image_path = "./image/" + uploaded_file.filename
        uploaded_file.save(image_path)
        if uploaded_file and uploaded_file.filename.endswith('.jpg'):
            
            # Make prediction using the prediction function
            prediction = predict(uploaded_file)
            # Render result
            return render_template('/scan.html', image_data=uploaded_file, prediction=prediction)
        else:
            return 'Invalid file format. Only JPG allowed.'

if __name__ == '__main__':
    app.run(debug=True)