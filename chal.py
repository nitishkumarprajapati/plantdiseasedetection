from flask import Flask, request, render_template
import numpy as np
'''import tensorflow as tf'''
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
app = Flask(__name__)

def model_prediction(file):
    model = tf.keras.models.load_model("fff.h5")
    # Create the directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # Save the file temporarily
    file_path = os.path.join(temp_dir, "temp_img.jpg")
    file.save(file_path)
    # Load the image
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    os.remove(file_path)  # Remove the temporary file
    return np.argmax(predictions)  # return index of max element

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/contact.html')
def contact():
    return render_template("contact.html")
@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/prediction.html', methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        img = request.files["img"]
        prediction_index = model_prediction(img)
        classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']  
        predicted_class = classes[prediction_index]
    
        return render_template("prediction.html", data=predicted_class)
    else:
        return render_template("prediction.html")
if __name__ == "__main__":
    app.run(debug=True)
