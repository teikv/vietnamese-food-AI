from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load trained model ---
model = load_model('vietnamese_food_classification_model.h5')

# --- Define labels and ingredients ---
# ⚠️ Thứ tự phải đúng với train_data_gen.class_indices:
# {'banh_mi': 0, 'com_ga_xoi_mo': 1, 'pho': 2}
food_labels = ['Bánh mì', 'Cơm gà xối mỡ', 'Phở']

food_ingredients = {
    'Bánh mì': 'Bánh mì, thịt nguội, pate, dưa leo, rau thơm, tương ớt, nước tương',
    'Cơm gà xối mỡ': 'Cơm trắng, gà chiên giòn, hành phi, nước mắm tỏi ớt, dưa leo, cà chua',
    'Phở': 'Bánh phở, thịt bò/gà, hành lá, rau thơm, chanh, giá, nước dùng xương'
}

# --- Route: Home page ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route: Serve uploaded images ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Route: Upload + Predict ---
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    preds = model.predict(img_array)
    index = np.argmax(preds)
    predicted_label = food_labels[index]
    confidence = round(100 * np.max(preds), 2)
    ingredients = food_ingredients[predicted_label]

    # Render page with results
    return render_template(
        'index.html',
        prediction=predicted_label,
        confidence=confidence,
        ingredients=ingredients,
        image_path=f"/uploads/{file.filename}"
    )

if __name__ == '__main__':
    app.run(debug=True)
