import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from src.predict import MosquitoClassifier

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MODEL_PATH = 'denseNet_161_64_25_0.001_adam.pt'
MEAN_STD_PATH = 'mean_and_std.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize classifier
classifier = MosquitoClassifier(MODEL_PATH, MEAN_STD_PATH)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = classifier.predict(filepath)
        
        # Add image URL to result
        result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})


if __name__ == '__main__':
    print("Starting Mosquito Classification Server...")
    print(f"Model: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)
