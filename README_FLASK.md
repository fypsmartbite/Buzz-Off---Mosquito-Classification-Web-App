# Buzz Off - Mosquito Classification Web App

A Flask-based web application that classifies mosquito images to detect if they are dengue carriers (Aedes mosquitoes) using a trained DenseNet161 model.

## Features

- 🦟 Upload mosquito images for classification
- 🎯 Real-time prediction with confidence scores
- 📊 Visual probability distribution
- 🎨 Modern, responsive UI
- 🚀 Easy to deploy

## Project Structure

```
Prototype/
├── app.py                          # Flask application
├── src/
│   ├── predict.py                  # Inference module
│   ├── transfer.py                 # Model architecture
│   ├── data.py                     # Data loading utilities
│   └── helpers.py                  # Helper functions
├── templates/
│   └── index.html                  # Web interface
├── static/
│   └── uploads/                    # Uploaded images storage
├── denseNet_161_64_25_0.001_adam.pt  # Trained model weights
├── mean_and_std.pt                 # Normalization parameters
└── requirements_flask.txt          # Python dependencies
```

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /Users/muhammadharis/Downloads/Prototype
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements_flask.txt
```

## Usage

### Start the Flask server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Access the web interface

1. Open your browser and go to `http://localhost:5000`
2. Upload a mosquito image (PNG, JPG, JPEG, GIF, BMP, WEBP)
3. Click "Analyze Mosquito"
4. View the prediction results with confidence scores

## API Endpoints

### `GET /`
- Returns the main web interface

### `POST /predict`
- Upload an image for classification
- **Request**: Multipart form data with `file` field
- **Response**: JSON with prediction results

```json
{
  "prediction": "Dengue Mosquito (Aedes)",
  "is_dengue": true,
  "confidence": 0.95,
  "probabilities": {
    "Non-Dengue": 0.05,
    "Dengue": 0.95
  },
  "image_url": "/static/uploads/mosquito.jpg"
}
```

### `GET /health`
- Health check endpoint
- **Response**: `{"status": "healthy", "model_loaded": true}`

## Model Details

- **Architecture**: DenseNet161 (Transfer Learning)
- **Classes**: 
  - Non-Dengue Mosquito (Class 0)
  - Dengue Mosquito - Aedes (Class 1)
- **Input Size**: 224x224 pixels
- **Preprocessing**: 
  - Resize to 256x256
  - Center crop to 224x224
  - Normalize with dataset mean and std

## Configuration

Edit `app.py` to modify:

- `UPLOAD_FOLDER`: Directory for uploaded images (default: `static/uploads`)
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 16MB)
- `MODEL_PATH`: Path to model weights
- `MEAN_STD_PATH`: Path to normalization parameters

## Troubleshooting

### Model Loading Error
- Ensure `denseNet_161_64_25_0.001_adam.pt` exists in the project root
- Ensure `mean_and_std.pt` exists in the project root

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements_flask.txt`
- Activate your virtual environment

### Port Already in Use
- Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

## Development

To run in development mode with auto-reload:

```bash
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
python app.py
```

## Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Security Notes

- File uploads are restricted to image formats only
- File size is limited to 16MB
- Filenames are sanitized using `secure_filename()`
- Consider adding authentication for production use

## License

This project is for educational and research purposes.

## Credits

- Model: DenseNet161 with custom classifier
- Framework: Flask, PyTorch, TorchVision
- Dataset: Mosquito classification dataset
