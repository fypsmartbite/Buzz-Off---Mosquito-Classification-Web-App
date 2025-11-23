# Buzz Off - Mosquito Classification Web App

## 1️⃣ Project Name / Problem Statement
Buzz Off targets the slow, manual identification of dengue-carrying Aedes mosquitoes. Field teams often rely on visual inspection, delaying outbreak response.

## 2️⃣ Problem Solution
The app lets anyone upload a mosquito image and instantly see if it's dengue-positive. DenseNet161 powers the predictions, while Flask/Gradio provide an accessible, drag-and-drop interface with confidence scores and probability bars.

## 3️⃣ Technology Stack Used
1. **Model Runtime**: Python, PyTorch, Torchvision (DenseNet161)
2. **Web Layer**: Flask for local/UI API, Gradio on Hugging Face Spaces for the hosted demo
3. **Infrastructure**: Hugging Face Spaces for deployment, Google Drive for hosting large `.pt` weights, GitHub for source control

## 4️⃣ High-Level Diagram
*(Paste architecture diagram here)*

## 5️⃣ Live Demo
- **Hosted Space**: https://huggingface.co/spaces/huzaifaiftikhar/buzz_off
- **Local quick start**
  ```bash
  git clone https://github.com/fypsmartbite/Buzz-Off---Mosquito-Classification-Web-App.git
  cd Buzz-Off---Mosquito-Classification-Web-App
  pip install -r requirements.txt
  python scripts/download_weights.py && python app.py
  ```
- Open `http://localhost:5000`, upload an image, click **Analyze Mosquito**, and review the result card.

## 6️⃣ Team Members
- Sonail Saqib – sonailsaqib2000@gmail.com
- Laiba Khan – laiba.khan0278@gmail.com
- Anoosha Khan – anooshkhan799@gmail.com
- Huzaifa Iftikhar – chhuzaifaiftikhar@gmail.com

---

## Appendix: Architecture & Developer Notes

### Project Structure
```
Prototype/
├── app.py                          # Flask application
├── scripts/download_weights.py     # Pulls weights from Google Drive
├── src/
│   ├── predict.py                  # Inference module
│   ├── transfer.py                 # Transfer-learning helper
│   ├── data.py                     # Data loading utilities
│   └── helpers.py                  # Helper functions
├── templates/index.html            # Web interface
├── static/uploads/                 # Uploaded images
├── mean_and_std.pt                 # Normalization tensors
├── denseNet_161_64_25_0.001_adam.pt# Model weights (downloaded at runtime)
├── requirements.txt                # Dependencies
└── README_FLASK.md                 # This document
```

### API Endpoints
- `GET /` → Renders the upload UI
- `POST /predict` → Multipart file upload, returns JSON (`prediction`, `confidence`, `probabilities`, `image_url`)
- `GET /health` → `{ "status": "healthy", "model_loaded": true }`

### Model & Config Highlights
- DenseNet161 backbone with custom classifier head (Linear → ReLU → Dropout stack)
- Preprocessing: Resize 256 → CenterCrop 224 → Normalize with dataset mean/std
- Upload validation: extension allow list, 16 MB limit, filenames sanitized via `secure_filename`

### Local Development Cheatsheet
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weights.py
python app.py  # http://localhost:5000
```

### Troubleshooting
- **Model not found**: ensure `denseNet_161_64_25_0.001_adam.pt` downloads (rerun `scripts/download_weights.py`).
- **Missing deps**: reinstall via `pip install -r requirements.txt` and restart the server.
- **Port busy**: edit `app.run(..., port=5001)` in `app.py`.

