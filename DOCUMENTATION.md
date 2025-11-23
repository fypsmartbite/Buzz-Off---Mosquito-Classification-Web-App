# Buzz Off Documentation

## Project Snapshot
- **Purpose**: Provide a quick, browser-based way to detect whether an uploaded mosquito image is an Aedes (dengue carrier) using a DenseNet161 model served through a Flask API.
- **Primary Users**: Public health researchers, mosquito-control teams, and citizen scientists needing a lightweight screening workflow.
- **Core Assets**: Flask backend (`app.py`), PyTorch inference logic (`src/predict.py`), training utilities (`src/data.py`, `src/transfer.py`, `src/helpers.py`), and a single-page HTML/CSS/JS client (`templates/index.html`).

## Non-Technical Overview
### Problem Statement & Value Proposition
Buzz Off helps field teams triage mosquito photos rapidly. Anyone can drag-drop an image, click "Analyze Mosquito," and receive a dengue/non-dengue decision plus confidence and probability breakdowns in seconds.

### User Journey
1. Start the Flask server (`python app.py`) and browse to `http://localhost:5000`.
2. Upload or drag-and-drop a mosquito image (PNG/JPG/JPEG/GIF/BMP/WEBP up to 16 MB).
3. View an instant classification card with dengue vs non-dengue probabilities and confidence. Use the reset button to analyze more images.

### Key Features for Stakeholders
- Real-time inference with visual probability bars for transparency.
- Health-check endpoint (`GET /health`) for monitoring and automation.
- Drag-and-drop UI, image previews, spinner, and friendly error states.

### Operational Considerations
- **Security defaults**: File-type allow list, 16 MB upload ceiling, sanitized filenames, and guarded error messaging.
- **Runtime footprint**: Requires `denseNet_161_64_25_0.001_adam.pt` (weights) and `mean_and_std.pt` (normalization stats) in project root.
- **Deployment**: Development uses Flask’s built-in server; production should wrap with Gunicorn or similar (`gunicorn -w 4 -b 0.0.0.0:5000 app:app`).

## High-Level Technical Overview
### Tech Stack & Dependencies
- **Backend**: Flask application (Werkzeug, Jinja) exposing `/`, `/predict`, `/health` routes.
- **Model Runtime**: PyTorch + Torchvision DenseNet161 with a custom classifier head and dataset-specific normalization tensors.
- **Frontend**: Vanilla HTML/CSS/JavaScript single page served from `templates/index.html`; uses Fetch API to POST multipart form data to `/predict`.
- **Dependencies**: Minimal Flask stack in `requirements_flask.txt`; extended ML/training stack in `requirements.txt`.

### System Architecture & Data Flow
1. **Client** gathers a file via click or drag events, renders a preview, and sends it to `/predict` with `fetch` + `FormData`.
2. **Flask Controller** validates the payload, enforces size/types, stores it under `static/uploads`, and calls the classifier service object.
3. **Inference Service** loads DenseNet161 with custom FC head, applies deterministic transforms (Resize → CenterCrop → Normalize), and produces class probabilities.
4. **Response** adds prediction metadata plus a static URL to the saved image for frontend rendering.
5. **Health Monitoring** hits `/health` to confirm server responsiveness and model availability.

### API Surface
| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Renders the SPA upload interface. |
| `/predict` | POST | Accepts multipart `file`, saves it, runs inference, returns JSON (`prediction`, `is_dengue`, `confidence`, `probabilities`, `image_url`). |
| `/health` | GET | Lightweight health check reporting service status. |

### Model Lifecycle
- Trained via transfer learning: DenseNet161 backbone with Linear → ReLU → Dropout stacks for the classifier head.
- Normalization statistics cached in `mean_and_std.pt` after a one-time pass through the dataset.
- Training utilities (data loaders, helpers) provide augmentation, deterministic splits, and reproducible pipelines for retraining.

## Low-Level Technical Details
### Directory Layout Reference
| Path | Description |
| --- | --- |
| `app.py` | Flask entry point, configuration, routes, upload handling, classifier wiring. |
| `src/predict.py` | `MosquitoClassifier` class for model loading, preprocessing, inference, and JSON-friendly outputs. |
| `src/transfer.py` | Transfer-learning helper that freezes torchvision backbones and replaces the final layer stack. |
| `src/data.py` | Torchvision dataset and DataLoader setup with augmentations for train/valid/test splits. |
| `src/helpers.py` | Environment prep, dataset download/extraction, reproducibility seeds, mean/std caching. |
| `templates/index.html` | Complete UI: layout, styling, client-side JS logic for uploads, previews, inference, and visualization. |
| `static/uploads/` | Runtime storage for user-uploaded images. |

### Backend Service Details (`app.py`)
- Configures upload folder, allowed extensions (`png`, `jpg`, `jpeg`, `gif`, `bmp`, `webp`), model paths, and a 16 MB limit.
- `allowed_file(filename)` enforces file-type checks before saving.
- `/predict` flow: validate → `secure_filename` → save → `classifier.predict()` → attach `image_url` → JSON response (400/500 on failure).
- `/health` returns a JSON status payload; can be extended with GPU/memory diagnostics if needed.

### Inference Pipeline (`src/predict.py`)
- Auto-selects CUDA when available, else CPU.
- Applies test-time transforms: Resize(256) → CenterCrop(224) → ToTensor → Normalize(mean, std).
- Rebuilds DenseNet161 head (Linear layers with ReLU + Dropout) before loading weights and switching to eval mode.
- `predict(image_path)` opens the image via PIL, processes it, runs a forward pass without gradients, and returns probabilities plus boolean `is_dengue`.

### Training Utilities
- `get_model_transfer_learning` freezes pretrained models and injects the custom classifier head for fine-tuning.
- `get_data_loaders` applies augmentations (random crop, flip, rotation, color jitter, affine, blur), performs random train/valid splits, and constructs DataLoaders.
- `helpers.py` ensures reproducible seeding, dataset download from Firebase storage, compute-and-cache of normalization stats, and checkpoint directory creation.

### Frontend Behavior (`templates/index.html`)
- Pure HTML/CSS/JS with modern card layout, gradient background, dashed upload drop zone, and animated spinner.
- JavaScript handles drag/drop, FileReader previews, fetch-based prediction calls, error handling, and probability bar updates for each class.
- Reset button clears state for fast repeated analyses; `accept="image/*"` restricts client-side file selection.

### Configuration & Assets
- **Model weights**: `denseNet_161_64_25_0.001_adam.pt` (~118 MB) must exist before inference. Keep it out of source control.
- **Normalization stats**: `mean_and_std.pt` reused across training and inference; regenerate via `helpers.compute_mean_and_std()` if dataset changes.
- **Environment**: Recommended to run inside a virtualenv (`python3 -m venv venv && source venv/bin/activate && pip install -r requirements_flask.txt`).

## Setup & Troubleshooting Reference
- **Start locally**: `python app.py` (or `./run_app.sh`) → open `http://localhost:5000`.
- **API smoke test**: `curl -X POST -F "file=@mosquito.jpg" http://localhost:5000/predict`.
- **Common fixes**:
  1. Missing modules → `pip install -r requirements_flask.txt`.
  2. Port 5000 busy → change `app.run(..., port=5001)`.
  3. Missing assets → verify weights and mean/std files in project root.
- **Production hardening ideas**: add authentication, structured logging, rate limiting, and deeper health checks before public exposure.
