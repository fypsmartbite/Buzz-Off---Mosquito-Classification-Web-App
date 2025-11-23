# 🦟 Buzz Off - Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements_flask.txt
```

Or if you prefer using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_flask.txt
```

### Step 2: Run the App

**Option A - Direct Python:**
```bash
python app.py
```

**Option B - Using the startup script (Mac/Linux):**
```bash
./run_app.sh
```

### Step 3: Open Your Browser

Navigate to: **http://localhost:5000**

---

## Usage

1. **Upload Image**: Click the upload area or drag & drop a mosquito image
2. **Analyze**: Click "Analyze Mosquito" button
3. **View Results**: See if it's a dengue carrier with confidence scores

---

## Verify Setup (Optional)

Run the test script to check if everything is configured correctly:

```bash
python test_setup.py
```

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements_flask.txt
```

### Port 5000 already in use
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Model file not found
Ensure these files exist in the project root:
- `denseNet_161_64_25_0.001_adam.pt`
- `mean_and_std.pt`

---

## API Usage (Optional)

You can also use the API directly:

```bash
curl -X POST -F "file=@mosquito.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "prediction": "Dengue Mosquito (Aedes)",
  "is_dengue": true,
  "confidence": 0.95,
  "probabilities": {
    "Non-Dengue": 0.05,
    "Dengue": 0.95
  }
}
```

---

## What's Next?

- Upload different mosquito images to test the model
- Check the confidence scores and probability distributions
- Integrate the API into your own applications

For more details, see [README_FLASK.md](README_FLASK.md)
