# Oil Spill Detection using Computer Vision

Segmentation of oil spills in SAR (synthetic aperture radar) and infrared satellite imagery, served through a web application. Upload an image and the app returns a segmentation mask, an overlay, and spill metrics.

**Live demo:** https://oil-spill-frontend.vercel.app
**Context:** final-year capstone project, B.Tech Computer Science and Engineering, VIT Vellore (Dec 2024 – Apr 2025)

## Team and roles

A two-person project.

| | |
|---|---|
| **Sowmya Sriram** | Inference service and deployment: the Flask prediction API, model loading, input validation, and hosting (React front end on Vercel, API on Render). |
| **[Sibi Karthik C V](https://github.com/sibi15)** | Co-author |

## Results

- **U-Net on SAR: F1 0.92.** This is the model the deployed app serves.
- DeepLabv3 and CNN models were also trained, on infrared and SAR imagery.
- Models were evaluated with Dice coefficient, IoU, precision and recall.

Training notebooks are in [`training_files/`](training_files), and exploratory data analysis is in [`data_analysis/`](data_analysis).

## How it works

```
React + TypeScript (Vite, Tailwind)  ──POST /predict──▶  Flask API (gunicorn)
        Vercel                                             Render
                                                              │
                                              TFLite U-Net (backend/models/)
                                                              │
                              mask, overlay, Dice / IoU, spill area and density
```

- The SAR U-Net is served as a **TensorFlow Lite** model, which is lighter to host than the full TensorFlow model.
- The model is **loaded once and reused** across requests rather than rebuilt each time. When the API is run directly, it loads at start-up. This keeps responses under 3 seconds.
- Missing or unreadable uploads get a clear `400` error instead of failing inside the model.

## Run locally

Requirements: Node 18+ and Python 3.10.

```bash
# API
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # loads the model, then serves on http://localhost:8080

# Front end (in a second terminal, from the repo root)
cp .env.example .env     # set VITE_API_URL=http://localhost:8080
npm install
npx vite
```

In production the API runs under gunicorn (see `backend/Procfile` and `backend/render.json`).

## Repository layout

```
backend/          Flask API, TFLite model, Render config
src/              React front end
training_files/   model training notebooks (U-Net, DeepLabv3)
data_analysis/    exploratory data analysis notebooks
```

## Tech

React 18, TypeScript, Vite, Tailwind CSS · Flask, gunicorn · TensorFlow / TFLite, OpenCV, scikit-image · Vercel, Render
