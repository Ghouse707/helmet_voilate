# Helmet Violation Detection System

**Author:** [Ghouse Pasha](https://github.com/Ghouse707)

A Python and Streamlit application that detects riders without helmets in video, reads number plates with OCR, saves evidence images, and stores each violation in SQLite.

This project uses public pretrained models. It does not train any models.

<img width="2838" height="1534" alt="Video Detection tab" src="https://github.com/user-attachments/assets/9f197077-7593-4245-971e-0beb628bdaa4" />

*Video Detection tab with the upload area and model settings.*

---

## Features

- Detects riders without helmets using YOLO and ByteTrack tracking.
- Confirms a violation before saving it, preventing repeated records for the same tracked vehicle.
- Detects Indian number plates and reads plate text with EasyOCR.
- Saves one best plate crop for each violation.
- Saves up to six timestamped rider and vehicle context images.
- Stores violation data in SQLite.
- Shows evidence images, plate crops, and vehicle image galleries in the Streamlit dashboard.

---

## Screenshots

<img width="2880" height="1530" alt="Violations Dashboard" src="https://github.com/user-attachments/assets/149b7b62-27df-4744-8e78-aba3519eca21" />

*Violations Dashboard with counts and saved violation records.*

<img width="2372" height="1518" alt="Evidence and plate crop" src="https://github.com/user-attachments/assets/4a753ca5-8061-4873-82e7-73b065dc834a" />


---

## Requirements

- Python 3.10 or 3.11
- Windows, Linux, or macOS
- Optional: NVIDIA GPU with CUDA for faster detection

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/Ghouse707/helmet_voilate.git
cd helmet_voilate
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA support, install the PyTorch version matching your CUDA driver from the official PyTorch website before installing the remaining dependencies.

---

## Model Files

Model weight files are intentionally excluded from this GitHub repository.

Download the compatible public pretrained model files and place them in:

```text
models/pretrained/
```

Required filenames:

```text
models/pretrained/helmet_yolov8n.pt
models/pretrained/indian_plate_yolo11n.pt
```

| File | Public source | Purpose |
|---|---|---|
| `helmet_yolov8n.pt` | `iam-tsr/yolov8n-helmet-detection` | Detects riders with and without helmets |
| `indian_plate_yolo11n.pt` | `vivekvar/helmet-v5` | Detects Indian vehicle number plates |

The helmet model uses the classes `With Helmet` and `Without Helmet`.

Publisher-reported helmet model metrics:

| Metric | Score |
|---|---|
| Precision | 0.855 |
| Recall | 0.808 |
| mAP@0.5 | 0.881 |

These metrics do not guarantee the same performance for every camera, road, lighting condition, or video.

---

## Run the Application

```powershell
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

1. Upload an MP4, AVI, MOV, or MKV video.
2. Set helmet and plate confidence thresholds if needed.
3. Click **Start detection**.
4. View live annotated detection output.
5. Open the **Violations Dashboard** to inspect saved violations.

---

## How It Works

### 1. Helmet detection and tracking

- YOLO detects helmet and no-helmet objects.
- ByteTrack associates the same rider across video frames.
- A no-helmet detection must appear at least four times within a 15-frame window.
- After confirmation, the tracked vehicle is marked so it is not counted repeatedly.

### 2. Plate detection and OCR

- The plate model searches the rider and vehicle area every five frames.
- Plate collection runs for up to 60 frames after confirmation.
- The system chooses the best crop using detector confidence and image sharpness.
- EasyOCR reads text from the selected crop once.
- Plate text is normalized to uppercase letters and numbers.
- OCR results shorter than six characters are stored as `UNKNOWN`.

### 3. Evidence capture

For each confirmed violation, the system saves:

| Evidence type | Folder | Description |
|---|---|---|
| Full evidence frame | `evidence/` | Annotated full-frame violation image |
| Vehicle context crops | `evidence/` | Up to six timestamped crops showing the rider and vehicle |
| Number plate crop | `plates/` | Best plate image selected for OCR |

### 4. Database storage

SQLite creates `violations.db` automatically on first run.

Each violation stores:

| Field | Description |
|---|---|
| `timestamp` | Time when the violation record was saved |
| `track_id` | ByteTrack ID for the detected rider |
| `plate_number` | OCR result or `UNKNOWN` |
| `helmet_status` | `NO_HELMET` |
| `plate_confidence` | Plate detector confidence |
| `helmet_confidence` | Helmet detector confidence |
| `video_source` | Name of the uploaded video |
| Evidence path | Saved full-frame evidence image |
| Plate path | Saved plate crop, when available |

Timestamped vehicle context images are stored in a related SQLite table.

### Duplicate protection

- One violation is stored for each confirmed tracker event in a source video.
- A recognized plate is stored once in the database.
- Reprocessing the same video can refresh context images without creating duplicate events.

---

## Project Structure

```text
helmet_voilate/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── database/               # SQLite database logic
├── detection/              # Helmet, plate, and tracker logic
├── ocr/                    # EasyOCR plate reading
├── utils/                  # Image helpers
├── models/
│   └── pretrained/         # Downloaded model weights
├── evidence/               # Generated evidence images
├── plates/                 # Generated plate crops
└── violations.db           # Created automatically at runtime
```

---

## Notes

- Generated videos, evidence images, plate crops, the SQLite database, local credentials, and model weights are excluded from GitHub.
- Detection and OCR quality depend on camera angle, lighting, video quality, plate size, and motion blur.
- Review evidence images before relying on a recognized plate number.
- Use **Clear saved violations** in the dashboard to remove all stored records and generated images.

---

## Author

**Ghouse Pasha**
GitHub: [@Ghouse707](https://github.com/Ghouse707)
