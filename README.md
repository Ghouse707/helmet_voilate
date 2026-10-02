# Helmet Violation Detection System

Python + Streamlit application for helmet violation detection, plate OCR, evidence capture, and SQLite storage. It uses public pretrained weights and does not train models.

## Setup

Use Python 3.10 or 3.11. From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your CUDA driver using the official PyTorch selector before installing the remaining requirements. The application selects CUDA when `torch.cuda.is_available()` and otherwise uses CPU.

## Model files

The default weights are in `models/pretrained/`:

- `helmet_yolov8n.pt`: `iam-tsr/yolov8n-helmet-detection`, with `With Helmet` and `Without Helmet` classes. Its model card reports precision 0.855, recall 0.808, and mAP@0.5 0.881 on its evaluation data.
- `indian_plate_yolo11n.pt`: `vivekvar/helmet-v5`'s fine-tuned Indian plate detector (`plate_yolo_ft.pt`). This is selected for its Indian CCTV/plate focus; the publisher does not provide a directly comparable plate-detector validation score in the model card.

The numbers above are publisher-reported and do not guarantee performance on a different camera, region, or video. Confidence thresholds and model paths can be changed in the sidebar. The original `helmet.pt` and `license_plate_detector.pt` files may remain in the project root, but the app does not load them.

## Run

```powershell
streamlit run app.py
```

Upload MP4, AVI, MOV, or MKV and select **Start detection**. The live annotated frame and processing progress appear on the page. The **Violations Dashboard** shows counts, all saved events, the evidence image, and the plate crop.

The dashboard stores OCR plate results as text in the `plate_number` database column. Use **Clear saved violations** there to delete all violation rows and their saved evidence and plate images; the app asks you to confirm first.

## Processing and storage

Ultralytics YOLO tracking with ByteTrack associates detections across frames. A no-helmet class must persist across four detections within a 15-frame window before an event is confirmed. After confirmation, that track is marked and no longer counted as a new violation. For speed, the plate model samples the rider area every five frames during a 60-frame capture window; the saved crop is selected using detector confidence and image sharpness, then passed to EasyOCR once. OCR text is normalized to uppercase alphanumeric characters, and fragments shorter than six characters are stored as `UNKNOWN`.

Annotated full-frame evidence and up to six timestamped context crops containing the rider and surrounding vehicle area are saved under `evidence/`; plate crops are saved under `plates/`. The dashboard displays the context crops for a selected violation and lets you choose one to inspect larger. SQLite creates `violations.db` and its tables on first run. Each event is stored once per source-video tracker key, and a recognized plate is stored once in the database even if the same vehicle appears in another upload. Database startup migrates older rows and removes duplicate entries, keeping a strongest readable plate row where one exists. Each violation row contains the timestamp, plate text, `NO_HELMET` status, evidence path, plate path (nullable), and uploaded video filename; timestamped context crop paths are kept in a related table. Keep the database and image folders together when moving the project.

## Notes

Processing the same video again can add another set of records. Detection and OCR quality still depend on plate size, focus, lighting, camera angle, and how closely the video resembles the models' evaluation footage. Review saved evidence before relying on a recognized plate.
