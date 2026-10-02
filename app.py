from pathlib import Path
import shutil
import tempfile
import cv2
import streamlit as st

from database.database import ViolationDatabase
from detection.pipeline import (
    HELMET_MODEL_PATH,
    PLATE_MODEL_PATH,
    HELMET_CONFIDENCE,
    PLATE_CONFIDENCE,
    IOU_THRESHOLD,
    process_video,
)

st.set_page_config(page_title="Helmet Violation Detection", page_icon="🪖", layout="wide")
st.title("Helmet Violation Detection")
st.caption(
    "Detect helmet violations in traffic videos using custom YOLO models + OCR.  "
    "Violations are saved to SQLite with plate crop and evidence images."
)

ROOT = Path(__file__).resolve().parent

with st.sidebar:
    st.header("⚙️ Model settings")
    helmet_path = st.text_input("Helmet model (.pt)", HELMET_MODEL_PATH)
    plate_path  = st.text_input("Number plate model (.pt)", PLATE_MODEL_PATH)

    st.divider()
    st.subheader("Detection thresholds")
    helmet_conf = st.slider(
        "Helmet confidence",
        min_value=0.10, max_value=0.95,
        value=float(HELMET_CONFIDENCE), step=0.05,
        help="Minimum confidence to accept a helmet/no-helmet detection.",
    )
    plate_conf = st.slider(
        "Plate confidence",
        min_value=0.10, max_value=0.95,
        value=float(PLATE_CONFIDENCE), step=0.05,
        help="Minimum confidence to accept a number-plate detection.",
    )
    iou_thresh = st.slider(
        "IoU threshold (NMS)",
        min_value=0.10, max_value=0.90,
        value=float(IOU_THRESHOLD), step=0.05,
        help="Non-maximum suppression IoU threshold for both models.",
    )

    st.divider()
    st.subheader("Performance")
    frame_stride = st.slider(
        "Frame stride",
        min_value=1, max_value=10, value=1, step=1,
        help="Process every Nth frame. 1 = every frame (most accurate). "
             "Higher values are faster but reduce temporal tracking quality.",
    )

    st.divider()
    debug_mode = st.toggle(
        "🐛 Debug mode",
        value=False,
        help=(
            "Shows helmet confidence, plate confidence, OCR confidence, "
            "and Track IDs on the evidence image and in the console log."
        ),
    )

# ── Tabs ────────────────────────────────────────────────────────────────────
video_tab, dashboard_tab = st.tabs(["🎥 Video Detection", "📋 Violations Dashboard"])

# ── Video Detection tab ──────────────────────────────────────────────────────
with video_tab:
    uploaded = st.file_uploader(
        "Upload a traffic video", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded:
        st.video(uploaded)
        if st.button("▶ Start detection", type="primary"):
            missing = []
            if not Path(helmet_path).is_file():
                missing.append(f"Helmet model not found: `{helmet_path}`")
            if not Path(plate_path).is_file():
                missing.append(f"Plate model not found: `{plate_path}`")
            if missing:
                for m in missing:
                    st.error(m)
            else:
                progress = st.progress(0.0, text="Starting inference…")
                live     = st.empty()
                temp_dir = Path(tempfile.mkdtemp(prefix="helmet_video_"))
                video_file = temp_dir / Path(uploaded.name).name
                video_file.write_bytes(uploaded.getbuffer())

                try:
                    def _on_frame(frame):
                        live.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            caption="Live annotated frame",
                            use_container_width=True,
                        )

                    records, device = process_video(
                        str(video_file),
                        helmet_model       = helmet_path,
                        plate_model        = plate_path,
                        helmet_confidence  = helmet_conf,
                        plate_confidence   = plate_conf,
                        iou_threshold      = iou_thresh,
                        frame_stride       = frame_stride,
                        debug              = debug_mode,
                        on_progress        = lambda v: progress.progress(
                            v, text=f"Processing… {v:.0%}"
                        ),
                        on_frame           = _on_frame,
                    )

                    if records:
                        st.success(
                            f"✅ Done ({device}). "
                            f"Saved **{len(records)}** new violation(s). "
                            "Switch to the **Violations Dashboard** tab to review them."
                        )
                    else:
                        st.info(
                            f"✅ Processing complete ({device}). "
                            "No new violations were confirmed in this video.  \n"
                            "If violations were expected, try **lowering the Helmet confidence** "
                            "threshold or check that the model classes are correct."
                        )

                except Exception as exc:
                    st.exception(exc)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    if debug_mode:
        st.info(
            "🐛 **Debug mode active.**  "
            "Confidence values and Track IDs will appear on the live preview "
            "and evidence images. Check the terminal / console for detailed logs."
        )

# ── Dashboard tab ────────────────────────────────────────────────────────────
with dashboard_tab:
    database = ViolationDatabase(str(ROOT / "violations.db"))
    records  = database.all()
    today    = __import__("datetime").date.today().isoformat()

    unique_plates = {
        r["plate_number"]
        for r in records
        if r.get("plate_number") and r["plate_number"] != "UNKNOWN"
    }
    todays = [r for r in records if r.get("timestamp", "").startswith(today)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Violations", len(records))
    c2.metric("Unique Number Plates", len(unique_plates))
    c3.metric("Today's Violations",  len(todays))

    # ── Clear all ──
    with st.expander("🗑️ Clear saved violations"):
        st.warning(
            "This permanently removes **all** violation rows and their "
            "saved evidence and plate images."
        )
        confirm_clear = st.checkbox(
            "I understand; delete all saved violations",
            key="confirm_clear_violations",
        )
        if st.button("Delete all violations", disabled=not confirm_clear, type="secondary"):
            image_paths = []
            for r in records:
                for col in ("violation_image_path", "evidence_image_path", "plate_image_path"):
                    val = r.get(col)
                    if val:
                        image_paths.append(val)
                image_paths.extend(
                    sample["image_path"] for sample in database.image_samples(r["id"])
                )
            database.clear()
            for img in image_paths:
                target = Path(img)
                if not target.is_absolute():
                    target = ROOT / target
                try:
                    if target.resolve().is_relative_to(ROOT.resolve()) and target.is_file():
                        target.unlink()
                except OSError:
                    pass
            st.session_state["confirm_clear_violations"] = False
            st.success("All violation records and images deleted.")
            st.rerun()

    if not records:
        st.info("No violations have been saved yet.")
    else:
        # ── Summary table ──
        table_rows = []
        for r in records:
            table_rows.append({
                "ID":           r["id"],
                "Date / Time":  r.get("timestamp", ""),
                "Track ID":     r.get("track_id", ""),
                "Plate Number": r.get("plate_number", "UNKNOWN"),
                "Helm conf":    f"{r.get('helmet_confidence') or 0:.2f}",
                "Plate conf":   f"{r.get('plate_confidence') or 0:.2f}",
                "Status":       r.get("helmet_status", ""),
                "Evidence":     f"🖼️ View image — {Path(r.get('violation_image_path') or r.get('evidence_image_path', '')).name}",
            })
        st.caption("Click any row, including its Evidence cell, to show the images below.")
        table_event = st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="violations_table",
        )

        # ── Detail view ──
        selected_rows = table_event.selection.rows
        if not selected_rows:
            st.info("Select a violation row above to view its evidence and plate images.")
            sel = None
        else:
            sel = records[selected_rows[0]]

        if sel:
            st.subheader(f"Violation #{sel['id']}  —  {sel.get('plate_number', 'UNKNOWN')}")
            left, right = st.columns(2)

            with left:
                st.markdown(
                    f"**Plate number:** `{sel.get('plate_number', 'UNKNOWN')}`  \n"
                    f"**Timestamp:** {sel.get('timestamp', '')}  \n"
                    f"**Track ID:** {sel.get('track_id', 'N/A')}  \n"
                    f"**Helmet status:** {sel.get('helmet_status', '')}  \n"
                    f"**Helmet confidence:** {sel.get('helmet_confidence') or 0:.2f}  \n"
                    f"**Plate confidence:** {sel.get('plate_confidence') or 0:.2f}  \n"
                    f"**Video source:** {sel.get('video_source', '')}  \n"
                )

                # Evidence image
                ev_path = sel.get("violation_image_path") or sel.get("evidence_image_path")
                if ev_path:
                    ev_file = Path(ev_path) if Path(ev_path).is_absolute() else ROOT / ev_path
                    if ev_file.is_file():
                        st.image(str(ev_file), caption="Violation evidence image", use_container_width=True)
                    else:
                        st.warning(f"Evidence image not found: {ev_path}")

            with right:
                st.markdown("#### Number plate crop")
                plate_img = sel.get("plate_image_path")
                if plate_img:
                    plate_file = Path(plate_img) if Path(plate_img).is_absolute() else ROOT / plate_img
                    if plate_file.is_file():
                        st.image(str(plate_file), caption=f"Plate: {sel.get('plate_number', 'UNKNOWN')}", use_container_width=True)
                    else:
                        st.warning(f"Plate image not found: {plate_img}")
                else:
                    st.info("No plate image was detected for this violation.")

            st.subheader("Vehicle and rider crops at different video times")
            samples = database.image_samples(sel["id"])
            available_samples = []
            for sample in samples:
                sample_path = Path(sample["image_path"])
                if not sample_path.is_absolute():
                    sample_path = ROOT / sample_path
                if sample_path.is_file():
                    available_samples.append((sample, sample_path))

            if available_samples:
                sample_columns = st.columns(3)
                for idx, (sample, sample_path) in enumerate(available_samples):
                    seconds = sample["video_seconds"]
                    timestamp = f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
                    with sample_columns[idx % 3]:
                        st.image(
                            str(sample_path),
                            caption=f"Capture {sample['sample_index']} · {timestamp}",
                            use_container_width=True,
                        )

                selected_sample_index = st.selectbox(
                    "Choose a rider crop to inspect larger",
                    range(len(available_samples)),
                    format_func=lambda i: (
                        f"Capture {available_samples[i][0]['sample_index']} · "
                        f"{int(available_samples[i][0]['video_seconds'] // 60):02d}:"
                        f"{available_samples[i][0]['video_seconds'] % 60:05.2f}"
                    ),
                    key=f"rider_crop_{sel['id']}",
                )
                selected_sample, selected_sample_path = available_samples[selected_sample_index]
                selected_seconds = selected_sample["video_seconds"]
                selected_timestamp = f"{int(selected_seconds // 60):02d}:{selected_seconds % 60:05.2f}"
                st.image(
                    str(selected_sample_path),
                    caption=f"Selected rider crop · {selected_timestamp}",
                    use_container_width=True,
                )
            else:
                    st.info("No timestamped vehicle crops are saved for this older violation.")
