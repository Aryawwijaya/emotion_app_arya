import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import pandas as pd

model = load_model("emotion_model.keras")
class_names = ['marah', 'netral', 'sedih', 'senang', 'terkejut']
negative_emotions = ['marah', 'sedih']

mp_face_detection = mp.solutions.face_detection

st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")
st.title("Deteksi Emosi Wajah Real-Time")

class EmotionProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        self.emotions = []
        self.last_time = time.time()
        self.current_label = "..."

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)

                face = rgb[y:y + h, x:x + w]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)

                if time.time() - self.last_time >= 2:
                    pred = model.predict(face_input, verbose=0)
                    label_idx = np.argmax(pred)
                    self.current_label = class_names[label_idx]

                    if self.current_label != "netral":
                        self.emotions.append(self.current_label)

                    self.last_time = time.time()

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, self.current_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx.video_processor:
    if st.button("Lihat Ringkasan"):
        emo_list = ctx.video_processor.emotions
        total = len(emo_list)
        if total == 0:
            st.warning("Belum ada emosi yang terdeteksi!.")
        else:
            st.subheader("Ringkasan Deteksi Emosi")
            summary = {}
            for cls in class_names:
                if cls == "netral":
                    continue
                count = emo_list.count(cls)
                pct = (count / total) * 100
                summary[cls] = pct
                st.write(f"- {cls}: {pct:.2f}%")

            neg_pct = sum([summary.get(e, 0) for e in negative_emotions])
            if neg_pct < 20:
                phq, kategori = "0–4", "Tidak ada"
            elif neg_pct < 40:
                phq, kategori = "5–9", "Ringan"
            elif neg_pct < 60:
                phq, kategori = "10–14", "Sedang"
            elif neg_pct < 80:
                phq, kategori = "15–19", "Sedang berat"
            else:
                phq, kategori = "20–24", "Berat"

            st.subheader("Estimasi Skor PHQ-8")
            st.write(f"Skor: **{phq}**, Kategori: **{kategori}**")

            df_summary = pd.DataFrame(list(summary.items()), columns=["Emosi", "Persentase"])
            df_summary.loc[len(df_summary)] = ["PHQ-8 Skor", phq]
            df_summary.loc[len(df_summary)] = ["Kategori", kategori]

            csv = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Ringkasan (CSV)",
                data=csv,
                file_name="ringkasan_emosi.csv",
                mime="text/csv",
            )

with st.expander("Petunjuk Penggunaan"):
    st.markdown("""
    ### Cara Menggunakan:
    1. Klik tombol **Start** untuk mengaktifkan kamera.
    2. Posisikan wajah dengan pencahayaan cukup.
    3. Tahan ekspresi selama beberapa detik agar sistem mengenali.
    4. Klik **Lihat Ringkasan** untuk hasil deteksi emosi dan estimasi PHQ-8.
    5. Tekan tombol **Download CSV** untuk menyimpan ringkasan.

    ### Tips:
    - Jaga jarak 25–50 cm dari kamera.
    - Hindari cahaya dari belakang (backlight).
    - Ekspresikan emosi dengan jelas agar terdeteksi.
    """)
