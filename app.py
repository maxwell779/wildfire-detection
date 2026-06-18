# -*- coding: utf-8 -*-
"""
🔥 산불·연기 조기탐지 관제 시스템 (산림청용) — 실모델 구동
학습된 YOLOv8s(smoke/fire 2-class, mAP50 ≈ 0.78)로 이미지·영상에서 연기/불을 실시간 탐지하고
경보 단계(정상/주의/경보)를 산출합니다.
"""
import os, tempfile, datetime, numpy as np, cv2, streamlit as st

st.set_page_config(page_title="산불 조기탐지 관제", page_icon="🔥", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "fire_smoke_yolov8s.pt")

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)

COLORS = {"smoke": (255, 170, 0), "fire": (0, 60, 255)}  # BGR

def annotate(img_bgr, res, names):
    out = img_bgr.copy()
    counts = {"smoke": 0, "fire": 0}
    for b in res.boxes:
        cls = names[int(b.cls)]; conf = float(b.conf)
        counts[cls] = counts.get(cls, 0) + 1
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{cls} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return out, counts

def alert_level(counts):
    if counts.get("fire", 0) > 0:
        return "🚨 화재 경보", "red", "즉시 출동 — 화염 감지"
    if counts.get("smoke", 0) > 0:
        return "⚠️ 연기 주의", "orange", "연기 감지 — 현장 확인 필요"
    return "🟢 정상", "green", "이상 없음"

st.title("🔥 산불·연기 조기탐지 관제 시스템")
st.caption("산림청 감시 카메라/드론 영상 기반 · YOLOv8s (smoke/fire) · mAP50 ≈ 0.78")

try:
    model = load_model(); names = model.names; ok = True
except Exception as e:
    ok = False; st.error(f"모델 로드 실패: {e}\n`pip install ultralytics` 필요")

with st.sidebar:
    st.header("⚙️ 탐지 설정")
    conf = st.slider("신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)
    iou = st.slider("IoU(중복 제거)", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.caption("**모델**: YOLOv8s · 2-class(smoke/fire)")
    st.caption("학습 14,122장 / 검증 3,099장 · mAP50 ≈ 0.78")
    st.caption("Hybrid(YOLO+ResNet 2차검증)으로 구름·안개 오탐 저감 연구")

tab1, tab2 = st.tabs(["🖼 이미지 탐지", "🎬 영상 탐지"])

with tab1:
    up = st.file_uploader("감시 이미지 업로드", type=["jpg", "jpeg", "png"], key="img")
    if ok and up:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        res = model(img, conf=conf, iou=iou, verbose=False)[0]
        out, counts = annotate(img, res, names)
        level, color, msg = alert_level(counts)
        st.markdown(f"## :{color}[{level}]  —  {msg}  `{datetime.datetime.now():%H:%M:%S}`")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True, caption="탐지 결과")
        with c2:
            st.metric("🔥 화염", counts.get("fire", 0))
            st.metric("💨 연기", counts.get("smoke", 0))
    elif ok:
        st.info("⬆️ 산림 감시 이미지를 올리면 연기/불을 탐지하고 경보 단계를 표시합니다.")

with tab2:
    vid = st.file_uploader("감시 영상 업로드 (mp4/avi)", type=["mp4", "avi", "mov"], key="vid")
    if ok and vid:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(vid.getvalue()); tf.close()
        cap = cv2.VideoCapture(tf.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // 30)  # 약 30프레임 샘플
        ph_img = st.empty(); ph_alert = st.empty(); prog = st.progress(0)
        max_fire = max_smoke = 0; fire_frames = 0; idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                res = model(frame, conf=conf, iou=iou, verbose=False)[0]
                out, counts = annotate(frame, res, names)
                level, color, msg = alert_level(counts)
                max_fire = max(max_fire, counts.get("fire", 0))
                max_smoke = max(max_smoke, counts.get("smoke", 0))
                if counts.get("fire", 0) > 0: fire_frames += 1
                ph_alert.markdown(f"### :{color}[{level}] — {msg}")
                ph_img.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                if total: prog.progress(min(idx / total, 1.0))
            idx += 1
        cap.release(); prog.progress(1.0)
        os.unlink(tf.name)
        st.success(f"분석 완료 · 최대 화염 {max_fire} / 최대 연기 {max_smoke} · 화재 감지 프레임 {fire_frames}")
        if max_fire > 0:
            st.error("🚨 영상에서 화재가 감지되었습니다 — 관할 기관 통보 권장")
    elif ok:
        st.info("⬆️ 감시 영상을 올리면 프레임을 샘플링해 연기/불을 추적합니다.")

st.divider()
st.caption("GitHub: wildfire-detection · 본인 역할: 딥러닝 모델링·Hybrid 구조 담당 (데이터 수집·라벨링은 팀 공동) · KDT 4인")
