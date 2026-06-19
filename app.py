# -*- coding: utf-8 -*-
"""
🔥 산불·연기 조기탐지 관제 시스템 (산림청용) — 실모델 구동 대시보드
학습된 YOLOv8s(smoke/fire, mAP50 ≈ 0.78)로 이미지·영상에서 연기/불을 실시간 탐지.
감시 카메라처럼 샘플이 자동 순환하며 즉시 탐지 결과를 보여줍니다.
"""
import os, glob, tempfile, datetime, numpy as np, cv2, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="산불 조기탐지 관제", page_icon="🔥", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "fire_smoke_yolov8s.pt")
SP = os.path.join(HERE, "samples")
COLORS = {"smoke": (255, 170, 0), "fire": (0, 60, 255)}

st.markdown("""<style>
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.2rem;max-width:1350px;}
[data-testid="stMetric"]{background:#fff;border:1px solid #eef0f2;border-radius:14px;
  padding:14px 18px;box-shadow:0 1px 4px rgba(16,36,43,.06);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
h1,h2,h3{color:#3b0d0d;}
.hero{background:linear-gradient(100deg,#b91c1c,#1f2937);color:#fff;border-radius:16px;
  padding:20px 26px;margin-bottom:16px;}
.hero h1{color:#fff;margin:0;font-size:1.7rem;} .hero p{color:#fed7d7;margin:.3rem 0 0;}
.alert{border-radius:14px;padding:14px 20px;font-size:1.35rem;font-weight:800;margin:6px 0;}
.a-red{background:#fee2e2;color:#b91c1c;border:2px solid #ef4444;}
.a-orange{background:#fef3c7;color:#b45309;border:2px solid #f59e0b;}
.a-green{background:#dcfce7;color:#15803d;border:2px solid #22c55e;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)

model = None
try:
    model = load_model(); names = model.names; ok = True
except Exception as e:
    ok = False; st.error(f"모델 로드 실패: {e} — `pip install ultralytics`")

def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

@st.cache_data(show_spinner=False)
def detect(path, conf, iou):
    """경로 이미지 탐지 → (RGB ndarray, counts, dets). 결과 캐시로 순환이 빠름."""
    img = imread_u(path)
    res = load_model()(img, conf=conf, iou=iou, verbose=False)[0]
    counts = {"smoke": 0, "fire": 0}; dets = []
    for b in res.boxes:
        cls = res.names[int(b.cls)]; cf = float(b.conf)
        counts[cls] = counts.get(cls, 0) + 1; dets.append((cls, cf))
        x1, y1, x2, y2 = map(int, b.xyxy[0]); c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        cv2.putText(img, f"{cls} {cf:.2f}", (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), counts, dets

def detect_arr(img_bgr, conf, iou):
    res = load_model()(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    counts = {"smoke": 0, "fire": 0}; dets = []
    for b in res.boxes:
        cls = res.names[int(b.cls)]; cf = float(b.conf)
        counts[cls] = counts.get(cls, 0) + 1; dets.append((cls, cf))
        x1, y1, x2, y2 = map(int, b.xyxy[0]); c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), c, 3)
        cv2.putText(img_bgr, f"{cls} {cf:.2f}", (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), counts, dets

def alert(counts):
    if counts.get("fire", 0) > 0: return ("🚨 화재 경보", "a-red", "즉시 출동 — 화염 감지")
    if counts.get("smoke", 0) > 0: return ("⚠️ 연기 주의", "a-orange", "연기 감지 — 현장 확인 필요")
    return ("🟢 정상", "a-green", "이상 없음")

def render(rgb, counts, dets, cap):
    lv, cls_, msg = alert(counts)
    st.markdown(f'<div class="alert {cls_}">{lv} &nbsp;—&nbsp; {msg} '
                f'<span style="font-size:.85rem;font-weight:600;color:#475569">{datetime.datetime.now():%H:%M:%S}</span></div>',
                unsafe_allow_html=True)
    cL, cR = st.columns([3, 1.1])
    with cL:
        st.image(rgb, use_container_width=True, caption=cap)
    with cR:
        st.metric("🔥 화염", counts.get("fire", 0))
        st.metric("💨 연기", counts.get("smoke", 0))
        st.metric("총 탐지", len(dets))
        if dets:
            fig = go.Figure(go.Bar(x=[d[1] for d in dets][::-1],
                                   y=[f"{d[0]} #{i+1}" for i, d in enumerate(dets)][::-1],
                                   orientation="h",
                                   marker_color=["#ef4444" if d[0] == "fire" else "#f59e0b" for d in dets][::-1]))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0),
                              xaxis_title="신뢰도", xaxis_range=[0, 1], template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="hero"><h1>🔥 산불·연기 조기탐지 관제 시스템</h1>'
            '<p>산림청 감시 카메라·드론 영상 기반 · YOLOv8s(smoke/fire) · mAP50 ≈ 0.78 · KDT 4인(본인=딥러닝 모델링·Hybrid)</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 탐지 설정")
    conf = st.slider("신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)
    iou = st.slider("IoU(중복 제거)", 0.1, 0.9, 0.45, 0.05)
    speed = st.slider("순환 주기(초)", 1.5, 6.0, 3.0, 0.5)
    st.divider()
    st.caption("**모델**: YOLOv8s · 2-class(smoke/fire)")
    st.caption("학습 14,122 / 검증 3,099장 · **mAP50 ≈ 0.78**")

tab1, tab2 = st.tabs(["🛰 실시간 관제 (자동 순환)", "🎬 영상 탐지"])

with tab1:
    samples = sorted(glob.glob(os.path.join(SP, "*.jpg")))
    if ok and samples:
        if "wf_i" not in st.session_state: st.session_state.wf_i = 0

        @st.fragment(run_every=f"{speed}s")
        def feed():
            i = st.session_state.wf_i % len(samples)
            p = samples[i]
            rgb, counts, dets = detect(p, conf, iou)
            st.caption(f"📡 감시 채널 {i+1}/{len(samples)} · {os.path.basename(p)} (자동 순환 중)")
            render(rgb, counts, dets, "실시간 탐지 결과")
            st.session_state.wf_i = i + 1
        feed()

        with st.expander("⬆️ 직접 이미지 업로드해서 탐지"):
            up = st.file_uploader("감시 이미지", type=["jpg", "jpeg", "png"], key="img")
            if up is not None:
                img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                rgb, counts, dets = detect_arr(img, conf, iou)
                render(rgb, counts, dets, "업로드 탐지 결과")
    elif ok:
        st.info("samples 폴더에 이미지가 없습니다.")

with tab2:
    vid = st.file_uploader("감시 영상 업로드 (mp4/avi/mov)", type=["mp4", "avi", "mov"], key="vid")
    if ok and vid:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tf.write(vid.getvalue()); tf.close()
        cap = cv2.VideoCapture(tf.name); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // 30)
        ph_a, ph_i, prog = st.empty(), st.empty(), st.progress(0)
        mf = ms = ff = idx = 0
        while True:
            ret, fr = cap.read()
            if not ret: break
            if idx % step == 0:
                rgb, counts, _ = detect_arr(fr, conf, iou); lv, cls_, msg = alert(counts)
                mf = max(mf, counts.get("fire", 0)); ms = max(ms, counts.get("smoke", 0))
                ff += 1 if counts.get("fire", 0) else 0
                ph_a.markdown(f'<div class="alert {cls_}">{lv} — {msg}</div>', unsafe_allow_html=True)
                ph_i.image(rgb, use_container_width=True)
                if total: prog.progress(min(idx / total, 1.0))
            idx += 1
        cap.release(); prog.progress(1.0); os.unlink(tf.name)
        st.success(f"분석 완료 · 최대 화염 {mf} / 최대 연기 {ms} · 화재 감지 프레임 {ff}개")
        if mf > 0: st.error("🚨 영상에서 화재 감지 — 관할 기관 통보 권장")
    elif ok:
        st.info("⬆️ 짧은 감시 영상(10~20초)을 올리면 프레임을 샘플링해 추적합니다.")

st.caption("GitHub: wildfire-detection · 본인 역할: 딥러닝 모델링·Hybrid 구조 (데이터 수집·라벨링은 팀 공동) · KDT 4인")
