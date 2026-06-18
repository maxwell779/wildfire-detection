# -*- coding: utf-8 -*-
"""
🔥 산불·연기 조기탐지 관제 시스템 (산림청용) — 실모델 구동 대시보드
학습된 YOLOv8s(smoke/fire, mAP50 ≈ 0.78)로 이미지·영상에서 연기/불을 실시간 탐지하고
경보 단계(정상/연기 주의/화재 경보)를 산출하는 관제 대시보드.
"""
import os, glob, tempfile, datetime, numpy as np, cv2, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="산불 조기탐지 관제", page_icon="🔥", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "fire_smoke_yolov8s.pt")
SP = os.path.join(HERE, "samples")

st.markdown("""<style>
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.3rem;max-width:1350px;}
[data-testid="stMetric"]{background:#fff;border:1px solid #eef0f2;border-radius:14px;
  padding:14px 18px;box-shadow:0 1px 4px rgba(16,36,43,.06);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
h1,h2,h3{color:#3b0d0d;}
.hero{background:linear-gradient(100deg,#b91c1c,#1f2937);color:#fff;border-radius:16px;
  padding:20px 26px;margin-bottom:16px;}
.hero h1{color:#fff;margin:0;font-size:1.7rem;} .hero p{color:#fed7d7;margin:.3rem 0 0;}
.alert{border-radius:14px;padding:16px 20px;font-size:1.4rem;font-weight:800;margin:8px 0;}
.a-red{background:#fee2e2;color:#b91c1c;border:2px solid #ef4444;}
.a-orange{background:#fef3c7;color:#b45309;border:2px solid #f59e0b;}
.a-green{background:#dcfce7;color:#15803d;border:2px solid #22c55e;}
</style>""", unsafe_allow_html=True)

COLORS = {"smoke": (255, 170, 0), "fire": (0, 60, 255)}

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)

def annotate(img, res, names):
    out = img.copy(); counts = {"smoke": 0, "fire": 0}; dets = []
    for b in res.boxes:
        cls = names[int(b.cls)]; conf = float(b.conf); counts[cls] = counts.get(cls, 0)+1
        dets.append((cls, conf))
        x1, y1, x2, y2 = map(int, b.xyxy[0]); c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 3)
        cv2.putText(out, f"{cls} {conf:.2f}", (x1, max(14, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return out, counts, dets

def alert(counts):
    if counts.get("fire", 0) > 0: return ("🚨 화재 경보", "a-red", "즉시 출동 — 화염 감지")
    if counts.get("smoke", 0) > 0: return ("⚠️ 연기 주의", "a-orange", "연기 감지 — 현장 확인 필요")
    return ("🟢 정상", "a-green", "이상 없음")

try:
    model = load_model(); names = model.names; ok = True
except Exception as e:
    ok = False; st.error(f"모델 로드 실패: {e} — `pip install ultralytics`")

st.markdown('<div class="hero"><h1>🔥 산불·연기 조기탐지 관제 시스템</h1>'
            '<p>산림청 감시 카메라·드론 영상 기반 · YOLOv8s(smoke/fire) · mAP50 ≈ 0.78 · KDT 4인(본인=딥러닝 모델링·Hybrid)</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 탐지 설정")
    conf = st.slider("신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)
    iou = st.slider("IoU(중복 제거)", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.caption("**모델**: YOLOv8s · 2-class(smoke/fire)")
    st.caption("학습 14,122 / 검증 3,099장 · **mAP50 ≈ 0.78**")
    st.caption("Hybrid(YOLO+ResNet 2차검증)으로 구름·안개 오탐 저감 연구")

tab1, tab2 = st.tabs(["🖼 이미지 탐지", "🎬 영상 탐지"])

with tab1:
    samples = sorted(glob.glob(os.path.join(SP, "*.jpg")))
    pick = st.selectbox("📁 샘플 이미지(또는 아래 업로드)", ["(선택)"] + [os.path.basename(p) for p in samples])
    up = st.file_uploader("⬆️ 감시 이미지 업로드", type=["jpg", "jpeg", "png"], key="img")
    img = None
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    elif pick != "(선택)":
        img = cv2.imread(os.path.join(SP, pick))
    if ok and img is not None:
        res = model(img, conf=conf, iou=iou, verbose=False)[0]
        out, counts, dets = annotate(img, res, names)
        lv, cls_, msg = alert(counts)
        st.markdown(f'<div class="alert {cls_}">{lv} &nbsp;—&nbsp; {msg} '
                    f'<span style="font-size:.9rem;font-weight:600;color:#475569">{datetime.datetime.now():%H:%M:%S}</span></div>',
                    unsafe_allow_html=True)
        cL, cR = st.columns([3, 1.1])
        with cL:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True, caption="탐지 결과")
        with cR:
            st.metric("🔥 화염", counts.get("fire", 0))
            st.metric("💨 연기", counts.get("smoke", 0))
            st.metric("총 탐지", len(dets))
            if dets:
                fig = go.Figure(go.Bar(
                    x=[d[1] for d in dets][::-1],
                    y=[f"{d[0]} #{i+1}" for i, d in enumerate(dets)][::-1],
                    orientation="h",
                    marker_color=["#ef4444" if d[0]=="fire" else "#f59e0b" for d in dets][::-1]))
                fig.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
                                  xaxis_title="신뢰도", xaxis_range=[0,1], template="simple_white")
                st.plotly_chart(fig, use_container_width=True)
    elif ok:
        st.info("⬆️ 샘플을 고르거나 이미지를 올리면 연기/불을 탐지하고 경보 단계를 표시합니다.")

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
                res = model(fr, conf=conf, iou=iou, verbose=False)[0]
                out, counts, _ = annotate(fr, res, names); lv, cls_, msg = alert(counts)
                mf = max(mf, counts.get("fire", 0)); ms = max(ms, counts.get("smoke", 0))
                ff += 1 if counts.get("fire", 0) else 0
                ph_a.markdown(f'<div class="alert {cls_}">{lv} — {msg}</div>', unsafe_allow_html=True)
                ph_i.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                if total: prog.progress(min(idx/total, 1.0))
            idx += 1
        cap.release(); prog.progress(1.0); os.unlink(tf.name)
        st.success(f"분석 완료 · 최대 화염 {mf} / 최대 연기 {ms} · 화재 감지 프레임 {ff}개")
        if mf > 0: st.error("🚨 영상에서 화재 감지 — 관할 기관 통보 권장")
    elif ok:
        st.info("⬆️ 짧은 감시 영상(10~20초)을 올리면 프레임을 샘플링해 추적합니다.")

st.caption("GitHub: wildfire-detection · 본인 역할: 딥러닝 모델링·Hybrid 구조 (데이터 수집·라벨링은 팀 공동) · KDT 4인")
