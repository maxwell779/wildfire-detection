# -*- coding: utf-8 -*-
"""
🔥 산불·연기 조기탐지 관제 시스템 (산림청용) — 실모델 구동 대시보드
학습된 YOLOv8s(smoke/fire, mAP50 ≈ 0.78)로 이미지·영상에서 연기/불을 실시간 탐지.
감시 카메라처럼 샘플이 자동 순환하며 즉시 탐지 결과를 보여줍니다.
"""
import os, glob, tempfile, datetime, numpy as np, pandas as pd, cv2, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="산불 조기탐지 관제", page_icon="🔥", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "fire_smoke_yolov8s.pt")
SP = os.path.join(HERE, "samples")
COLORS = {"smoke": (255, 170, 0), "fire": (0, 60, 255)}

st.markdown("""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, [class*="css"], .stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"]{
  font-family:'Pretendard','Malgun Gothic',sans-serif;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
.block-container{padding-top:1.2rem;max-width:1350px;}
[data-testid="stMetric"]{background:#fff;border:1px solid #f0e6e6;border-radius:16px;
  padding:14px 18px;box-shadow:0 2px 10px rgba(59,13,13,.05);transition:transform .15s, box-shadow .15s;}
[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(185,28,28,.12);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
[data-testid="stMetricValue"]{color:#3b0d0d;font-weight:800;}
h1,h2,h3,h4{color:#3b0d0d;letter-spacing:-.3px;}
.hero{background:linear-gradient(110deg,#b91c1c 0%,#1f2937 100%);color:#fff;border-radius:18px;
  padding:22px 28px;margin-bottom:16px;box-shadow:0 12px 32px rgba(185,28,28,.20);}
.hero h1{color:#fff;margin:0;font-size:1.85rem;font-weight:800;letter-spacing:-.5px;} .hero p{color:#fed7d7;margin:.4rem 0 0;font-size:.97rem;}
.hero .chip{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.28);
  color:#ffecec;border-radius:20px;padding:4px 13px;font-size:.8rem;font-weight:600;margin:11px 6px 0 0;}
.alert{border-radius:14px;padding:13px 20px;font-size:1.25rem;font-weight:800;margin:6px 0;}
.a-red{background:#fee2e2;color:#b91c1c;border:2px solid #ef4444;}
.a-orange{background:#fef3c7;color:#b45309;border:2px solid #f59e0b;}
.a-green{background:#dcfce7;color:#15803d;border:2px solid #22c55e;}
.statebox{background:#fff;border:1px solid #f0e6e6;border-radius:14px;padding:14px 18px;box-shadow:0 2px 10px rgba(59,13,13,.05);}
.statebox b{color:#3b0d0d;} .statebox ul{margin:.5rem 0 0;padding-left:1.1rem;} .statebox li{margin:.25rem 0;color:#334155;font-size:.92rem;}
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

def overlay_label(rgb, text, color=(255, 255, 255)):
    """프레임 좌상단에 'Frame N | SMOKE DETECTED' 라벨을 영상처럼 올린다."""
    img = rgb.copy()
    cv2.putText(img, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
    return img

def stabilized_state(hist):
    """최근 프레임(fire,smoke) 이력으로 깜빡임을 누른 '안정화 경고'. (오탐 억제 = Hybrid 취지)"""
    recent = hist[-3:]
    fires = sum(1 for f, s in recent if f)
    smokes = sum(1 for f, s in recent if s)
    if fires >= 2: return ("🚨 화재 발생", "a-red")
    if smokes >= 2: return ("⚠️ 연기 감지", "a-orange")
    if recent and (recent[-1][0] or recent[-1][1]): return ("🟡 관찰 중", "a-orange")
    return ("🟢 정상", "a-green")

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
            '<p>산림청 감시 카메라·드론 영상 기반 · YOLOv8s(smoke/fire) · mAP50 ≈ 0.78 · KDT 4인(본인=딥러닝 모델링·Hybrid)</p>'
            '<div><span class="chip">🔥 YOLOv8s mAP50 0.78</span><span class="chip">🛡 Hybrid 오탐 억제</span>'
            '<span class="chip">📡 실시간 관제 + 안정화 경고</span></div></div>',
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
        # 합성 카메라 메타(강원·경북·지리·한라 산지 관제 컨셉)
        CAMS = [("CH1 설악-A", 38.12, 128.46), ("CH2 오대-B", 37.79, 128.54),
                ("CH3 태백-C", 37.16, 128.99), ("CH4 소백-D", 36.96, 128.46),
                ("CH5 지리-E", 35.34, 127.73), ("CH6 한라-F", 33.36, 126.53)]
        if "wf_focus" not in st.session_state: st.session_state.wf_focus = 0
        if "wf_hist" not in st.session_state: st.session_state.wf_hist = []

        # 전 채널 탐지(detect는 캐시되어 빠름)
        feeds = []
        for idx, p in enumerate(samples[:len(CAMS)]):
            rgb, counts, dets = detect(p, conf, iou)
            fire, smoke = counts.get("fire", 0), counts.get("smoke", 0)
            nm, la, lo = CAMS[idx]
            feeds.append(dict(i=idx, name=nm, lat=la, lon=lo, rgb=rgb, fire=fire, smoke=smoke,
                              dets=dets, lv=("fire" if fire else "smoke" if smoke else "clear")))
        nfire = sum(1 for f in feeds if f["lv"] == "fire")
        nsmoke = sum(1 for f in feeds if f["lv"] == "smoke")
        allconf = [cf for f in feeds for _, cf in f["dets"]]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📡 감시 채널", len(feeds))
        k2.metric("🚨 화재 경보", nfire, f"{nfire}건" if nfire else None, delta_color="inverse")
        k3.metric("⚠️ 연기 주의", nsmoke, f"{nsmoke}건" if nsmoke else None, delta_color="inverse")
        k4.metric("평균 탐지 신뢰도", f"{np.mean(allconf)*100:.0f}%" if allconf else "—")

        st.divider()
        cMap, cSum = st.columns([1.5, 1])
        with cMap:
            st.markdown("#### 🗺 관제 지도 — 카메라 위치·경보 강조")
            cmap = {"fire": "#ef4444", "smoke": "#f59e0b", "clear": "#22c55e"}
            gmap = go.Figure(go.Scattergeo(
                lon=[f["lon"] for f in feeds], lat=[f["lat"] for f in feeds],
                text=[f"{f['name']}<br>🔥{f['fire']} 💨{f['smoke']}" for f in feeds],
                mode="markers+text", textposition="top center", textfont=dict(size=10),
                marker=dict(size=[22 if f["lv"] != "clear" else 12 for f in feeds],
                            color=[cmap[f["lv"]] for f in feeds],
                            line=dict(width=1, color="white"))))
            gmap.update_geos(scope="asia", center=dict(lat=36.6, lon=127.9), projection_scale=5.2,
                             showland=True, landcolor="#eef3f0", showocean=True, oceancolor="#dbeafe",
                             showcountries=True, countrycolor="#cbd5e1", resolution=50)
            gmap.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(gmap, use_container_width=True)
        with cSum:
            st.markdown("#### 📊 채널 상태 요약")
            for f in feeds:
                badge = "🔴" if f["lv"] == "fire" else "🟠" if f["lv"] == "smoke" else "🟢"
                st.markdown(f"{badge} **{f['name']}** &nbsp; 🔥 {f['fire']} · 💨 {f['smoke']}")

        st.divider()
        st.markdown("#### 📺 다중 카메라 관제 월 (6채널 동시)")
        grid = st.columns(3)
        for f in feeds:
            with grid[f["i"] % 3]:
                st.image(f["rgb"], use_container_width=True)
                tag = "🔴 화재" if f["lv"] == "fire" else "🟠 연기" if f["lv"] == "smoke" else "🟢 정상"
                st.caption(f"{f['name']} · {tag} · 🔥{f['fire']} 💨{f['smoke']}")

        st.divider()
        st.markdown("#### 🔍 포커스 채널 (자동 순환 · 정밀 분석)")

        @st.fragment(run_every=f"{speed}s")
        def focus():
            f = feeds[st.session_state.wf_focus % len(feeds)]
            st.session_state.wf_hist = (st.session_state.wf_hist + [(f["fire"] > 0, f["smoke"] > 0)])[-6:]
            stab_txt, stab_cls = stabilized_state(st.session_state.wf_hist)
            cL, cR = st.columns([2.2, 1])
            with cL:
                fs = "FIRE DETECTED" if f["fire"] else "SMOKE DETECTED" if f["smoke"] else "CLEAR"
                lab = (90, 90, 255) if f["fire"] else (0, 200, 255) if f["smoke"] else (120, 255, 120)
                st.image(overlay_label(f["rgb"], f"{f['name']} | {fs}", lab), use_container_width=True)
                lv, cls_, msg = alert({"fire": f["fire"], "smoke": f["smoke"]})
                st.markdown(f'<div class="alert {cls_}">현재 프레임 상태: {lv} — {msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="alert {stab_cls}">안정화된 경고 상태(최근 3프레임 다수결): {stab_txt}</div>', unsafe_allow_html=True)
            with cR:
                risk = min(100, f["fire"] * 45 + f["smoke"] * 20)
                gc = "#ef4444" if risk >= 60 else "#f59e0b" if risk >= 25 else "#22c55e"
                gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk, number={'suffix': "%"},
                    title={'text': "화재 위험도"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gc},
                           'steps': [{'range': [0, 25], 'color': '#dcfce7'},
                                     {'range': [25, 60], 'color': '#fef3c7'},
                                     {'range': [60, 100], 'color': '#fee2e2'}]}))
                gauge.update_layout(height=230, margin=dict(l=10, r=10, t=46, b=8))
                st.plotly_chart(gauge, use_container_width=True)
                if f["dets"]:
                    bar = go.Figure(go.Bar(x=[d[1] for d in f["dets"]][::-1],
                        y=[f"{d[0]} #{i+1}" for i, d in enumerate(f["dets"])][::-1], orientation="h",
                        marker_color=["#ef4444" if d[0] == "fire" else "#f59e0b" for d in f["dets"]][::-1]))
                    bar.update_layout(height=150, margin=dict(l=0, r=0, t=6, b=0),
                                      xaxis_range=[0, 1], xaxis_title="신뢰도", template="simple_white")
                    st.plotly_chart(bar, use_container_width=True)
            st.session_state.wf_focus += 1
        focus()

        with st.expander("🛡 Hybrid 2차 검증 — 이 프로젝트의 핵심 (본인 담당)"):
            st.markdown(
                "**문제**: 단독 YOLO 탐지기는 구름·안개·노을을 '연기'로 자주 오인합니다(오탐↑ → 경보 신뢰도↓).\n\n"
                "**해결(본인 설계)**: YOLO 후보 박스를 잘라 **ResNet50/DeepCNN 2차 분류기로 재검증**해 배경 오탐을 걸러내는 "
                "**Hybrid 구조**. 오탐 1,253장을 hard negative로 재학습, 2차 분류 정확도 **97.4%**. **+ SAHI** 타일 추론으로 먼 소형 연기 보완.\n\n"
                "> ⚙️ 본 라이브 데모는 경량 배포를 위해 **YOLO 1-stage + 시간축 안정화(최근 3프레임 다수결)** 로 오탐 억제를 재현합니다. "
                "(2차 ResNet50 가중치 282MB는 배포 제외)")

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
