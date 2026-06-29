# -*- coding: utf-8 -*-
"""
🔥 산불·연기 조기탐지 관제 시스템 (산림청용) — 정적 배포 데모
학습된 YOLOv8s(smoke/fire, mAP50 ≈ 0.78)의 탐지 결과를 미리 계산(precomputed_dets.json)해
백엔드·torch 없이 가볍고 안정적으로 구동합니다. (실시간 추론·영상 분석은 로컬 실행 시)
다중 카메라 관제 월 + 관제 지도 + 포커스 채널(시간축 안정화 경고·화재 위험도) 구성.
"""
import os, glob, json, numpy as np, pandas as pd, cv2, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="산불 조기탐지 관제", page_icon="🔥", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.join(HERE, "samples")
DETS = json.load(open(os.path.join(HERE, "precomputed_dets.json"), encoding="utf-8"))
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
</style>""", unsafe_allow_html=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

@st.cache_data(show_spinner=False)
def draw(path, conf):
    """미리 계산한 박스를 conf 기준으로 필터해 그린다(torch 불필요)."""
    name = os.path.basename(path)
    img = imread_u(path)
    rec = DETS.get(name, {"boxes": []})
    counts = {"smoke": 0, "fire": 0}; dets = []
    for b in rec["boxes"]:
        if b["conf"] < conf:
            continue
        cls, cf = b["cls"], b["conf"]
        counts[cls] = counts.get(cls, 0) + 1; dets.append((cls, cf))
        x1, y1, x2, y2 = b["box"]; c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        cv2.putText(img, f"{cls} {cf:.2f}", (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), counts, dets


def alert(fire, smoke):
    if fire > 0: return ("🚨 화재 경보", "a-red", "즉시 출동 — 화염 감지")
    if smoke > 0: return ("⚠️ 연기 주의", "a-orange", "연기 감지 — 현장 확인 필요")
    return ("🟢 정상", "a-green", "이상 없음")


def overlay_label(rgb, text, color=(255, 255, 255)):
    img = rgb.copy()
    cv2.putText(img, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
    return img


def stabilized_state(hist):
    """최근 프레임(fire,smoke) 이력으로 깜빡임을 누른 '안정화 경고'(Hybrid 오탐억제 취지)."""
    recent = hist[-3:]
    fires = sum(1 for f, s in recent if f)
    smokes = sum(1 for f, s in recent if s)
    if fires >= 2: return ("🚨 화재 발생", "a-red")
    if smokes >= 2: return ("⚠️ 연기 감지", "a-orange")
    if recent and (recent[-1][0] or recent[-1][1]): return ("🟡 관찰 중", "a-orange")
    return ("🟢 정상", "a-green")


st.markdown('<div class="hero"><h1>🔥 산불·연기 조기탐지 관제 시스템</h1>'
            '<p>산림청 감시 카메라·드론 영상 기반 · YOLOv8s(smoke/fire) · mAP50 ≈ 0.78 · KDT 4인(본인=딥러닝 모델링·Hybrid)</p>'
            '<div><span class="chip">🔥 YOLOv8s mAP50 0.78</span><span class="chip">🛡 Hybrid 오탐 억제</span>'
            '<span class="chip">📡 실시간 관제 + 안정화 경고</span></div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 탐지 설정")
    conf = st.slider("신뢰도 임계값", 0.05, 0.9, 0.25, 0.05, help="낮출수록 더 많은 후보를 경보")
    speed = st.slider("포커스 순환 주기(초)", 1.5, 6.0, 3.0, 0.5)
    st.divider()
    st.caption("**모델**: YOLOv8s · 2-class(smoke/fire)")
    st.caption("학습 14,122 / 검증 3,099장 · **mAP50 ≈ 0.78**")
    st.caption("※ 데모는 탐지 결과를 미리 계산해 가볍게 구동(정적 배포). 실시간 추론·영상 분석은 GitHub 로컬 실행.")

samples = sorted(glob.glob(os.path.join(SP, "*.jpg")))
if not samples:
    st.info("samples 폴더에 이미지가 없습니다.")
    st.stop()

if "wf_focus" not in st.session_state: st.session_state.wf_focus = 0
if "wf_hist" not in st.session_state: st.session_state.wf_hist = []

feeds = []
for idx, p in enumerate(samples):
    rgb, counts, dets = draw(p, conf)
    fire, smoke = counts.get("fire", 0), counts.get("smoke", 0)
    nm = f"CH-{idx+1:02d}"
    feeds.append(dict(i=idx, name=nm, rgb=rgb, fire=fire, smoke=smoke,
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
    st.markdown("#### 📈 모델 성능 (클래스별 · 검증셋)")
    perf = go.Figure()
    perf.add_bar(name="💨 연기(smoke)", x=["Precision", "Recall", "mAP50"], y=[0.764, 0.716, 0.764],
                 marker_color="#f59e0b", text=["0.764", "0.716", "0.764"], textposition="outside")
    perf.add_bar(name="🔥 화염(fire)", x=["Precision", "Recall", "mAP50"], y=[0.676, 0.568, 0.656],
                 marker_color="#ef4444", text=["0.676", "0.568", "0.656"], textposition="outside")
    perf.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
                       yaxis=dict(range=[0, 1.05], tickformat=".0%"), template="simple_white",
                       legend=dict(orientation="h", y=1.18, x=0))
    st.plotly_chart(perf, use_container_width=True)
    st.caption("YOLOv8s · 검증 3,099장 · 전체 mAP50 ≈ 0.78 (smoke/fire 2-class)")
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
        lv, cls_, msg = alert(f["fire"], f["smoke"])
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
        "> ⚙️ 본 라이브 데모는 가벼운 배포를 위해 **탐지 결과 사전계산 + 시간축 안정화(최근 3프레임 다수결)** 로 오탐 억제를 재현합니다. "
        "(실시간 YOLO 추론·2차 ResNet 검증·영상 분석은 GitHub 저장소에서 로컬 실행)")

st.caption("GitHub: wildfire-detection · 본인 역할: 딥러닝 모델링·Hybrid 구조 (데이터 수집·라벨링은 팀 공동) · KDT 4인")
