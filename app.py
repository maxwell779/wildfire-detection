# -*- coding: utf-8 -*-
"""
🔥 산불·연기 실시간 탐지 — Streamlit 대시보드 데모
모델 실험 비교(실데이터 CSV)와 YOLO+ResNet Hybrid 파이프라인을 보여줍니다.
이미지를 올리면(선택) 범용 YOLO로 객체탐지 데모를 실행합니다.

※ 실제 산불 모델 가중치는 용량 관계로 제외. 대시보드는 실험 결과 CSV로 동작합니다.
"""
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="산불·연기 탐지", page_icon="🔥", layout="wide")

st.title("🔥 산불·연기 실시간 탐지 (YOLO + ResNet Hybrid)")
st.caption("YOLO 후보를 ResNet으로 2차 검증해 구름·안개 오탐을 줄인 딥러닝 객체탐지 · KDT 4인(본인=딥러닝 모델링·Hybrid)")

with st.expander("ℹ️ 이 데모에 대하여", expanded=False):
    st.markdown(
        "- 실제 성과: YOLOv8m **mAP50 ≈ 0.78**, 4-class 분류 **97.4%**, 오탐 1,253장 hard negative 수집.\n"
        "- 핵심 발견(정직): Hybrid 5변형이 단독 YOLO(0.78)를 못 넘음 → *\"복잡한 구조가 항상 답은 아니다\"*를 데이터로 확인.\n"
        "- 본 대시보드는 실험 결과 CSV로 동작하며, 모델 가중치는 제외되었습니다."
    )

st.subheader("🔧 파이프라인")
st.markdown("`영상 → YOLOv8 후보 탐지 → 크롭 → ResNet 2차 분류(배경 오탐 제거) → 최종 판정` + SAHI 타일 추론(소형 객체)")

st.subheader("📊 모델 실험 비교")
csv_path = "final_model_comparison.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True, height=320)
        # 수치 컬럼 자동 탐지해 막대그래프
        num_cols = [c for c in df.columns if df[c].dtype != object]
        label_col = df.columns[0]
        metric = st.selectbox("지표 선택", num_cols, index=0 if num_cols else None) if num_cols else None
        if metric:
            chart = df[[label_col, metric]].dropna().set_index(label_col)
            st.bar_chart(chart)
    except Exception as e:
        st.warning(f"CSV 표시 중 오류: {e}")
else:
    st.info("final_model_comparison.csv 가 있으면 실험 비교표·그래프가 표시됩니다.")

st.divider()
st.subheader("🖼 (선택) 객체탐지 데모")
st.caption("※ 산불 전용 가중치가 아닌 범용 YOLO 데모입니다. 실제 모델은 산불·연기 4-class로 학습됨.")
up = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
if up is not None:
    from PIL import Image
    img = Image.open(up).convert("RGB")
    st.image(img, caption="입력 이미지", use_container_width=True)
    try:
        from ultralytics import YOLO
        with st.spinner("범용 YOLO 추론 중..."):
            model = YOLO("yolov8n.pt")  # 최초 1회 자동 다운로드
            res = model(img)
            st.image(res[0].plot()[:, :, ::-1], caption="범용 YOLO 탐지(데모)", use_container_width=True)
    except Exception:
        st.info("실제 탐지 데모를 보려면 `pip install ultralytics` 후 다시 실행하세요. (산불 전용 모델은 별도)")

st.divider()
st.caption("GitHub: wildfire-detection · 본인 역할: 딥러닝 모델링·Hybrid 구조 (데이터 라벨링·수집은 팀 공동)")
