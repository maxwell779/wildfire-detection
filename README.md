# 🔥 산불·연기 실시간 탐지 (YOLO + ResNet Hybrid)

> 감시 영상에서 산불·연기를 실시간 탐지하되, **YOLO 후보를 ResNet으로 2차 검증하는 Hybrid 구조**로 구름·안개 오탐을 줄인 딥러닝 객체탐지 프로젝트.

| 항목 | 내용 |
|---|---|
| 기간 | 2026.03.24 ~ 2026.04.02 |
| 팀 | 4인 (KDT 팀 프로젝트) |
| **나의 역할** | **딥러닝 모델링·Hybrid 구조 담당** — YOLO/분류기 학습, Hybrid 검증 설계, 평가 |

> ℹ️ KDT 부트캠프 팀 프로젝트입니다. 본 저장소는 **본인(딥러닝 모델링) 작업** 중심으로 정리한 것이며, 데이터 라벨링·수집 등은 팀 공동 작업이었습니다.

---

## 🎯 문제 정의
단독 객체탐지기는 구름·안개·노을을 연기로 오인(오탐↑)해 경보 신뢰도가 떨어진다 → **실시간성을 유지하면서 오탐을 줄이는 검증 구조**가 필요.

## 🛠 기술 스택
`Python` · `PyTorch` · `YOLOv8 (s/m)` · `RT-DETR` · `ResNet50 / DeepCNN` · `SAHI` · `W&B`

## 🔧 핵심 구현
1. **탐지 학습** — YOLOv8s(30ep, mAP50 0.71) → YOLOv8m(50ep, mixup 0.15·copy_paste 0.1, **mAP50 ≈ 0.78**), RT-DETR-S 비교.
2. **2차 분류기 4종** — DeepCNN_640 / ResNet50의 3-class·4-class (4-class: DeepCNN **97.43%**).
3. **Hybrid 파이프라인** — YOLO 후보 박스 → 크롭 → ResNet 재분류로 배경 오탐 제거.
4. **증강 분리 설계** — 분류용 / 탐지용(bbox 동시 변환) 분리, **train에만** 적용해 검증셋 오염 방지 (`scripts/`).
5. **SAHI 타일 추론** — 먼 산불 연기(소형 객체) 보완.

## 🔧 트러블슈팅
| 문제 | 해결 |
|---|---|
| 구름·안개 오탐 | ResNet 2차 검증 Hybrid + **오탐 1,253장 hard negative 재학습** |
| 소형 객체 미탐지 | SAHI 슬라이스 타일 추론 |
| 라벨 모호(연기·불 동시/부재) | **4-class**(none/smoke_only/fire_only/both) 신설 |
| 증강이 평가 왜곡 | 분류/탐지 증강 분리 + train-only |
| **Hybrid 효과 검증** | 5변형을 단독 YOLO와 정량 비교 → **단독 YOLO를 못 넘음을 데이터로 확인**("복잡한 구조가 항상 답은 아니다") |

## 📈 결과
- YOLOv8m/l **mAP50 ≈ 0.78**, 4-class 분류 **97.4%**, hard negative 1,253장.
- 전체 실험 비교: [`final_model_comparison.csv`](final_model_comparison.csv) (60행).

## 🖥 데모 (Streamlit)
**모델 실험 비교 대시보드**(실데이터 CSV 시각화) + 파이프라인 설명 + (선택)객체탐지 데모.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 구조
```
app.py                           # Streamlit 데모(실험 비교 대시보드)
wildfire_hybrid_modeling.ipynb   # 본인 작업: 탐지·분류·Hybrid 전 과정
final_model_comparison.csv       # 모델/조건별 실험 비교표
data.yaml                        # YOLO 데이터 설정
scripts/                         # 데이터 증강 패키지(팀 전달용으로 문서화)
requirements.txt
```
> 데이터셋·모델 가중치(.pt/.pth)·실행 산출물은 용량 관계로 제외했습니다. 학습 과정·결과는 노트북에 포함되어 있습니다.
