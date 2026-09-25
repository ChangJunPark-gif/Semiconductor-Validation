# 반도체 수율 프로젝트

> **후속 검증 저장소:** 이 저장소는 [원본 프로젝트](https://github.com/ChangJunPark-gif/Semiconductor)를 복사하여 평가 한계를 조사한 독립 저장소입니다. 원본 1–7단계와 기존 보고서는 아래에 보존했습니다. 새로운 결과는 [한계 검증 개요](LIMITATIONS.md), [입력 중복·lot 단위 불확실성](reports/validation/lot_bootstrap.md), [정상 wafer 포함 OOD 평가](reports/validation/ood_with_normal.md)를 먼저 보세요. 아래의 GitHub Pages 링크와 기존 성능 요약은 **원본 프로젝트**의 기록입니다.

웨이퍼 수율·공간 패턴 분석과 신규 패턴 탐지를 위한 프로젝트입니다.

**웹사이트:** [Wafer Signal Lab — GitHub Pages](https://changjunpark-gif.github.io/Semiconductor/)

구현 범위와 평가 기준은 [프로젝트 명세서](PROJECT_SPEC.md)를 참고하세요.

**결과부터 보기:** [포트폴리오 요약](PORTFOLIO_SUMMARY.md) · [공정 검토 사례 카드](reports/portfolio_case_cards.md) · [면접 발표 구성](PRESENTATION.md) · [진행 현황](PROJECT_PROGRESS.md)

결함 8종의 동일 lot 분할 테스트에서 수율만 사용한 모델의 macro-F1은 **0.318**, 공간 특징 모델은 **0.755**, CNN은 **0.734**였습니다. 보류 유형 OOD 탐지는 평균 AUROC **0.780**이었지만, 검증 세트 임계값에서 평균 탐지율은 **36.5%**였습니다. 이 결과를 실제 공정 원인 규명이나 수율 개선 효과로 해석하지 않습니다.

## 1단계 · 데이터 품질과 수율·bin 분포

원본 [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)를 내려받아 `data/raw/LSWMD.pkl`에 둡니다. 이 컴퓨터의 Codex Python 환경에서 Windows PowerShell로 다음을 실행합니다.

```powershell
$python = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements-step1.txt
.\.venv\Scripts\python.exe -c "import kagglehub; kagglehub.dataset_download('qingyi/wm811k-wafer-map', output_dir='data/raw')"
.\.venv\Scripts\python.exe -m src.data.step1_yield
```

분석 결과는 [1단계 보고서](reports/data_report.md)에 정리합니다. wafer별·lot별 CSV 원본은 `data/processed/`에 저장하며 Git에는 올리지 않습니다.

## 2단계 · wafer 공간 시각화

1단계 실행 후 다음 명령으로 패턴별 대표 wafer, 저수율 wafer, lot 내 wafer 순서 그림을 만듭니다.

```powershell
.\.venv\Scripts\python.exe -m src.data.step2_visualize
```

결과와 사례 선정 기준은 [2단계 보고서](reports/step2_visualization.md)에 기록합니다.

## 3단계 · Moran's I와 공간 자기상관

유효 die의 상하좌우 이웃을 기준으로 전체 wafer의 Moran's I를 계산하고, 라벨별로 선정한 사례에 조건부 permutation 검정을 적용합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.spatial.step3_moran
```

수치, 검정 범위와 해석상 주의점은 [3단계 보고서](reports/step3_moran.md)에 기록합니다. wafer별 전체 결과 CSV는 `data/processed/`에만 보관합니다.

## 4단계 · 공간 특징과 PCA

전체 wafer의 공간 특징을 추출하고 lot 단위로 train/validation/test를 나눕니다. 결측 대체·표준화·PCA는 train에만 적합합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step4.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.features.step4_features
```

실제 결과와 한계는 [4단계 보고서](reports/step4_features_pca.md), 각 변수의 정의는 [특징 사전](reports/feature_dictionary.md)에 있습니다. 전체 특징 CSV·PCA 임베딩·변환기는 `data/processed/`에 저장하며 Git에는 올리지 않습니다.

## 5단계 · DBSCAN과 HDBSCAN 군집

학습 lot의 wafer 12,000장을 고정 시드로 추출해 9차원 PCA 공간에서 밀도 기반 군집을 비교합니다. 설정별 군집 수·noise 비율·안정성, 기존 라벨 교차표, 대표 wafer map을 기록합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step5.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.clustering.step5_cluster
```

[5단계 보고서](reports/step5_clustering.md)에 실제 결과와 한계를 정리했습니다. 웨이퍼별 군집 할당 파일은 `data/processed/`에만 보관합니다.

## 6단계 · 결함 패턴 분류

결함 라벨 8종을 lot 분할을 유지한 채 CNN과 공간 특징 로지스틱 회귀로 분류합니다. 원본 map의 분할 간 완전 중복은 평가에서 제거합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step6.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.models.step6_classify
```

Macro-F1, 클래스별 precision/recall, confusion matrix, 오류 사례와 한계는 [6단계 보고서](reports/step6_classification.md)에 있습니다. 이미지 캐시와 모델 체크포인트는 `data/processed/`에만 보관합니다.

## 7단계 · 신규 패턴 OOD 탐지

결함 라벨 8종을 하나씩 학습에서 제외하고, 보류 유형의 test wafer를 신규 패턴으로 간주합니다. 학습 데이터의 클래스별 Mahalanobis 거리와 분류기의 최대 softmax 확률을 비교하고, 알려진 유형 validation 데이터만으로 경보 임계값을 정합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step7.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.evaluation.step7_ood
```

보류 유형별 AUROC·AUPR·FPR@95% TPR·고정 임계값 경보율과 한계는 [7단계 보고서](reports/step7_ood.md)에 있습니다. 특히 보류 유형 탐지율이 낮은 경우가 있어 실제 공정 경보로 바로 사용할 수 없습니다.

## 8단계 · 최종 결과와 포트폴리오

같은 분할에서 수율만 사용한 기준선을 평가하고, 공간 특징·CNN 결과와 비교합니다.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.step8_yield_baseline
```

[포트폴리오 요약](PORTFOLIO_SUMMARY.md), [사례 카드](reports/portfolio_case_cards.md), [발표 구성](PRESENTATION.md)에 관찰·한계·후속 공정 검증 항목을 정리했습니다.
