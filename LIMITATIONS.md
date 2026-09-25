# 웨이퍼 수율·패턴 연구의 한계 검증

이 저장소는 [원본 Semiconductor 프로젝트](https://github.com/ChangJunPark-gif/Semiconductor)를 복사해 **평가의 취약점을 검증하는 후속 연구**로 분리했다. 원본의 단계별 산출물은 보존하고, 새 실험과 결과는 `src/validation/` 및 `reports/validation/`에 둔다. 여기서 검토하는 수치는 WM-811K를 사용한 **이 프로젝트의 결과**이며, 참고 문헌의 논문 결과를 직접 재현하거나 평가한 것은 아니다.

## 검증 결과

| 원본 연구의 한계 | 이번 검증과 조치 | 관찰 결과 | 상태 |
| :--- | :--- | :--- | :--- |
| 원본 map만 분할 간 중복 제거 | CNN에 실제 입력되는 48×48 map의 hash도 확인하고 재학습 | 기존 validation/test 제거 15/23장 → 17/34장. CNN test macro-F1 0.7347, 공간 특징 기준선 0.7550 | **입력 완전 중복 해결**; 유사 중복은 남음 |
| 단일 test 점수에 불확실성 없음 | test lot 1,195개를 통째로 1,000회 재표본추출 | CNN macro-F1 95% 구간 0.696–0.761. 기준선 − CNN 차이 +0.020, 구간 −0.004–+0.054 | **관찰 lot 내 불확실성 측정**; 새 공정 일반화는 아님 |
| 신규 패턴 탐지에 정상 wafer가 빠짐 | `none`을 알려진 유형에 포함해 8개 leave-one-defect-out 실험을 반복 | 기본 Mahalanobis: 평균 AUROC 0.898, 정상 오경보 0.8%, **알려진 결함 오경보 28.6%**, 보류 유형 탐지 66.7% | **문제 확인**, 해결되지 않음 |
| 단일 임계값이 알려진 유형 간 점수 차이를 무시 | 유형별 최대 임계값과 유형별 거리 보정을 별도 검증 | 최대 임계값의 보류 탐지 0.0%; 거리 보정 0.7%. 오경보 감소와 함께 탐지도 붕괴 | **개선 실패를 공개**; 운영 경보기 아님 |
| 희귀 결함과 한 데이터셋에 의존 | lot 단위 구간·유형별 OOD 표본 수를 보고 | Near-full test 17장. 외부 fab·장비·시점 자료 없음 | **미해결** |

숫자와 평가 설계는 [입력 중복·lot 불확실성 보고서](reports/validation/lot_bootstrap.md), [정상 포함 OOD 보고서](reports/validation/ood_with_normal.md)에 있다. 특히 AUROC는 순위 성능이고, 선택한 임계값에서의 오경보율·탐지율을 대신하지 않는다. 정상 포함 OOD의 모집단은 원본 7단계와 다르므로 두 AUROC를 직접 개선 폭으로 해석하지 않는다.

## 다음 연구에서 해결할 과제

1. 같은 map의 작은 회전·이동·리사이즈 차이까지 탐지할 수 있는 **유사 중복 감사**를 하고, 감사 규칙을 test를 보기 전에 고정한다.
2. 정상과 결함의 비용을 정의하고, 별도 calibration lot에서 임계값을 결정한다. **정상 오경보·알려진 결함 오경보·신규 결함 탐지**를 모두 목표로 보고, 하나라도 실패하면 운영 적용을 보류한다.
3. Near-full 등 희귀 유형의 label 품질과 표본 수를 재검토한다. 부족한 유형은 별도 성능 보증을 하지 않는다.
4. 다른 제조 라인·장비·시간대와 공정 메타데이터를 확보해 외부 검증한다. WM-811K만으로는 결함 원인이나 실제 수율 개선 효과를 입증할 수 없다.

## 재현

원본 README의 1–4단계로 `data/raw/LSWMD.pkl`과 `data/processed/wafer_features.csv`를 만들고, 6단계의 `step6_labeled_maps.npz`를 준비한다. 다음 명령은 새 저장소 루트에서 실행한다. 큰 원본·중간 데이터와 모델 checkpoint는 Git에 포함하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.models.step6_classify --report-output reports/validation
.\.venv\Scripts\python.exe -m src.validation.lot_bootstrap
.\.venv\Scripts\python.exe -m src.validation.ood_with_normal
```

모델 검증용 주요 의존성은 `requirements-step7.txt`와 6단계 PyTorch 요구사항을 따른다. 새 코드는 시드 42, 원본 lot 분할을 유지한다. OOD 실험은 알려진 train/validation만으로 적합·보정하고, test의 보류 유형은 최종 평가에만 쓴다.
