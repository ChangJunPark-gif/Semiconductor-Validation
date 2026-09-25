"""Measure classification uncertainty by resampling whole test lots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.step6_classify import LABELS


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def bootstrap_by_lot(predictions: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    """Resample lot IDs with replacement, then include all their test wafers."""
    groups = [part.index.to_numpy() for _, part in predictions.groupby("lot_id", sort=True)]
    if len(groups) < 2:
        raise ValueError("At least two test lots are required")
    rng = np.random.default_rng(seed)
    truth = predictions.true_label.to_numpy()
    cnn = predictions.cnn_prediction.to_numpy()
    baseline = predictions.baseline_prediction.to_numpy()
    records = []
    for repeat in range(repeats):
        sample = np.concatenate([groups[index] for index in rng.integers(0, len(groups), size=len(groups))])
        cnn_score = macro_f1(truth[sample], cnn[sample])
        baseline_score = macro_f1(truth[sample], baseline[sample])
        records.append({"repeat": repeat, "cnn_macro_f1": cnn_score,
                        "baseline_macro_f1": baseline_score,
                        "baseline_minus_cnn": baseline_score - cnn_score,
                        "near_full_present": bool(np.any(truth[sample] == "Near-full"))})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("reports/validation/step6_test_predictions.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/validation/step6_metrics.json"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/validation"))
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    args.report_output.mkdir(parents=True, exist_ok=True)
    lookup = pd.read_csv(args.features, usecols=["source_row", "lot_id", "failure_type", "split"])
    predictions = pd.read_csv(args.predictions).merge(lookup, on="source_row", validate="one_to_one")
    predictions = predictions.reset_index(drop=True)
    if not (predictions.split == "test").all() or not (predictions.true_label == predictions.failure_type).all():
        raise ValueError("Prediction labels do not match the frozen test set")
    bootstrap = bootstrap_by_lot(predictions, args.repeats, seed=42)
    bootstrap.to_csv(args.report_output / "lot_bootstrap_samples.csv", index=False)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    point = {"CNN": metrics["CNN"]["macro_f1"],
             "spatial baseline": metrics["logistic"]["macro_f1"]}
    difference = point["spatial baseline"] - point["CNN"]
    columns = {"CNN": "cnn_macro_f1", "spatial baseline": "baseline_macro_f1",
               "baseline - CNN": "baseline_minus_cnn"}
    intervals = {name: np.quantile(bootstrap[column], [0.025, 0.975]) for name, column in columns.items()}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(bootstrap.cnn_macro_f1, bins=30, alpha=0.55, label="CNN", color="#42b69a")
    ax.hist(bootstrap.baseline_macro_f1, bins=30, alpha=0.55, label="Spatial baseline", color="#277da1")
    ax.set(xlabel="Test macro-F1 (lot bootstrap)", ylabel="Replicates", title="Whole-lot resampling uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.report_output / "lot_bootstrap.png", dpi=150)
    plt.close(fig)
    report = ["# 모델 입력 중복과 lot 단위 성능 불확실성", "",
              "## 무엇을 고쳤나", "",
              "원본 연구는 원본 map SHA-256으로 분할 간 완전 중복을 제거했지만, CNN이 실제 보는 **48×48 최근접 리사이즈 입력**의 충돌은 확인하지 않았다. 새 평가에서는 원본 hash와 변환 입력 hash를 모두 비교해 train/validation/test 간 중복 입력을 제거하고 분류 모델을 다시 학습했다.", "",
              "| 평가 | validation 제거 | test 제거 | test 결함 wafer |",
              "| :--- | ---: | ---: | ---: |",
              "| 원본 map hash만 | 15 | 23 | 3,637 |",
              "| 원본+48×48 입력 hash | 17 | 34 | 3,626 |", "",
              "추가 제거 2장/11장은 서로 다른 원본 map이 모델 입력에서는 같아지는 사례다. 전체 데이터나 다른 리사이즈 크기의 유사 중복까지 제거했다는 뜻은 아니다.", "",
              "## 같은 설정으로 재학습한 test 결과", "",
              "| 모델 | 원본 연구 macro-F1 | 입력 중복 제거 후 macro-F1 | lot bootstrap 95% 구간 |",
              "| :--- | ---: | ---: | ---: |",
              f"| CNN | 0.734 | {point['CNN']:.3f} | {intervals['CNN'][0]:.3f}–{intervals['CNN'][1]:.3f} |",
              f"| 공간 특징 기준선 | 0.755 | {point['spatial baseline']:.3f} | {intervals['spatial baseline'][0]:.3f}–{intervals['spatial baseline'][1]:.3f} |", "",
              f"같은 test lot을 재표본추출한 **기준선 − CNN macro-F1** 차이는 {difference:+.3f}, 95% 백분위 구간은 **{intervals['baseline - CNN'][0]:+.3f}–{intervals['baseline - CNN'][1]:+.3f}**이다. Bootstrap {args.repeats:,}회, 시드 42, test lot {predictions.lot_id.nunique():,}개. `Near-full`이 빠진 재표본 비율은 {(~bootstrap.near_full_present).mean():.1%}다.", "",
              "![lot 재표본 성능 분포](lot_bootstrap.png)", "",
              "[재표본별 수치](lot_bootstrap_samples.csv) · [재학습 세부 지표](step6_metrics.json)", "",
              "## 해석 경계", "",
              "- 이 구간은 **관찰된 test lot을 재표본추출**한 불확실성이다. 새 장비·공정·시점으로의 일반화 구간은 아니다.",
              "- 같은 분할 안의 유사 중복은 남아 있고, 48×48 hash는 완전 동일 입력만 잡는다. 구조적으로 비슷하지만 픽셀이 조금 다른 wafer는 별도 검사해야 한다.",
              "- CNN epoch는 validation macro-F1로 선택했고 test는 최종 평가에만 사용했다. 입력 중복 제거 후 모델을 다시 학습했으므로 단순히 test 행만 삭제한 결과가 아니다."]
    (args.report_output / "lot_bootstrap.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"lots": predictions.lot_id.nunique(), "test": len(predictions),
           "point": point, "difference_ci": intervals["baseline - CNN"].tolist()})


if __name__ == "__main__":
    main()
