"""Repeat leave-one-defect-class-out OOD evaluation with normal wafers known."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.step1_yield import load_legacy_pickle
from src.evaluation.step7_ood import class_mahalanobis, score_metrics
from src.features.step4_features import PCA_FEATURES
from src.models.step6_classify import LABELS, cross_split_keep, map_digest


FEATURES = ["fail_rate", *PCA_FEATURES]


def class_scaled_scores(train: np.ndarray, train_labels: np.ndarray,
                        validation: np.ndarray, validation_labels: np.ndarray,
                        test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each class-center distance by its own known-validation 95th percentile."""
    classes = np.unique(train_labels)
    centers = np.vstack([train[train_labels == label].mean(axis=0) for label in classes])
    residuals = train - centers[np.searchsorted(classes, train_labels)]
    covariance = LedoitWolf().fit(residuals).covariance_
    precision = np.linalg.inv(covariance + 0.01 * np.eye(covariance.shape[0]))

    def distances(query: np.ndarray) -> np.ndarray:
        difference = query[:, None, :] - centers[None, :, :]
        return np.einsum("ncd,df,ncf->nc", difference, precision, difference)

    val_distances = distances(validation)
    radii = np.array([np.quantile(val_distances[validation_labels == label, index], 0.95)
                      for index, label in enumerate(classes)])
    if np.any(radii <= 0):
        raise ValueError("Nonpositive class radius")
    return (np.min(val_distances / radii, axis=1),
            np.min(distances(test) / radii, axis=1))


def load_or_create_hashes(frame: pd.DataFrame, raw_path: Path, defect_cache: Path, cache_path: Path) -> np.ndarray:
    """Hash every known normal and defect original map, preserving source-row order."""
    source_rows = frame.source_row.to_numpy()
    if cache_path.exists():
        saved = np.load(cache_path)
        if not np.array_equal(saved["source_rows"], source_rows):
            raise ValueError("Normal-map hash cache is not aligned with features")
        return saved["hashes"]
    saved_defect = np.load(defect_cache)
    defect_hashes = {int(row): digest for row, digest in zip(saved_defect["source_rows"], saved_defect["hashes"])}
    original = load_legacy_pickle(raw_path)
    hashes = np.empty(len(frame), dtype="S64")
    for index, (source_row, label) in enumerate(zip(source_rows, frame.failure_type)):
        hashes[index] = (map_digest(original.iloc[int(source_row)].waferMap) if label == "none"
                         else defect_hashes[int(source_row)]).encode("ascii")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, source_rows=source_rows, hashes=hashes)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--defect-cache", type=Path, default=Path("data/processed/step6_labeled_maps.npz"))
    parser.add_argument("--hash-cache", type=Path, default=Path("data/processed/validation_known_hashes.npz"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/validation"))
    args = parser.parse_args()
    args.report_output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.features, usecols=["source_row", "lot_id", "failure_type", "split", *FEATURES])
    frame = frame[frame.failure_type.isin([*LABELS, "none"])].reset_index(drop=True)
    hashes = load_or_create_hashes(frame, args.raw, args.defect_cache, args.hash_cache)
    keep, removed = cross_split_keep(hashes, frame.split.to_numpy())
    frame = frame.loc[keep].reset_index(drop=True)
    labels = frame.failure_type.to_numpy()
    split = frame.split.to_numpy()
    train, validation, test = (split == name for name in ("train", "validation", "test"))
    if any(set(frame.lot_id[split == a]) & set(frame.lot_id[split == b])
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Lot leakage across splits")

    rows = []
    for heldout in LABELS:
        known = labels != heldout
        fit_mask = train & known
        validation_mask = validation & known
        truth = labels[test] == heldout
        normal_mask = labels[test] == "none"
        known_defect_mask = (labels[test] != heldout) & ~normal_mask
        if not truth.any() or not normal_mask.any():
            raise ValueError(f"Missing test groups for {heldout}")
        model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scaler", StandardScaler()),
                          ("classifier", LogisticRegression(class_weight="balanced", solver="newton-cholesky",
                                                             max_iter=500, random_state=42))])
        model.fit(frame.loc[fit_mask, FEATURES], labels[fit_mask])
        preprocessor = model[:-1]
        train_x = preprocessor.transform(frame.loc[fit_mask, FEATURES])
        val_x = preprocessor.transform(frame.loc[validation_mask, FEATURES])
        test_x = preprocessor.transform(frame.loc[test, FEATURES])
        val_distance = class_mahalanobis(train_x, labels[fit_mask], val_x)
        test_distance = class_mahalanobis(train_x, labels[fit_mask], test_x)
        val_scaled, test_scaled = class_scaled_scores(train_x, labels[fit_mask], val_x,
                                                     labels[validation_mask], test_x)
        val_msp = 1 - model.predict_proba(frame.loc[validation_mask, FEATURES]).max(axis=1)
        test_msp = 1 - model.predict_proba(frame.loc[test, FEATURES]).max(axis=1)
        for method, val_scores, test_scores in (("mahalanobis", val_distance, test_distance),
                                                 ("one_minus_max_softmax", val_msp, test_msp),
                                                 ("class_scaled_mahalanobis", val_scaled, test_scaled)):
            result = score_metrics(val_scores, test_scores, truth)
            class_thresholds = [np.quantile(val_scores[labels[validation_mask] == known_class], 0.95)
                                for known_class in np.unique(labels[validation_mask])]
            policies = (("class_radius_95", 1.0),) if method == "class_scaled_mahalanobis" else (
                ("pooled", result["validation_threshold"]),
                ("per_class_guard", float(max(class_thresholds))))
            for policy, threshold in policies:
                alarm = test_scores > threshold
                rows.append({"heldout_class": heldout, "method": method, "threshold_policy": policy,
                             "known_train": int(fit_mask.sum()), "known_validation": int(validation_mask.sum()),
                             "known_test": int((~truth).sum()), "normal_test": int(normal_mask.sum()),
                             "heldout_test": int(truth.sum()),
                             "normal_false_alarm_rate": float(alarm[normal_mask].mean()),
                             "known_defect_false_alarm_rate": float(alarm[known_defect_mask].mean()),
                             **result, "validation_threshold": threshold,
                             "test_known_false_alarm_rate": float(alarm[~truth].mean()),
                             "test_heldout_detection_rate": float(alarm[truth].mean())})
        scaled = rows[-1]
        print(f"{heldout}: scaled AUROC {scaled['auroc']:.3f}; normal FPR {scaled['normal_false_alarm_rate']:.1%}; known-defect FPR {scaled['known_defect_false_alarm_rate']:.1%}; held-out TPR {scaled['test_heldout_detection_rate']:.1%}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.report_output / "ood_with_normal.csv", index=False)
    means = results.groupby(["method", "threshold_policy"])[["auroc", "aupr", "fpr_at_95_tpr", "normal_false_alarm_rate",
                                                                    "known_defect_false_alarm_rate", "test_heldout_detection_rate"]].mean()
    selected = results[results.method == "class_scaled_mahalanobis"].set_index("heldout_class").loc[LABELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(LABELS))
    ax.bar(positions - 0.2, selected.normal_false_alarm_rate, width=0.4, label="Normal false alarm")
    ax.bar(positions + 0.2, selected.test_heldout_detection_rate, width=0.4, label="Held-out detection")
    ax.set(xticks=positions, xticklabels=LABELS, ylabel="Rate", ylim=(0, 1),
           title="Normal-aware OOD: fixed validation threshold")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.report_output / "ood_with_normal_rates.png", dpi=150)
    plt.close(fig)
    report = ["# 정상 wafer를 포함한 보류 유형 OOD 재평가", "",
              "기존 7단계는 `none`을 제외한 결함 라벨끼리만 알려진 유형/보류 유형을 비교했다. 이 실험은 `none`을 **알려진 정상 유형**으로 train·validation·test에 포함해 정상 wafer 오경보를 따로 측정한다. `unlabeled`는 정답이 없어 제외한다.", "",
              "## 평가 설계", "",
              "- 각 결함 유형을 하나씩 학습에서 빼고 나머지 7종과 `none`으로 train에 적합했다. 결측 대체·표준화·로지스틱 회귀 및 Mahalanobis 중심·공분산은 train만 사용한다.",
              "- 원래 거리의 `pooled` 임계값은 알려진 유형 validation 전체의 95백분위수다. `per_class_guard`는 유형별 95백분위수 중 최댓값이다. 추가한 `class_scaled_mahalanobis`는 각 중심까지의 거리를 해당 유형 validation의 95백분위 거리로 나누고 최솟값을 취한다. 임계값 1은 validation에서 정했다. 보류 유형 validation과 test는 설정에 쓰지 않았다.",
              f"- 원본 map SHA-256으로 분할 간 완전 중복을 제거했다. Validation {removed['validation']:,}장, test {removed['test']:,}장 제거. 동일 lot은 분할 간 겹치지 않는다.",
              "- 평가에는 수율에 대응하는 fail rate와 공간 특징 11개를 사용한다. 기본 거리·확신도는 7단계 정의를 유지했고 유형별 거리 보정을 새로 시험했다.", "",
              "## 결과", "",
              "| 방법 | 임계값 정책 | 평균 AUROC | 평균 AUPR | 평균 FPR@95% TPR | 정상 오경보율 | 알려진 결함 오경보율 | 보류 유형 탐지율 |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
              *[f"| {method} | {policy} | {row.auroc:.3f} | {row.aupr:.3f} | {row.fpr_at_95_tpr:.3f} | {row.normal_false_alarm_rate:.1%} | {row.known_defect_false_alarm_rate:.1%} | {row.test_heldout_detection_rate:.1%} |" for (method, policy), row in means.iterrows()],
              "", "**해석:** 기본 Mahalanobis의 pooled 임계값은 보류 유형을 비교적 많이 잡지만 알려진 결함 오경보가 높다. 유형별 최대 임계값과 거리 보정은 오경보를 거의 없애는 대신 보류 유형도 거의 탐지하지 못했다. 따라서 이 자료에서 안전한 신규 패턴 경보기를 확보했다고 볼 수 없다. AUROC와 임계값 적용 탐지율은 서로 다른 질문에 답한다.",
              "", "다음 표는 유형별 거리 보정 결과다.", "",
              "| 보류 유형 | 정상 test 장수 | 보류 test 장수 | 보정 거리 AUROC | 정상 오경보율 | 보류 탐지율 |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |",
              *[f"| {name} | {int(selected.loc[name, 'normal_test']):,} | {int(selected.loc[name, 'heldout_test']):,} | {selected.loc[name, 'auroc']:.3f} | {selected.loc[name, 'normal_false_alarm_rate']:.1%} | {selected.loc[name, 'test_heldout_detection_rate']:.1%} |" for name in LABELS],
              "", "![정상 오경보와 보류 유형 탐지율](ood_with_normal_rates.png)", "",
              "자세한 임계값·표본 수·세 방법 지표는 [CSV](ood_with_normal.csv)에 있다. 기존 7단계와는 알려진 유형 구성·중복 제거 대상이 달라 숫자를 직접 같은 모집단의 전후 개선으로 해석하면 안 된다.", "",
              "## 남는 한계", "",
              "- `none`은 공개 데이터의 라벨이지 실제 제조 현장의 정상 판정 절차를 대체하지 않는다. 미라벨 wafer, 외부 라인·장비·시점의 오경보는 검증하지 못했다.",
              "- 유형별 거리 보정과 guard 모두 test에서 유형별 오경보 5%를 보장하지 않는다. 희귀 유형의 백분위 추정은 불안정하다. 두 임계값 정책이 보류 탐지를 없애는 경우도 결과에 그대로 표시했다.",
              "- 원본 map의 완전 중복은 제거했지만 유사 중복은 남는다. 특정 보류 유형의 test 수가 적어 불확실성이 크다.",
              "- OOD 점수는 원인 판정이 아니며 공정·장비 이력과 결합한 엔지니어 검토가 필요하다."]
    (args.report_output / "ood_with_normal.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(means.to_string())


if __name__ == "__main__":
    main()
