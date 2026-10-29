"""Train and validate the CityFix classifier; optionally evaluate the held-out test once.

  python train.py                  # fit on data/gpt_train.csv, select thresholds on data/gpt_validation.csv
  python train.py --evaluate-test  # one-time evaluation on data/gpt_tel_aviv_test.csv (no tuning)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

import core

ROOT = Path(__file__).parent
DATA, ART = ROOT / "data", ROOT / "artifacts"
CITY = "il_tel_aviv"
TFIDF = dict(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
LOGREG = dict(max_iter=1000, class_weight="balanced", random_state=42)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def policy_eval(df: pd.DataFrame, model, cfg, meta) -> tuple[dict, pd.DataFrame]:
    rows = []
    for r in df.itertuples(index=False):
        res = core.route(r.text, cfg, model, meta)
        rows.append({"text": r.text, "true_category": r.category or "", "predicted_category": res["predicted_category"],
                     "score": round(res["score"], 3), "margin": round(res["margin"], 3),
                     "routing_state": res["routing_state"], "reasons": ";".join(res["reasons"]),
                     "expected_status": getattr(r, "expected_status", "routed")})
    out = pd.DataFrame(rows)
    inscope = out[out["true_category"] != ""]
    routed = inscope[inscope["routing_state"] == "routed"]
    m = {
        "n_in_scope": int(len(inscope)),
        "accuracy": round(accuracy_score(inscope["true_category"], inscope["predicted_category"]), 4),
        "macro_f1": round(f1_score(inscope["true_category"], inscope["predicted_category"], average="macro"), 4),
        "per_class": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                      for k, v in classification_report(inscope["true_category"], inscope["predicted_category"],
                                                        output_dict=True, zero_division=0).items()
                      if k in core.load_cities()[CITY]["departments"]},
        "routing_coverage": f"{len(routed)}/{len(inscope)}",
        "routing_coverage_rate": round(len(routed) / max(len(inscope), 1), 4),
        "accepted_route_accuracy": f"{int((routed['true_category'] == routed['predicted_category']).sum())}/{len(routed)}",
        "accepted_route_accuracy_rate": round(float((routed["true_category"] == routed["predicted_category"]).mean()), 4)
        if len(routed) else None,
    }
    return m, out


REQUIRED = [core.TRAIN_CSV, core.VALIDATION_CSV, core.TEST_CSV, core.DEMO_CSV, core.PROVENANCE_PATH]
SUPERVISED = ["text", "category", "family_id", "source", "source_key"]


def check_files():
    missing = [f"data/{p.name}" for p in REQUIRED if not p.exists()]
    if missing:
        raise SystemExit("Missing required data file(s): " + ", ".join(missing) + ". No replacement data is generated.")


def read_split(path, extra=()):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    want = SUPERVISED + list(extra)
    if list(df.columns) != want:
        raise SystemExit(f"{path.name}: expected columns {want}, found {list(df.columns)}")
    return df


def train():
    check_files()
    cfg = core.load_cities()[CITY]
    train_df = read_split(core.TRAIN_CSV)
    val_df = read_split(core.VALIDATION_CSV)
    for stale in ("test_metrics.json", "test_failures.csv", "test_predictions.csv"):
        (ART / stale).unlink(missing_ok=True)  # a new model needs a new one-time test evaluation
    model = Pipeline([("tfidf", TfidfVectorizer(**TFIDF)), ("clf", LogisticRegression(**LOGREG))])
    model.fit(train_df["text"], train_df["category"])  # text only

    data_hash = sha(core.TRAIN_CSV)
    meta = {"model_version": f"tfidf-lr-{data_hash[:8]}", "data_hash": data_hash,
            "thresholds": {"min_score": cfg["min_score"], "min_margin": cfg["min_margin"]}}

    # validation at starting thresholds
    v0, _ = policy_eval(val_df, model, cfg, meta)
    decision = "kept starting thresholds (0.65 / 0.15)"
    # One adjustment allowed, validation only: if coverage < 75%, pick the highest-coverage
    # pair whose accepted-route accuracy on validation is still 100%.
    if v0["routing_coverage_rate"] < 0.75:
        best = None
        for s in np.arange(0.30, 0.66, 0.05):
            for mg in np.arange(0.05, 0.16, 0.05):
                trial = {**meta, "thresholds": {"min_score": round(float(s), 2), "min_margin": round(float(mg), 2)}}
                vm, _ = policy_eval(val_df, model, cfg, trial)
                if vm["accepted_route_accuracy_rate"] == 1.0:
                    key = (vm["routing_coverage_rate"], s, mg)
                    if best is None or key > best[0]:
                        best = (key, trial["thresholds"])
        if best and best[0][0] > v0["routing_coverage_rate"]:
            meta["thresholds"] = best[1]
            decision = (f"adjusted once on validation: coverage at 0.65/0.15 was {v0['routing_coverage']}; "
                        f"chose highest coverage with 100% accepted-route accuracy on validation")
    v1, v_rows = policy_eval(val_df, model, cfg, meta)

    import streamlit, pydeck, anthropic, dotenv  # noqa: F401  (versions only)
    from importlib.metadata import version
    meta.update({
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": {"tfidf": {**TFIDF, "ngram_range": list(TFIDF["ngram_range"])}, "logreg": LOGREG},
        "classes": list(model.classes_),
        "train_counts": {"total": len(train_df), "by_source": train_df["source"].value_counts().to_dict(),
                         "by_category": train_df["category"].value_counts().to_dict(),
                         "families": int(train_df["family_id"].nunique())},
        "validation_counts": {"total": len(val_df)},
        "file_hashes": {core.TRAIN_CSV.name: data_hash, core.VALIDATION_CSV.name: sha(core.VALIDATION_CSV)},
        "data_files": {"train": core.TRAIN_CSV.name, "validation": core.VALIDATION_CSV.name,
                       "test": core.TEST_CSV.name, "demo": core.DEMO_CSV.name, "provenance": core.PROVENANCE_PATH.name},
        "threshold_decision": decision,
        "validation_at_start_thresholds": v0,
        "validation": v1,
        "versions": {"python": platform.python_version(), **{p: version(p) for p in
                     ["scikit-learn", "pandas", "numpy", "joblib", "streamlit", "pydeck", "anthropic", "python-dotenv"]}},
        "note": "Model score is an uncalibrated estimate, not a reliability guarantee. "
                "Synthetic validation shares generation bias with training.",
        "test_evaluated": False,
    })
    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    v_rows.to_csv(ART / "validation_predictions.csv", index=False)
    print(json.dumps({"threshold_decision": decision, "thresholds": meta["thresholds"],
                      "validation_start": {k: v0[k] for k in ("accuracy", "macro_f1", "routing_coverage", "accepted_route_accuracy")},
                      "validation_final": {k: v1[k] for k in ("accuracy", "macro_f1", "routing_coverage", "accepted_route_accuracy")}},
                     indent=2))


def evaluate_test(force: bool = False):
    meta = json.loads((ART / "model_metadata.json").read_text())
    if meta.get("test_evaluated") and not force:
        raise SystemExit("Held-out test already evaluated for this model; see artifacts/test_metrics.json "
                         "(use --force only if you retrained).")
    cfg = core.load_cities()[CITY]
    model, meta = core.load_model()
    check_files()
    test_df = read_split(core.TEST_CSV, extra=("expected_status",))
    m, rows = policy_eval(test_df, model, cfg, meta)
    rows["status_ok"] = rows["routing_state"] == rows["expected_status"]
    rows["category_ok"] = (rows["true_category"] == "") | (rows["true_category"] == rows["predicted_category"])
    policy = rows[rows["true_category"] == ""][["text", "expected_status", "routing_state", "predicted_category",
                                               "score", "margin", "reasons", "status_ok"]]
    m["policy_cases"] = policy.to_dict(orient="records")
    m["policy_cases_passed"] = f"{int(policy['status_ok'].sum())}/{len(policy)}"
    m["status_match_all_24"] = f"{int(rows['status_ok'].sum())}/{len(rows)}"
    m["thresholds_used"] = meta["thresholds"]
    m["model_version"] = meta["model_version"]
    m["evaluated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m["note"] = "Evaluated once; no tuning on test. Synthetic test - does not establish real-city accuracy."
    (ART / "test_metrics.json").write_text(json.dumps(m, indent=2))
    rows[~(rows["status_ok"] & rows["category_ok"])].to_csv(ART / "test_failures.csv", index=False)
    rows.to_csv(ART / "test_predictions.csv", index=False)
    meta["test_evaluated"] = True
    meta["test_file_hash"] = sha(core.TEST_CSV)
    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "routing_coverage", "accepted_route_accuracy",
                                         "policy_cases_passed", "status_match_all_24")}, indent=2))
    print(policy.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    evaluate_test(a.force) if a.evaluate_test else train()
