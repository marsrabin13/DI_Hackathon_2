# CityFix — Tel Aviv-Yafo demo

Turns an English citizen complaint into a **proposed department route**, an **acknowledgment draft**, and
**evidence for geographic investigation**. Badge: **Simulated Tel Aviv reports**.

Two pages (top navigation):
- **Report an issue** — resident chat. Submitting a complaint routes, saves and replies automatically
  (reference, status, department, acknowledgment).
- **Admin dashboard** — statistics, map & hotspots, review queue with corrections, weekly brief,
  model & data, portability check.

Demo only: no login, no attachments, no geocoding, no dispatch. Saves and drafts stay in a local SQLite file;
nothing is sent to a municipality. Department labels are illustrative, not verified municipal ownership.

## Run on another computer (Git Bash on Windows)

Requires Python 3.11+ (tested on 3.13).

```bash
cd "/c/path/to/cityfix"
python -m venv .venv
source .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # optional: add ANTHROPIC_API_KEY for live AI drafts
python ai_service.py                 # provider access check (never prints the key)
python smoke_check.py                # 40 focused checks, temporary DB
python -m streamlit run app.py       # http://localhost:8501
```

The model is already trained and the held-out test already evaluated. To reproduce from the supplied data:

```bash
python train.py                      # fit on gpt_train.csv, thresholds chosen on gpt_validation.csv
python train.py --evaluate-test      # one-time evaluation on gpt_tel_aviv_test.csv
```

`train.py` deletes old test results when it retrains; `--evaluate-test` refuses a second run for the same
model unless `--force` is given. If a required data file is missing, both scripts stop and name it — no
replacement data is generated.

## Data (supplied package — not regenerated)

All files live in `data/`, prefixed `gpt_`. Hashes were verified against `gpt_data_provenance.json`.

| File | Rows | Use |
| --- | --- | --- |
| `gpt_train.csv` | 254 | Training only: 240 synthetic complaints (60/category, 12 families × 5 variants) + 14 deduplicated NYC structured details. Roads 64, waste 62, lighting 66, noise 62. |
| `gpt_validation.csv` | 24 | Validation and threshold selection only (6/category, separately authored, synthetic). |
| `gpt_tel_aviv_test.csv` | 24 | Final test: 20 in-scope (5/category) + 4 policy cases with blank category. Never used for fitting or tuning. |
| `gpt_tel_aviv_demo.csv` | 90 | Map/dashboard fixtures only: planted 12 roads, 10 waste, 8 lighting + 60 scattered. Seed 42. |
| `gpt_data_provenance.json` | — | Sources, counts, generation method, seed, demo anchor `2026-10-29T12:00:00Z`, split/map checks, hashes. |
| `gpt_nyc_sample.csv`, `gpt_nyc_download_receipt.json`, `gpt_nyc_source_lineage.json` | 600 raw | Retained NYC snapshot and lineage for the 14 training details (provenance only). |

Rules applied: classifier input is `text` only; `category` is the target; `family_id`, `source` and
`source_key` are kept as metadata. Splits are used as supplied — never merged or re-split. Labels are the
supplied ones; no pseudo-labels. Policy cases are scored on `expected_status`, separately from classification.

Independent verification on integration: all file hashes match; schemas match the brief; counts match
provenance; zero family-ID and normalized-text overlap between train/validation/test; demo IDs unique, all
inside the demo bbox and within the 7 days ending at the anchor; DBSCAN (250 m, 5 samples) finds exactly
roads 12, waste 10, lighting 8 and no noise cluster.

`data_package/` holds the package's own README, verification report and `prepare_data.py`. That script was
not run here; it writes to `data_package/data/` by default, so it cannot overwrite the project's `data/`.

Map fixtures are imported as `status=open`, `routing_state=routed`, `category_source=seed_fixture`, keeping
the supplied report IDs, coordinates, categories, timestamps and `synthetic_demo` source. The stored demo
anchor is the default admin window end. If an older fixture file was imported before, those fixture rows
(and their corrections) are replaced automatically on the next start; citizen reports are kept.

## Model and policy

`TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)` →
`LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`, fitted on `gpt_train.csv` only.

Policy: urgent phrase guard → `urgent_review` (demo guard, **not** reliable emergency detection); no known
vocabulary, low score, low margin, missing department mapping or the *Multiple issues / unsure* option →
`review_required`; otherwise `routed` with the department from the city config.

**Threshold selection (validation only, once, then frozen):** at the starting 0.65 / 0.15 the model routed
**0/24** validation cases — Logistic Regression scores on this small 4-class set are compressed (top scores
mostly 0.3–0.6). The single adjustment chose the highest-coverage pair that kept 100% accepted-route accuracy
on validation: **min_score 0.40, min_margin 0.05**. Stored in `artifacts/model_metadata.json`. Scores are
uncalibrated estimates.

## Measured results (synthetic data — not proven Tel Aviv accuracy)

| Set | Accuracy | Macro-F1 | Routed | Routed correctly | Policy cases |
| --- | --- | --- | --- | --- | --- |
| Validation (24) | 0.792 | 0.765 | 15/24 | 15/15 | — |
| Held-out test (20 in-scope + 4) | 0.800 | 0.801 | 11/20 | 11/11 | 3/4 |

Per-class test F1: noise 1.00, waste 0.75, roads 0.73, lighting 0.73.

**Failures (kept visible, `artifacts/test_failures.csv`):**
- **Policy failure:** "There is a new pothole beside an overflowing rubbish bin, and both need attention" was
  **routed to waste** instead of going to review. Nothing detects multiple issues except the citizen's
  *Multiple issues / unsure* option. Not patched after seeing the test.
- 9 in-scope cases went to staff review instead of routing; 4 of them were also mis-predicted (missing step
  tread → lighting; garden trimmings → lighting; spilled refuse → roads; salt-coated lamp → roads).
- Validation shows the same weakness: lighting had 2/6 correct predictions.

Reading: whatever the model routes was correct on both sets, but roughly half of reports need a human. That's
expected with 254 mostly templated training texts.

**Smoke checks: 40/40 passed** — required files and provenance counts; split isolation; model fitted on
`gpt_train.csv` (hash); 6 nearby same-category points vs a distant point; category and city separation;
missing/outside location; duplicate save; idempotent fixture import; fixtures keep IDs, coordinates,
categories and timestamps; fixtures stored open + routed + seed_fixture; anchor = default window end; planted
hotspots 12/10/8 with no noise cluster; stale fixture replacement; correction audit with preserved prediction;
cross-city correction blocked; config-change route with unchanged model; labeled fallback on missing key and
API failure; rejection of deadline/dispatch claims, unknown evidence IDs and numbers in brief prose;
known/unclear/unsupported/multi-issue/urgent inputs.

**Live AI: not verified here.** No `ANTHROPIC_API_KEY` was available in the build environment, so drafts
produced during the build were **Template fallback**. Add a key to `.env`, run `python ai_service.py`, and
confirm the **Live AI draft** label in the app before claiming live GenAI.

## Files

| File | Role |
| --- | --- |
| `app.py` | Entry point: navigation between the two pages. |
| `citizen_page.py` | Resident chat page with automatic routing reply. |
| `admin_page.py` | Admin dashboard (demo, no login). |
| `ui_common.py` | Shared styling and cached model/DB (`st.cache_resource`). |
| `core.py` | Data paths, validation, inference, routing policy, SQLite (city-scoped), DBSCAN hotspots, facts. |
| `ai_service.py` | Anthropic SDK adapter: acknowledgment + weekly brief, validation, template fallback. |
| `train.py` | Training, validation threshold selection, one-time `--evaluate-test`. |
| `smoke_check.py` | Focused checks; results in `artifacts/smoke_results.json`. |
| `config/cities.json` | `il_tel_aviv` + `example_city` (portability: roads → Public Works). |
| `.streamlit/config.toml` | Navy/teal light theme. |
| `data/` | Supplied `gpt_` files. `cityfix.db` is created on first app run. |
| `data_package/` | Data package README, verification report and generator (not run). |
| `artifacts/` | Model, metadata, validation/test predictions, test metrics/failures, smoke results. |

## AI drafts

Generated automatically when a report is submitted (one per saved report) and by the **Generate weekly
brief** button; 12 s timeout, `max_retries=0`, bounded tokens, cached in session. The acknowledgment
receives only reference, routing state, department (routed only), category and whether a location was
given — never the complaint text. The brief receives only Python-computed facts with IDs. Output is
rejected (→ Template fallback) for wrong keys/types, >60 words, unknown evidence IDs, numbers in brief prose,
missing "demo"/"simulated" mention, or obvious deadline/dispatch/completion claims. Urgent cases always show
the static notice. Drafts never change routing, counts or the database.

## Demo script (≈3 min)

1. Report an issue: "There is a deep pothole on our street near the bus stop, cars keep swerving." + Demo
   Point A → automatic reply: reference, Routed, Road Maintenance (demo), acknowledgment.
2. "Please help, it is a problem." → staff review. "Exposed live wire near the school" → urgent notice.
3. Admin dashboard → Map & hotspots: three planted hotspots (roads 12, waste 10, lighting 8).
4. Review queue: correct the vague report to `waste` → department recalculated, audit row appears.
5. Weekly brief → observations and investigations citing fact IDs.
6. Model & data → Portability check: the same pothole text routes to **Public Works (demo)** in Example City,
   no retraining.
7. Model & data: show the measured numbers, the 0/24 → 15/24 threshold story, and the multi-issue failure.

## Pilot path (not done)

Collect authorized local English reports, agree department ownership with city staff, evaluate on a clean
local holdout, retrain/calibrate, then shadow-test beside staff. Pilot measures: triage time, reassignment
rate, review rate, accepted-route accuracy. No business improvement is claimed. Production would add tenant
access control, approved boundaries, case-system integration and monitored model releases. The buyer model
(annual licence + onboarding) is an unvalidated hypothesis.

## Tested versions

Python 3.13 · streamlit 1.66.1 · pandas 3.0.5 · numpy 2.5.3 · scikit-learn 1.9.1 · joblib 1.6.0 ·
pydeck 0.9.3 · anthropic 2.3.0 · python-dotenv 1.2.3 (pinned in `requirements.txt`, also recorded in
`artifacts/model_metadata.json`). If your scikit-learn version can't load `model.joblib`, run
`python train.py` then `python train.py --evaluate-test` to rebuild — results may differ slightly.
