# CityFix data package

Data preparation only. No application, classifier fitting, model predictions, or measured accuracy is included.

## Delivered data

| File | Actual contents |
| --- | --- |
| `data/gpt_train.csv` | **254 rows**: 14 useful deduplicated real NYC structured details + 240 synthetic English narratives. Roads 64, waste 62, lighting 66, noise 62. |
| `data/gpt_validation.csv` | **24** separately authored synthetic cases: six per category. |
| `data/gpt_tel_aviv_test.csv` | **24** separately authored synthetic cases: five per category + four policy cases with blank category. |
| `data/gpt_tel_aviv_demo.csv` | **90** synthetic historical map reports: roads 27, waste 25, lighting 23, noise 15. Never use for classifier training/evaluation. |
| `data/gpt_nyc_sample.csv` | **600 actual API rows**, 150 per canonical category, restricted to the four requested fields. |
| `data/gpt_nyc_source_lineage.json` | Actual source keys and collapsed group membership for the 14 retained details. |
| `data/gpt_nyc_download_receipt.json` | Actual request results/timings and SHA-256 of the retained raw CSV. |
| `data/gpt_data_provenance.json` | Mode, download results, labels, actual counts, seed, file hashes, split checks, map checks and fixture membership. |

The bounded NYC download succeeded in **24.271 seconds**. Four URL-encoded queries returned 150 rows each, with 10-second request/socket limits, a 180-second overall download cutoff and no retries. Endpoint: <https://data.cityofnewyork.us/resource/erm2-nwe9.json>.

Source labels follow the supplied brief exactly: Street Condition → roads; Dirty Condition → waste; Street Light Condition → lighting; Noise - Residential / Noise - Street/Sidewalk → noise. These are structured problem details, not original citizen narratives.

Repeated primary descriptors are collapsed even when the secondary location type differs. Location-type metadata is removed from training text. Ambiguous/generic details, conflicting labels, unreviewed detail families and families reserved for evaluation are excluded from training; all downloaded rows remain in the raw sample. A conservative reviewed descriptor map in the script links overlapping source/training scenarios to the same family. The result is **14 useful details**, rather than treating hundreds of repeated source rows as independent examples. Novel live descriptors need deliberate curation before being admitted.

The 240 generated narratives were authored independently of NYC details, so they are all `synthetic`, with blank `source_key`. There are **zero `synthetic_from_nyc` rows**. There are 12 synthetic training families per category, five variants each. Real and generated members of a known common scenario share a training family. Labels come from authored scenario membership or the explicit source-label mapping, never classifier predictions.

## Reproduce without an API

Python **3.10+**, standard library only. Extract the ZIP and run from its root.

```bash
# Recreate all synthetic files offline in a new directory (240 training rows).
python prepare_data.py --output-dir synthetic_data

# Recreate the exact delivered CSVs from the retained actual NYC snapshot.
python prepare_data.py --nyc-csv data/gpt_nyc_sample.csv --nyc-download-log data/gpt_nyc_download_receipt.json --output-dir replay_data

# Optional fresh bounded download; the live NYC sample can change.
python prepare_data.py --download-nyc --output-dir refreshed_data

# Or use an uploaded bounded NYC CSV, without accessing the endpoint.
python prepare_data.py --nyc-csv your_bounded_nyc.csv --output-dir imported_data
```

The receipt option verifies the snapshot hash and retains the original retrieval evidence; replay makes no network calls. CSV bytes reproduce exactly. Provenance records the chosen input path's basename and replay mode, so its metadata can differ. Two offline runs are byte-identical, including provenance. A fresh download may produce different real rows/counts, and records its actual status. If download access fails completely, preparation continues with `mode: synthetic_only` and 240 training rows; a partial download retains only actual usable rows. No mock NYC rows are fabricated.

Uploaded NYC CSVs must include `unique_key,complaint_type,descriptor,descriptor_2`, contain no more than 600 raw rows, no more than 150 per mapped category, and be at most 2 MB. Their authenticity is user-supplied unless accompanied by the retained hash-matched receipt.

## Schemas and split handling

Supervised files: `text,category,family_id,source,source_key`. Test adds `expected_status`.

Use **text only** as classifier input. Keep `category` as the ground-truth target. All other columns are metadata. Do not recombine and randomly split these files. Fit on train only, choose thresholds using validation only, then evaluate test once after choices are frozen.

The 20 in-scope test cases have `expected_status=routed` as the intended policy outcome, not an achieved model result. The four policy cases are vague, passport/unsupported, multi-issue and exposed live wire. Their categories are blank; three expect `review_required`, and the live-wire case expects `urgent_review`.

Scenario leaves for training, validation and test were curated separately, with no shared paraphrase family. Assertions confirm **zero normalized-text overlap and zero family-ID overlap between splits**, no within-split normalized duplicates, and the requested family counts. Normalization uses NFKC, casefolding and collapsed alphanumeric tokens. Semantic family separation is curated, not proven by those string/ID checks. Synthetic evaluation shares authoring bias and does not establish real Tel Aviv performance.

## Map fixtures

Fields: `report_id,city_id,text,category,latitude,longitude,created_at_utc,source`. Every row is `il_tel_aviv` / `synthetic_demo`. Seed **42**; stored anchor **2026-10-29T12:00:00Z**; all timestamps lie in the seven days ending at that anchor. Use the stored anchor as the demo window end.

| Planted group | Center (latitude, longitude) | Reports |
| --- | --- | ---: |
| roads | 32.068, 34.774 | 12 |
| waste | 32.080, 34.782 | 10 |
| lighting | 32.103, 34.792 | 8 |
| scattered | Within the brief's bounding box | 60 |

Planted points are within 60 metres of their center. Scattered points are more than 300 metres apart and more than 600 metres from the three planted centers, avoiding accidental hotspots. Category-specific checks at 250 metres / five samples verify dense groups of 12, 10 and 8, and no dense noise group. Coordinates are illustrative; the rough bounding box is not an official municipal boundary and can include areas outside municipal land.

Fixture IDs and group membership are in provenance. Import these as `status=open`, `routing_state=routed`, `category_source=seed_fixture`. These fields are import instructions, not extra CSV columns. All report IDs, texts and coordinates are distinct. No NYC locations are moved to Tel Aviv.

## Verification

The package's checks cover schemas/counts, policy labels, normalized text/family separation, raw-source limits, planted radii, seven-day timestamp limits, bounding-box validity and expected geographic groups. Offline determinism and exact real-snapshot CSV replay were checked separately. No model was trained or tested on the held-out file.
