# CityFix

An AI-assisted city governance application that is capable of sorting city complaints and spotting recurring issues on a map.

CityFix takes an English-language report about roads, waste, street lighting or noise, suggests a department, and prepares an acknowledgment. Staff can review uncertain reports, correct categories, update case status and investigate areas with several similar complaints. Residents can look up their report later by its reference number.

**Demo scope:** Tel Aviv-Yafo, using simulated local reports and illustrative department names. Nothing is sent to a municipality.

## The problem

A pothole, an overflowing bin and a broken streetlight can all arrive as free-text complaints. Someone needs to read each one, decide where it belongs and notice whether nearby residents are reporting the same kind of problem.

CityFix explores how much of that first pass can be automated while keeping staff involved when the answer is unclear. The intended value is less time spent sorting reports and better information for deciding where to investigate.

## What it does

- **Report an issue:** a chat. The resident types the complaint and CityFix replies straight away with a compact receipt: the reference (with a copy button), the **routing** result kept separate from **progress** toward resolution, the suggested department, a short reply with an *AI-written reply* or *Template response* badge, and a **Track report** button. An optional panel adds an area (tap it on a small map of 14 named Tel Aviv areas, or pick it from the list), coordinates, a photo or clip, or *Multiple issues / unsure*. "How was this routed?" stays collapsed. **Start new visit** clears the conversation; saved reports stay trackable.
- **Track a report:** a resident enters their reference number to see the case status and history, including notes from staff. No SMS or email is sent.
- **For Staff Review:** staff see the cases the system wasn't confident about in a single case card, then make one decision: the category (the department follows from the city's settings; the original prediction is kept) or **Spam / not a city issue**, which routes nowhere and closes the case, plus the case status — *Received, In progress, Resolved* or *Closed – not actionable* — and an optional note for the resident. One **Save decision** step applies both, and every action appears in a simple audit trail (when, action, note, staff name).
- **Admin dashboard:** filters and summary cards (*Needs review*, *Active cases*, *Resolved*, *Hotspots*) lead, followed by the map beside the review queue, the department queues and the hotspots. Report tables use plain labels (Reference, Issue, Area, Department, Status, Submitted); scores, assignment source and the audit trail sit in an expandable report view. Live reports classified by the model are labeled and counted separately from the simulated history. The routing explanation, thresholds and AI diagnostics are under **Model & data**.
- **Weekly Action Brief:** opens with what needs attention, computed in Python and useful without AI: a short operational summary, up to three **priority hotspots** (category and area, active report count, oldest active report, suggested department, a specific next step, and **Open related reports** / **View on map** buttons), then the cases waiting for staff review and the oldest active cases with their last update. Priorities are ranked by active report count, ties broken by the oldest active report — a workload ranking, not a danger assessment; resolved and closed cases are excluded. An optional **Draft handover summary** asks the AI only to explain why each priority is on the list and what to verify next; counts, dates, departments and ranking are inserted by code, and causes, trends, schedules, dispatch, promises and danger language are rejected (otherwise a labeled *Template draft* is shown). Evidence and methodology, including the computed facts, sit in a collapsed section.
- **Try another city configuration:** the same model can use a different department mapping without retraining. Local accuracy would still need separate evaluation.

Residents can attach a photo or short video (jpg, png, webp, mp4 or mov, up to 10 MB). Attachments are saved for staff review; the app does not analyze them or use them to classify a report.

## Run it on your computer

You need the complete project folder, including its supplied data and model files, and Python 3.11 or newer. The build was tested on Python 3.13. Core package versions are pinned in `requirements.txt`.

Open the project folder in VS Code, then open a terminal. These commands use **Git Bash on Windows**:

```bash
cd "/c/path/to/cityfix"
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Replace the example path with your project folder. On macOS or Linux, activate the environment with `source .venv/bin/activate`. Open [localhost:8501](http://localhost:8501) if the browser does not open automatically.

**Troubleshooting on Windows:** if importing numpy fails with *"An Application Control policy has blocked this file"*, Windows Smart App Control is blocking the package's unsigned files (Windows Security → App & browser control → Smart App Control). In Git Bash, use forward slashes: `source .venv/Scripts/activate`.

### Enable live AI drafts

The app works without any AI model: residents then see a **Template response** badge (staff see **Template fallback** with the reason). By default the drafts use **FLAN-T5-small**, a small model that runs locally — no API key or Hugging Face account is needed. To install it:

```bash
cp .env.example .env
python -m pip install -r requirements-ai.txt
python ai_service.py
python -m streamlit run app.py
```

`requirements-ai.txt` installs `transformers` and `torch` (a few hundred MB). The first `python ai_service.py` downloads the public model (about 300 MB) and prints a sample sentence. `.env` sets `AI_PROVIDER=flan_t5`; it can also be set to `anthropic` (with an `ANTHROPIC_API_KEY`) or `template`. Keep `.env` out of Git.

**Live generation was not verified during the build:** the build environment could not download the model. The integration was tested with a stand-in generator. Before presenting it as a live GenAI demonstration, check **Admin dashboard → Model & data → Run AI check** and confirm residents see the **AI-written reply** badge. When the model is missing or its output is unusable, residents see a **Template response** badge; the technical reason appears only under Model & data.

### Check or reproduce the build

```bash
python smoke_check.py
```

The supplied model is already trained and its held-out test has been evaluated. To rebuild from the supplied data:

```bash
python train.py
python train.py --evaluate-test
```

Training uses `gpt_train.csv`; routing thresholds are selected using `gpt_validation.csv`. The test file is reserved for final evaluation. Retraining clears old test results, and a second test evaluation for the same model requires `--force`. Missing required files cause the scripts to stop and name the file; they do not generate replacement data.

## The data story

The dataset combines a small amount of NYC 311 source material with synthetic English complaints. The Tel Aviv test examples and map reports are synthetic, so the results describe this demo dataset rather than real municipal performance.

All supplied project data files are in `data/` and start with `gpt_`.

| File | Contents | Purpose |
| --- | --- | --- |
| `gpt_train.csv` | 254 rows: 240 synthetic complaints and 14 deduplicated NYC structured details | Model training |
| `gpt_validation.csv` | 24 separately authored synthetic complaints, six per category | Validation and routing threshold selection |
| `gpt_tel_aviv_test.csv` | 20 in-scope complaints and four routing-policy cases | Final evaluation |
| `gpt_tel_aviv_demo.csv` | 90 simulated reports with locations and timestamps | Map and dashboard demonstration |
| `gpt_data_provenance.json` | Sources, generation method, counts, seed and file hashes | Traceability |
| `gpt_nyc_sample.csv`, `gpt_nyc_download_receipt.json`, `gpt_nyc_source_lineage.json` | A 600-row raw NYC snapshot, download receipt and lineage | Source documentation |

The 240 synthetic training complaints cover four categories, with 12 scenario families per category and five variants per family. Only complaint text is passed to the classifier. Source fields and family IDs remain metadata.

The train, validation and test splits stay separate. Integration checks found no shared family IDs or normalized complaint text across them, and file hashes matched the supplied provenance. The package README, verification report and reproducible `prepare_data.py` are in `data_package/`; the generator was not rerun during app integration.

The map data deliberately contains three hotspots: **12 roads reports, 10 waste reports and eight lighting reports**, plus 60 scattered reports. These demonstrate clustering; they are not evidence of actual problems in Tel Aviv. The simulated history ends at the fixture anchor, `2026-10-29T12:00:00Z`. The dashboard's default view, **Sample week + live reports**, shows the seven days before that anchor plus every report entered since; views for live reports only and the sample week only are also available.

The resident's area list contains 14 approximate points (Jaffa, South, Center, North and East). They are not exact addresses or official boundaries, and there is no geocoding. The three planted hotspots sit near *Rothschild Boulevard*, the *Rabin Square area* and the *Yarkon Park area*.

## How it works

| Part | Approach | Why it fits this prototype |
| --- | --- | --- |
| Complaint classification | TF-IDF text features and Logistic Regression | A small, inspectable model that runs locally on CPU |
| Routing | Score and margin thresholds, review rules and a city routing table | Lets the app defer uncertain cases to staff; the department comes from configuration, not the model |
| Hotspots | DBSCAN with haversine distance, per city and category; 250 m neighbour distance and five-report minimum | Finds nearby concentrations of similar reports |
| Drafting | Local FLAN-T5-small writes the human-sounding sentences; code fills every fact; template fallback | Generates acknowledgments and briefs without an API key and without letting the model invent facts |
| Interface and storage | Streamlit, pandas, pydeck and SQLite | Supports a local demo with saved reports, decisions and status history |

TF-IDF uses word unigrams and bigrams, up to 10,000 features and sublinear term frequency. Logistic Regression uses balanced class weights, `max_iter=1000` and `random_state=42`.

A report can be `routed`, `review_required` or `urgent_review`. Low scores, small gaps between the top two categories, unfamiliar vocabulary, missing department mappings and the resident's **Multiple issues / unsure** selection send a report to review. An urgent-phrase rule shows a static notice; it is not a reliable emergency detector.

Routing state is separate from **case status** (*Received → In progress → Resolved*, or *Closed – not actionable*). Received and in-progress cases count as active in queues and hotspots; resolved and closed cases don't. In this demo, anyone with a reference number can view that report's status; a real service would need verification.

The AI drafts text after routing and never changes routing, counts or saved data. FLAN-T5-small writes only the opening sentence of the acknowledgment and the hotspot sentences of the brief; it never sees the complaint text or attachments. Code adds the reference, department, evidence IDs and "recorded in this demo" notice. Each generated sentence is rejected if it contains numbers, promises about dispatch, deadlines or completion, department names or commitments, or isn't plain English. The optional Anthropic provider uses a 12-second timeout and no automatic retries.

## Results and what was learned

These are the recorded build results on the supplied synthetic validation and test sets:

| Measure | Validation | Held-out test |
| --- | --- | --- |
| Classification accuracy | 79.2% — 24 complaints | 80.0% — 20 in-scope complaints |
| Macro-F1 | 0.765 | 0.801 |
| In-scope reports routed automatically | 15/24 | 11/20 |
| Correct routes among those accepted | 15/15 | 11/11 |
| Separate routing-policy cases passed | — | 3/4 |

The classifier and routing policy answer different questions. A prediction can be wrong while the routing policy still sends that report to a person. All accepted in-scope routes were correct in these small sets, but nine of the 20 test complaints needed review. That is too little evidence to claim reliable real-world accuracy.

**The main challenge was choosing when to defer.** The starting score/margin thresholds of 0.65/0.15 routed none of the 24 validation complaints. Selecting 0.40/0.05 on validation increased coverage to 15/24 while keeping all accepted validation routes correct. Those thresholds were then frozen for the test. Model scores are uncalibrated estimates.

**A visible failure:** a report mentioning both a pothole and an overflowing bin was routed to waste instead of review. The app currently depends on the resident selecting **Multiple issues / unsure** to flag this situation. The failure remains documented in `artifacts/test_failures.csv`; the policy was not changed after seeing the test. Lighting was another weakness, with only two of six validation complaints classified correctly; for example, "flickering all night" is routed to noise.

The build also recorded **74/74 smoke checks passing**, covering data separation, routing, corrections, spam handling and the combined staff decision, shared case counts, Weekly Action Brief ranking and exclusions, handover text limits, case status and resident lookup, fixture imports, clustering, area names, attachment handling and AI fallback and safety filters. Results are stored in `artifacts/`.

## Business value and next steps

For a city service team, the useful decisions are practical: which department should review a report, which cases need a person, and which areas deserve investigation. A concentration of lighting reports could help staff choose an area to inspect. The weekly brief gives them a short starting point with traceable evidence, and residents can follow their case without calling.

The demo has not measured time savings, cost reduction or service improvement. A pilot would need authorized local reports, agreed department ownership and a fresh local holdout set. Running beside staff would let the team measure triage time, reassignment rate, review rate and accepted-route accuracy.

The next development priorities are automatic multi-issue detection and more varied lighting examples. Further roadmap items: retrieval over the city's responsibility rules and past staff decisions to suggest a department for unclear cases (staff decide); SMS or email updates when a case status changes, which needs contact details and consent; and image analysis, which would need privacy controls and evaluation on local photos. Production use would also require access control, approved city boundaries, case-system integration and monitored model releases. An annual licence plus onboarding is a possible business model, still unvalidated.

## Demo walkthrough

Use these steps for the live portion of the five-minute pitch, leaving time to explain the problem, data and results:

1. In **Report an issue**, enter "There is a deep pothole on our street near the bus stop, cars keep swerving." Select the area **Center · Rothschild Boulevard** and show the reference, proposed route, acknowledgment label and "How was this routed?".
2. Submit "Please help, it is a problem." to show the review path, and note its reference. "Exposed live wire near the school" demonstrates the urgent notice.
3. In **Admin dashboard**, point out the cards (*Needs review* matches the review queue) and the live reports on the map, then show the three planted hotspots.
4. In **For Staff Review**, open the vague report, choose waste and **Resolved** with a short note, and press **Save decision**; show the audit trail. Submitting "I need to renew my passport" and marking it **Spam** shows a message that routes nowhere and closes.
5. Back in **Report an issue → Track a report** (or the receipt's **Track report** button), show the history and staff note the resident sees.
6. Open **Weekly Action Brief**: show the ranked priorities, use **Open related reports** and **View on map**, then **Draft handover summary** and point out its label (*AI-written draft* or *Template draft*) and evidence links.
7. Open **Model & data** to show the results and known failure. Use **Portability check** to show the same pothole report mapped to **Public Works (demo)** in Example City without retraining.

## Project files

| File or folder | Role |
| --- | --- |
| `app.py` | Streamlit entry point and navigation |
| `citizen_page.py`, `admin_page.py`, `ui_common.py` | Resident page, dashboard and shared UI resources |
| `core.py` | Validation, classification, routing, storage, case status, clustering and calculated facts |
| `ai_service.py` | AI drafts (local FLAN-T5-small or Anthropic), provider check, output validation and fallback |
| `train.py`, `smoke_check.py` | Training, evaluation and build checks |
| `config/cities.json` | Demo city settings, named areas and department mappings |
| `requirements.txt`, `requirements-ai.txt` | Core packages (pinned) and optional local-AI packages |
| `.env.example`, `.streamlit/config.toml` | AI provider settings template and app theme (light-gray background, teal actions, 18 px text), upload limit |
| `assets/` | CityFix logo and icon shown in the top bar |
| `data/` | Supplied data; local `cityfix.db` and `uploads/` are created during use |
| `data_package/` | Data documentation, verification report and generator |
| `artifacts/` | Trained model, metadata, predictions, metrics and check results |

Department names are shown without the "(demo)" suffix that the configuration and saved history keep. Reports, decisions and status history are saved locally in SQLite; attachments are stored in `data/uploads/`. `.env`, the database, uploads and `data_picture/` are excluded from Git. This prototype has no login, geocoding or municipal dispatch integration.
