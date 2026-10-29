# CityFix

CityFix helps residents report street problems and city admins keep track of them. A resident describes a pothole, an overflowing bin, a broken streetlight or a noise issue. The app suggests a department, flags cases for review and shows where similar reports are gathering.

The aim is to spend less time sorting complaints and make follow-up easier for everyone.

> This version uses simulated Tel Aviv-Yafo reports and illustrative departments. Nothing is sent to a municipality.

## What you can do

- **Report an issue:** chat with CityFix, choose an area or coordinates, and optionally attach a photo or clip. Start a new visit without losing saved reports.
- **Track a report:** use its reference number to see the status, history and staff notes.
- **Manage cases:** review uncertain reports, correct categories, close spam and update case progress. Decisions stay in the audit history.
- **Plan investigations:** use the map and Weekly Action Brief to find recurring issues, review priorities and open the reports behind them.

Photos and clips are for staff review only. The app does not analyze them with AI.

## App structure

### Reporting and follow-up

```text
Resident complaint + optional location or attachment
       |
       v
TF-IDF + Logistic Regression
(trained complaint-category classifier)
       |
       v
Review rules + city department configuration
       |
       v
Routed / Review required / Urgent review
       |
       v
SQLite: saved report, routing result and case history
       |
       +--> Review cases --> Admin review queue
       |                          |
       |                    Staff decision + status update
       |                          |
       |<-------------------------+
       |
       +--> Track report: current status and history
       |
       v
Optional FLAN-T5-small wording / template response
       |
       v
Acknowledgment: code adds reference, routing result,
case status and simulated-data notice
```

A department suggestion and case progress are separate. Cases move through **Received, In progress, Resolved** or **Closed - not actionable**. The acknowledgment does not wait for a staff decision.

### Dashboard and Weekly Action Brief

```text
User-submitted reports + 90 simulated history records
       |
       v
Filter by city, reporting period and data source
       |
       +--> Dashboard: case counts, reports and department queues
       |
       v
Active cases: Received / In progress
       |
       +--> DBSCAN hotspots, grouped by city and category
       +--> Cases waiting for review + oldest active cases
       |
       v
Weekly Action Brief: Python computes priorities and report links
       |
       v
Optional FLAN-T5-small handover wording
(code supplies counts, dates, departments and evidence IDs)
```

The brief shows up to three hotspots, ranked by active report count, then oldest active report. It stays useful without AI. DBSCAN uses a 250 m neighbour distance and a five-report minimum.

The language model writes short sentences only. It receives no raw complaint text or attachments and cannot change routing or case records. If generation fails, the app shows a labeled template response or draft. The urgent-review flag is only a basic phrase check.

## Run the app

You need **Python 3.11 or newer** and the complete project folder, including the supplied data and model files. Open the folder in VS Code and use its terminal.

**Windows - PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac - Terminal**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Then, on either system:**

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) if the browser does not open automatically. The supplied classifier is already trained.

### Enable local AI wording

Copy `.env.example` to `.env` and set `AI_PROVIDER=flan_t5`. Keep `.env` out of Git.

```bash
python -m pip install -r requirements-ai.txt
python ai_service.py
```

The first check downloads FLAN-T5-small. No API key is needed. Restart the app afterward.

Live generation was not verified during the build. Use **Admin dashboard → Model & data → Run AI check** to check your installation. Without a working model, the app uses labeled templates.

## Data and results

Training uses synthetic English complaints plus a small amount of NYC 311 source material. The Tel Aviv test examples and map history are synthetic.

| File in `data/` | Purpose |
| --- | --- |
| `gpt_train.csv` | 254 training examples: 240 synthetic complaints and 14 deduplicated NYC structured details |
| `gpt_validation.csv` | 24 complaints for validation and routing-threshold selection |
| `gpt_tel_aviv_test.csv` | 20 in-scope complaints and four separate routing-policy cases |
| `gpt_tel_aviv_demo.csv` | 90 simulated reports for the dashboard |

Training, validation and test data stay separate. Sources and file hashes are recorded in `gpt_data_provenance.json`; data preparation details are in `data_package/`.

**Recorded held-out test results:**

| Measure | Result |
| --- | --- |
| Classification accuracy | 80.0% on 20 in-scope complaints |
| Macro-F1 | 0.801 |
| Automatically routed | 11/20 |
| Correct accepted routes | 11/11 |
| Separate policy cases passed | 3/4 |

Nine of the 20 in-scope reports went to staff review. A complaint mentioning both a pothole and an overflowing bin incorrectly routed to waste. Lighting language also needs improvement. These small synthetic sets do not establish real-world accuracy or time savings.

The build recorded **74/74 smoke checks passing**. To run them:

```bash
python smoke_check.py
```

Evaluation outputs and documented failures are in `artifacts/`.

## Next steps

- **Test with a city team:** agree department responsibilities, evaluate authorized local reports and measure triage time, reassignments and routing accuracy.
- **Connect messaging:** add a shared report API for WhatsApp and other messaging tools, with opted-in status updates.
- **Add assistance:** explore image analysis and retrieval over city responsibility rules and approved staff decisions.
- **Prepare for real use:** add verified access, city permissions, approved boundaries and municipal case-system integration.

The same classifier can use another city's department mapping without retraining, as shown by the portability check. Each new city still needs local evaluation. The current app has no login, geocoding, automatic notifications or municipal dispatch connection.
