#!/usr/bin/env python3
"""Prepare CityFix data only. Python 3.10+; standard library; no API key.

Offline: python prepare_data.py
Live:    python prepare_data.py --download-nyc
Import:  python prepare_data.py --nyc-csv path/to/bounded.csv
Replay:  python prepare_data.py --nyc-csv data/gpt_nyc_sample.csv --output-dir replay
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

SEED = 42
ANCHOR = "2026-10-29T12:00:00Z"
ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
CATEGORIES = ("roads", "waste", "lighting", "noise")
MAPPING = {"Street Condition": "roads", "Dirty Condition": "waste",
           "Street Light Condition": "lighting", "Noise - Residential": "noise",
           "Noise - Street/Sidewalk": "noise"}
RAW_FIELDS = ("unique_key", "complaint_type", "descriptor", "descriptor_2")
FIELDS = ("text", "category", "family_id", "source", "source_key")
DEMO_FIELDS = ("report_id", "city_id", "text", "category", "latitude",
               "longitude", "created_at_utc", "source")
BBOX = (32.04, 34.74, 32.15, 34.84)
EARTH = 6_371_000

# Conservative, manually reviewed source-family membership. Unreviewed source
# descriptors stay in gpt_nyc_sample.csv; they cannot accidentally introduce a
# held-out family into training. Output labels still come only from MAPPING.
SOURCE_FAMILIES = {
    "roads": {
        "pothole": "train_roads_pothole",
        "rough pitted or cracked roads": "train_roads_surface_cracks",
        "failed street repair": "nyc_roads_failed_repair",
        "blocked construction": "nyc_roads_construction_obstruction"},
    "waste": {"broken glass": "train_waste_glass_shards",
              "dog waste": "train_waste_dog_fouling"},
    "lighting": {
        "fixture luminaire out of position": "train_lighting_wrong_beam",
        "multiple street lights out": "train_lighting_row_lamps_out",
        "street light cycling": "train_lighting_erratic_flicker",
        "street light out": "train_lighting_single_lamp_out",
        "street light lamp missing": "nyc_lighting_missing_bulb",
        "lamppost knocked down": "nyc_lighting_fallen_column"},
    "noise": {"loud television": "train_noise_tv_volume",
              "loud music party": "train_noise_apartment_music"}}
ALLOWED_SECONDARY = {"", "n a", "other", "unknown", "unspecified", "not applicable",
                     "not cleaned by property owner", "not picked up by dog owner"}

# Each row is one scenario family: ID | issue | specific context | effect.
# Five narrative variants share that family and always stay in training.
TRAIN_CATALOG = {
"roads": """
pothole|a deep pothole has opened in the asphalt|beside the bus stop|vehicles swerve around the hole
surface_cracks|the road surface has long branching cracks|outside the grocery shop|pieces of asphalt are starting to break away
loose_pavers|several paving slabs rock underfoot|on the pedestrian pavement|people stumble when the slabs move
sunken_trench|a refilled utility trench has sunk below the road|across the junction approach|cars hit a sharp dip
broken_speed_hump|part of the speed hump has crumbled away|at the entrance to the residential lane|the remaining ridge has a jagged edge
raised_bridge_joint|a bridge expansion joint sits above the carriageway|on the small road bridge|tyres strike the raised metal strip
broken_kerb|the kerbstone has broken into pieces|at the corner of the pavement|the pavement edge is collapsing
root_lift|tree roots have lifted the pavement|along the planted walking strip|the raised edge trips pedestrians
rocking_grate|a drain grate rocks in its road frame|beside the parking bay|the metal cover moves under passing wheels
loose_gravel|loose gravel covers the road surface|after resurfacing near the roundabout|bicycles skid on the stones
wheel_ruts|deep wheel ruts run along the asphalt|in the bus lane|small vehicles are pulled into the grooves
damaged_ramp|the public pavement ramp has a broken concrete lip|at the pedestrian crossing approach|wheelchairs cannot roll over it smoothly
""",
"waste": """
overflowing_bin|the public litter bin is overflowing|beside the bus shelter|rubbish is piling onto the pavement
discarded_sofa|a discarded sofa blocks the public passage|behind the apartment block|pedestrians have to squeeze past it
missed_collection|the scheduled household rubbish collection was missed|along our row of buildings|the tied refuse bags are still waiting outside
glass_shards|broken bottles cover the public footpath|near the outdoor seating area|glass shards are scattered where people walk
dog_fouling|dog droppings have accumulated|along the park entrance path|people keep stepping around the mess
restaurant_sacks|restaurant rubbish sacks leak greasy liquid|beside the public collection point|the pavement is dirty and smells bad
cardboard_pile|flattened cardboard boxes fill the walkway|outside the delivery entrance|the stacked cartons leave little room to pass
plastic_verge|plastic wrapping is tangled across the grass verge|beside the main road|packaging scraps cover the planted strip
beach_picnic|picnic leftovers and disposable plates were left on the sand|at the public beach access|food litter is attracting birds
cigarette_butts|cigarette butts cover the ground|around the public bench|the seating area needs cleaning
alley_food_scraps|loose food scraps have been dumped in the alley|behind the shops|the exposed waste is attracting flies
fence_litter|bags and wrappers are caught in a public fence|along the playground boundary|the fence is covered in litter
""",
"lighting": """
single_lamp_out|one streetlamp does not come on at night|beside the bus shelter|the waiting area stays dark
row_lamps_out|a whole row of streetlights is unlit|along the residential block|the entire pavement is dark after sunset
erratic_flicker|a streetlamp flickers erratically|outside the corner shop|the light repeatedly flashes while people walk below
daytime_on|the streetlights remain on in full daylight|along the shopping street|the lamps are burning through the afternoon
late_switch_on|the streetlights switch on long after dusk|beside the park entrance|the path stays dark during the evening commute
dim_bulb|a streetlight has become very dim|at the public crossing approach|the pavement is barely visible below it
wrong_beam|a streetlight points away from the pavement|near the community hall|the beam lights an empty wall instead of the path
cracked_lens|the protective lens of a streetlight is cracked|over the public walkway|the lamp cover needs maintenance
tilted_head|the head of a streetlamp has tilted sideways|beside the parking area|the lamp assembly is visibly out of position
foliage_shield|branches completely shield a public path light|along the planted promenade|almost no illumination reaches the walking surface
solar_early_flat|a solar path lamp goes dark early each evening|in the small public garden|the lamp stops working well before the other lights
underpass_outage|the fixed ceiling lamps in the pedestrian underpass are out|beneath the main road|the tunnel is dark even during the day
""",
"noise": """
apartment_music|amplified music is playing very loudly|inside a neighbouring apartment late at night|the bass can be heard through closed windows
sidewalk_speaker|a portable speaker is blasting music|on the public pavement|nearby residents cannot sleep
street_shouting|a group is repeatedly shouting|on the street below our building after midnight|the voices keep waking the household
barking_dog|a dog barks for long periods|inside the neighbouring flat|the barking continues through the quiet hours
power_tools|power tools are being used loudly|inside the apartment upstairs before dawn|the drilling interrupts sleep
furniture_dragging|heavy furniture is dragged across the floor repeatedly|in the flat above us late at night|the scraping and thuds carry into our bedroom
tv_volume|a television is turned up extremely loud|in the next apartment after midnight|we can hear the programmes word for word
home_drum_practice|someone is practising a drum kit loudly|inside a residential flat at night|the repeated beats shake our room
balcony_party|a balcony gathering has loud cheering and singing|across from our bedroom late in the evening|the voices continue well into the night
door_slamming|an apartment door is slammed over and over|in the residential corridor during the night|the impacts keep waking us
sidewalk_skateboarding|skateboards repeatedly strike the pavement|below the apartment windows late at night|the clattering echoes through the bedrooms
car_stereo_gathering|people are playing a loud car stereo while gathering outside|on the street after midnight|the bass carries into nearby homes
"""
}

# Independently authored sentences; no training template is used here.
VALIDATION = {
"roads": [
("faded_crossing", "The painted pedestrian crossing has almost disappeared; drivers cannot make out its white stripes."),
("broad_subsidence", "A broad section of carriageway has dropped noticeably, leaving the middle of the junction lower than its edges."),
("asphalt_blister", "A raised bubble has formed in the road's asphalt coating and is lifting the surface into a dome."),
("tactile_paving", "The tactile paving tiles at the crossing have broken studs, so the guidance surface is incomplete."),
("lifted_road_plate", "A temporary steel road plate has lifted at one end, creating a hard step for passing cars."),
("traffic_island", "The concrete pedestrian refuge in the middle of the road is chipped away and needs repair.")],
"waste": [
("recycling_bank", "The bottle recycling bank is full to its opening, and more empty containers have been left around it."),
("abandoned_electronics", "An old television and a broken computer were abandoned beside the communal bins for removal."),
("market_crates", "Empty produce crates are strewn across the public market after the stalls closed."),
("foam_packaging", "White foam packing pieces are scattered over the public square following a delivery."),
("discarded_textiles", "Sacks of unwanted clothing have been left on the municipal stair landing and need collection."),
("torn_refuse_bags", "Refuse bags left for pickup have torn open, spilling their contents across the shared collection area.")],
"lighting": [
("excessive_glare", "The public streetlight shines with such harsh glare into our bedroom that we need its brightness checked."),
("headlight_sensor", "Each time a car passes, its headlights trigger the streetlamp sensor and the pavement light switches off."),
("water_in_lens", "Rainwater is collecting inside the streetlight's transparent cover; droplets are visible around the bulb."),
("rusted_bracket", "The mounting bracket holding the public lamp is badly corroded and needs inspection."),
("colour_shift", "The lamp on the public footpath has changed from white to an unusual purple colour."),
("loose_pole_base", "The base of the lighting column has loose bolts, and the pole wobbles slightly when touched; no wires are visible.")],
"noise": [
("piano_rehearsal", "Piano scales from the next flat continue past midnight at a volume that keeps our child awake."),
("loud_domestic_calls", "Our neighbour makes very loud phone calls beside the shared bedroom wall every night."),
("exercise_impacts", "Someone jumps repeatedly on an exercise machine in the apartment above during sleeping hours."),
("hallway_running", "Children run and stamp along the residential hallway very early each morning, disturbing the flats."),
("street_whistling", "A person whistles sharply and repeatedly on the pavement beneath our windows long after midnight."),
("sidewalk_game", "A late-night pavement game involves repeated ball bouncing directly outside our sleeping area.")]
}

TEST = {
"roads": [
("damaged_delineators", "On the Tel Aviv demo cycle route, several fixed lane delineators have snapped off at ground level."),
("uneven_cobbles", "The cobblestones in the shared street have shifted into an uneven ridge that catches bicycle tyres."),
("old_lane_markings", "Old lane arrows remain visible after the road layout changed and are directing drivers into the wrong lane."),
("public_steps", "A tread is missing from the municipal pedestrian steps, leaving a gap in the staircase."),
("cycle_separator", "A concrete separator beside the cycle lane has broken apart and is jutting into the riding space.")],
"waste": [
("paper_in_courtyard", "Windblown leaflets now blanket the paved public courtyard; please arrange a sweep."),
("construction_rubble", "Bags of renovation rubble have been dumped on a public verge in the Tel Aviv demo area."),
("garden_cuttings", "A large heap of cut branches and garden trimmings was left on the promenade for removal."),
("truck_spillage", "A refuse truck spilled mixed rubbish at the junction, leaving a trail across the carriageway."),
("spoiled_produce", "Rotting vegetables have been left beside the public food-sharing cupboard and need to be cleared away.")],
"lighting": [
("partial_highmast_panel", "Half the LED panels on the high-mast public light have stopped working while the remaining panels still glow."),
("detached_shade", "The shade has come off the promenade lamp, so its light spills upwards instead of being contained."),
("missing_service_id", "The service identification plate is missing from a municipal lighting column, preventing us from identifying the lamp for maintenance."),
("jammed_service_hatch", "The maintenance hatch on the streetlight column is jammed open; the internal wiring remains behind its intact sealed barrier."),
("salt_deposits", "A thick white salt coating has formed on the seaside lamp housing and its glass needs specialist cleaning.")],
"noise": [
("alarm_clock", "An alarm clock rings repeatedly in the empty neighbouring flat for hours every morning and wakes us."),
("bedroom_fan_rattle", "A neighbour's indoor bedroom fan rattles loudly against our shared wall throughout the night."),
("home_gaming", "Late-night gaming in the adjacent apartment includes repeated loud yelling into a headset."),
("street_dancing", "A group practises tap dancing on the pavement beneath our windows at night, making continuous sharp tapping sounds."),
("home_singing_lessons", "Singing lessons in the flat opposite are held at amplified volume during our building's quiet evening hours.")]
}
POLICY = [
("vague", "Something is wrong near my place in Tel Aviv. Can somebody help?", "review_required"),
("unsupported_passport", "How do I renew my passport and change the name printed in it?", "review_required"),
("multiple_issues", "There is a new pothole beside an overflowing rubbish bin, and both need attention.", "review_required"),
("exposed_live_wire", "An exposed live wire is sparking from the streetlight pole beside the footpath.", "urgent_review")]

TEMPLATES = [
"I noticed {issue} {context}; {effect}.",
"Please check this: {issue} {context}. As a result, {effect}.",
"At this location, {context}, {issue}. We have found that {effect}.",
"Could someone inspect the following problem? {issue_cap} {context}, and {effect}.",
"Reporting a local maintenance concern: {issue} {context}. Right now, {effect}."
]

def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))

def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def csv_bytes(rows, fields):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")

def training_rows():
    rows = []
    for category, catalog in TRAIN_CATALOG.items():
        entries = [line.split("|") for line in catalog.strip().splitlines()]
        assert len(entries) == 12
        for slug, issue, context, effect in entries:
            for template in TEMPLATES:
                text = template.format(issue=issue, issue_cap=issue[0].upper()+issue[1:],
                                       context=context, effect=effect)
                rows.append(dict(zip(FIELDS, [text, category, f"train_{category}_{slug}",
                                              "synthetic", ""])))
    return rows

def authored_rows(catalog, split):
    return [dict(zip(FIELDS, [text, cat, f"{split}_{cat}_{slug}", "synthetic", ""]))
            for cat, examples in catalog.items() for slug, text in examples]

def fetch_worker(url):
    """Subprocess worker: socket timeout plus parent-enforced wall timeout."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                  "User-Agent": "CityFix-data-prep/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("response exceeded two-megabyte cap")
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) > 150 or not all(isinstance(x, dict) for x in rows):
            raise ValueError("expected an array of at most 150 objects")
        print(json.dumps({"status": "success", "rows": rows}))
    except urllib.error.HTTPError as exc:
        print(json.dumps({"status": "failed", "error": f"HTTP {exc.code}"}))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))

def download_nyc():
    start = time.monotonic()
    deadline = start + 180
    rows, attempts = [], []
    for category in CATEGORIES:
        labels = [k for k, v in MAPPING.items() if v == category]
        quoted = ",".join("'" + x.replace("'", "''") + "'" for x in labels)
        params = {"$select": ",".join(RAW_FIELDS),
                  "$where": f"complaint_type in ({quoted}) AND descriptor IS NOT NULL AND descriptor != ''",
                  "$limit": "150", "$order": "unique_key DESC"}
        url = ENDPOINT + "?" + urllib.parse.urlencode(params)
        attempt = {"category": category, "parameters": params, "encoded_url": url,
                   "returned_rows": 0}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            attempt.update(status="cutoff", error="three-minute total cutoff")
        else:
            t0 = time.monotonic()
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_fetch", url],
                                      text=True, capture_output=True, timeout=min(10, remaining), check=True)
                result = json.loads(proc.stdout)
                attempt["status"] = result["status"]
                if result["status"] == "success":
                    received = result["rows"]
                    attempt["returned_rows"] = len(received)
                    # Preserve only the requested fields, never contacts/locations.
                    rows.extend({f: str(r.get(f) or "").strip() for f in RAW_FIELDS} for r in received)
                else:
                    attempt["error"] = result["error"]
            except subprocess.TimeoutExpired:
                attempt.update(status="failed", error="10-second request wall timeout")
            except Exception as exc:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            attempt["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        attempts.append(attempt)
    return rows, {"status": "complete" if all(a["status"] == "success" for a in attempts)
                  else "partial" if rows else "failed", "attempts": attempts,
                  "elapsed_seconds": round(time.monotonic() - start, 3),
                  "request_timeout_seconds": 10, "total_cutoff_seconds": 180,
                  "retries_per_query": 0, "max_rows_per_canonical_category": 150}

def load_bounded_csv(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("uploaded NYC CSV exceeds the two-megabyte bound")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(RAW_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("NYC CSV requires unique_key,complaint_type,descriptor,descriptor_2")
        rows = []
        for raw in reader:
            if len(rows) >= 600:
                raise ValueError("uploaded NYC CSV exceeds 600 raw rows")
            rows.append({f: (raw.get(f) or "").strip() for f in RAW_FIELDS})
    counts = collections.Counter(MAPPING.get(r["complaint_type"]) for r in rows)
    if any(counts[c] > 150 for c in CATEGORIES):
        raise ValueError("uploaded NYC CSV exceeds 150 raw rows per canonical category")
    return rows

def useful_source_rows(raw_rows, heldout):
    """Labels come only from the explicit source mapping; conflicting details excluded."""
    stats = collections.Counter()
    groups = collections.defaultdict(list)
    seen_keys = set()
    reserved_texts = {normalize(r["text"]) for r in heldout}
    for r in raw_rows:
        category = MAPPING.get(r["complaint_type"])
        if category is None:
            stats["unmapped_rows_excluded"] += 1
            continue
        if not r["unique_key"] or not r["descriptor"]:
            stats["missing_key_or_detail_excluded"] += 1
            continue
        # Inspect conflicts before key deduplication, including malformed imports.
        bits = []
        for value in (r["descriptor"], r["descriptor_2"]):
            clean = " ".join(value.split())
            if clean.casefold().startswith("location type:"):
                stats["location_metadata_fields_removed"] += 1
                continue
            if clean and normalize(clean) not in {"other", "unknown", "unspecified", "n a", "not applicable", "condition"}:
                if normalize(clean) not in {normalize(b) for b in bits}:
                    bits.append(clean)
        text = "; ".join(bits)
        normalized = normalize(text)
        if not normalized:
            stats["uninformative_details_excluded"] += 1
            continue
        # A repeated descriptor with a different generic location type is not
        # another independent language example. Group all secondary qualifiers.
        groups[normalize(r["descriptor"])].append((r, category, text))
    output, lineage = [], []
    for normalized, entries in sorted(groups.items()):
        categories = {e[1] for e in entries}
        keys = {e[0]["unique_key"] for e in entries}
        if len(categories) > 1:
            stats["conflicting_descriptor_rows_excluded"] += len(entries)
            continue
        category = next(iter(categories))
        family = SOURCE_FAMILIES[category].get(normalized)
        if not family:
            stats["unreviewed_or_heldout_family_rows_excluded"] += len(entries)
            continue
        reviewed_entries = []
        for entry in entries:
            secondary = entry[0]["descriptor_2"]
            if secondary.casefold().startswith("location type:") or normalize(secondary) in ALLOWED_SECONDARY:
                reviewed_entries.append(entry)
            else:
                stats["unreviewed_secondary_detail_rows_excluded"] += 1
        entries = reviewed_entries
        if not entries:
            continue
        keys = {e[0]["unique_key"] for e in entries}
        if any(normalize(e[2]) in reserved_texts for e in entries):
            stats["heldout_or_generated_text_collision_rows_excluded"] += len(entries)
            continue
        chosen = next((e for e in entries if e[0]["unique_key"] not in seen_keys), None)
        if chosen is None:
            stats["repeated_source_key_rows_excluded"] += len(entries)
            continue
        raw, category, text = chosen
        seen_keys.update(keys)
        output.append(dict(zip(FIELDS, [text, category, family, "nyc_structured", raw["unique_key"]])))
        lineage.append({"family_id": family, "category": category,
                        "retained_source_key": raw["unique_key"], "all_source_keys": sorted(keys),
                        "raw_rows_in_group": len(entries), "text": text})
        stats["repetitive_detail_rows_collapsed"] += len(entries)-1
    stats["useful_unique_details"] = len(output)
    return output, dict(stats), lineage

def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, [*a, *b])
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*EARTH*math.asin(min(1, math.sqrt(h)))

def offset(center, radius, angle):
    north, east = radius*math.cos(angle), radius*math.sin(angle)
    return (round(center[0]+math.degrees(north/EARTH), 7),
            round(center[1]+math.degrees(east/(EARTH*math.cos(math.radians(center[0])))), 7))

def map_rows():
    rng = random.Random(SEED)
    anchor = dt.datetime.fromisoformat(ANCHOR.replace("Z", "+00:00"))
    rows, manifest = [], []
    centers = [("roads", (32.068, 34.774), 12), ("waste", (32.080, 34.782), 10),
               ("lighting", (32.103, 34.792), 8)]
    def add(category, point, text, group):
        report_id = f"demo_{len(rows)+1:03d}"
        stamp = anchor-dt.timedelta(seconds=rng.randint(0, 7*86400-1))
        rows.append(dict(zip(DEMO_FIELDS, [report_id, "il_tel_aviv", text, category,
                         point[0], point[1], stamp.isoformat().replace("+00:00", "Z"), "synthetic_demo"])))
        manifest.append({"report_id": report_id, "fixture_group": group})
    messages = {
        "roads": ["A rough patch of road jolts vehicles at this illustrative location",
                  "The synthetic street surface has a damaged section needing inspection",
                  "An uneven section of carriageway makes cycling uncomfortable here"],
        "waste": ["Litter is accumulating at this illustrative collection area",
                  "Loose refuse needs clearing from this synthetic public space",
                  "Discarded packaging covers part of this demonstration walkway"],
        "lighting": ["Public illumination needs inspection at this illustrative location",
                     "The synthetic path lighting is not working properly here",
                     "A public lamp needs maintenance at this demonstration point"],
        "noise": ["Loud voices are disturbing residents at this illustrative location",
                  "Late evening amplified sound is reported at this synthetic point",
                  "Repeated loud sound interrupts sleep near this demonstration location"]}
    for category, center, count in centers:
        for i in range(count):
            point = offset(center, 59*math.sqrt(rng.random()), rng.random()*2*math.pi)
            add(category, point, f"{messages[category][i%3]}. Simulated report {len(rows)+1:03d}.",
                f"planted_{category}")
    # Random scattered fixtures use minimum spacing to prevent accidental DBSCAN groups.
    scattered = []
    attempts = 0
    while len(scattered) < 60:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not place scattered map points")
        point = (round(rng.uniform(BBOX[0], BBOX[2]), 7), round(rng.uniform(BBOX[1], BBOX[3]), 7))
        if any(haversine(point, center) <= 600 for _, center, _ in centers):
            continue
        if any(haversine(point, other) <= 300 for other in scattered):
            continue
        category = CATEGORIES[len(scattered)%4]
        i = len(scattered)
        scattered.append(point)
        add(category, point, f"{messages[category][i%3]}. Simulated report {len(rows)+1:03d}.", "scattered")
    return rows, manifest

def check_splits(splits):
    checks = {"normalization": "NFKC, casefold, ASCII alphanumeric tokens, collapsed whitespace",
              "within_split_normalized_duplicates": {}, "between_splits": {},
              "family_isolation_scope": "training/validation/test, not map fixtures",
              "semantic_family_method": "Manually curated scenario leaves: 48 training families; each validation/test case has a separate issue mechanism. Shared category vocabulary is allowed; no source-derived evaluation narratives.",
              "limitations": "ID/text checks do not prove semantic independence. Synthetic authoring and taxonomy are shared; evaluation is not real-city accuracy."}
    for name, rows in splits.items():
        duplicates = len(rows)-len({normalize(r["text"]) for r in rows})
        checks["within_split_normalized_duplicates"][name] = duplicates
        assert duplicates == 0, (name, "duplicate normalized text")
        assert all(0 < len(r["text"]) <= 1000 for r in rows)
    for a, b in itertools.combinations(splits, 2):
        texts = {normalize(r["text"]) for r in splits[a]} & {normalize(r["text"]) for r in splits[b]}
        families = {r["family_id"] for r in splits[a]} & {r["family_id"] for r in splits[b]}
        checks["between_splits"][f"{a}__{b}"] = {"normalized_text_overlap": len(texts), "family_id_overlap": len(families)}
        assert not texts and not families
    family_counts = collections.Counter(r["family_id"] for r in splits["train"] if r["source"] == "synthetic")
    assert len(family_counts) == 48 and set(family_counts.values()) == {5}
    assert collections.Counter(r["category"] for r in splits["validation"]) == dict.fromkeys(CATEGORIES, 6)
    assert collections.Counter(r["category"] for r in splits["test"] if r["category"]) == dict.fromkeys(CATEGORIES, 5)
    assert sum(r["category"] == "" for r in splits["test"]) == 4
    assert all(r["source"] == "synthetic" for r in splits["validation"]+splits["test"])
    assert all(r["expected_status"] in {"review_required", "urgent_review"} for r in splits["test"] if not r["category"])
    checks["training_synthetic_families"] = 48
    checks["variants_per_training_family"] = 5
    checks["all_assertions_passed"] = True
    return checks

def check_map(rows, manifest):
    assert len(rows) == 90 and len({r["report_id"] for r in rows}) == 90
    assert len({normalize(r["text"]) for r in rows}) == 90
    assert len({(r["latitude"], r["longitude"]) for r in rows}) == 90
    anchor = dt.datetime.fromisoformat(ANCHOR.replace("Z", "+00:00"))
    for r in rows:
        lat, lon = r["latitude"], r["longitude"]
        assert math.isfinite(lat) and math.isfinite(lon)
        assert BBOX[0] <= lat <= BBOX[2] and BBOX[1] <= lon <= BBOX[3]
        age = (anchor-dt.datetime.fromisoformat(r["created_at_utc"].replace("Z", "+00:00"))).total_seconds()
        assert 0 <= age < 7*86400
    groups = {m["report_id"]: m["fixture_group"] for m in manifest}
    expected = {"planted_roads": 12, "planted_waste": 10, "planted_lighting": 8, "scattered": 60}
    assert collections.Counter(groups.values()) == expected
    max_radius = {}
    for category, center in [("roads", (32.068, 34.774)), ("waste", (32.080, 34.782)), ("lighting", (32.103, 34.792))]:
        planted = [r for r in rows if groups[r["report_id"]] == f"planted_{category}"]
        maximum = max(haversine(center, (r["latitude"], r["longitude"])) for r in planted)
        assert maximum <= 60
        max_radius[category] = round(maximum, 3)
    # Reference connectivity check at DBSCAN eps=250m, min_samples=5.
    # Clusters planted here are cliques; scattered points have no neighbours within eps.
    cluster_sizes = {}
    for cat in CATEGORIES:
        group = [r for r in rows if r["category"] == cat]
        dense = []
        for r in group:
            n = sum(haversine((r["latitude"], r["longitude"]), (q["latitude"], q["longitude"])) <= 250 for q in group)
            if n >= 5:
                dense.append(r)
        cluster_sizes[cat] = len(dense)
        assert len(dense) == {"roads":12, "waste":10, "lighting":8, "noise":0}[cat]
    return {"all_assertions_passed": True, "fixture_groups": expected,
            "max_planted_distance_meters": max_radius,
            "dense_points_by_category_eps250_min5": cluster_sizes,
            "scattered_min_separation_meters": 300,
            "scattered_distance_from_planted_centers_meters": 600}

def counts(rows):
    return {"rows": len(rows), "by_category": dict(sorted(collections.Counter(r["category"] or "policy_unlabeled" for r in rows).items())),
            "by_source": dict(sorted(collections.Counter(r["source"] for r in rows).items())),
            "families": len({r["family_id"] for r in rows}) if rows and "family_id" in rows[0] else None}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-nyc", action="store_true")
    mode.add_argument("--nyc-csv", type=Path)
    parser.add_argument("--nyc-download-log", type=Path,
                        help="Optional retained API receipt for --nyc-csv; verifies snapshot SHA-256")
    args = parser.parse_args()
    if args.nyc_download_log and not args.nyc_csv:
        parser.error("--nyc-download-log requires --nyc-csv")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    synthetic = training_rows()
    validation = authored_rows(VALIDATION, "validation")
    test = authored_rows(TEST, "test")
    for row in test:
        row["expected_status"] = "routed"
    for slug, text, status in POLICY:
        test.append(dict(zip((*FIELDS, "expected_status"), [text, "", "test_policy_"+slug, "synthetic", "", status])))
    raw = []
    imported = None
    download = {"status": "not_attempted", "reason": "offline synthetic-only mode requested by default"}
    if args.download_nyc:
        raw, download = download_nyc()
    elif args.nyc_csv:
        raw = load_bounded_csv(args.nyc_csv)
        imported = {"filename": args.nyc_csv.name, "sha256": digest_bytes(args.nyc_csv.read_bytes()),
                    "authenticity": "User-supplied structured data; not independently verified. Never generated by this script."}
        download = {"status": "not_attempted", "reason": "bounded CSV supplied instead"}
        if args.nyc_download_log:
            receipt = json.loads(args.nyc_download_log.read_text(encoding="utf-8"))
            if receipt["snapshot_sha256"] != digest_bytes(csv_bytes(raw, RAW_FIELDS)):
                raise ValueError("NYC API receipt does not match the CSV snapshot")
            download = receipt["download"]
            imported["authenticity"] = "Retained snapshot of the actual recorded API retrieval; receipt hash verified. No new request during replay."
    useful, source_checks, lineage = useful_source_rows(raw, synthetic+validation+test)
    train = useful+synthetic
    maps, fixture_manifest = map_rows()
    split_checks = check_splits({"train": train, "validation": validation, "test": test})
    map_checks = check_map(maps, fixture_manifest)
    data_mode = ("nyc_structured_plus_synthetic" if useful else "synthetic_only")
    used_source = "bounded_nyc_csv" if args.nyc_csv else "nyc_api" if raw else None
    # Clear only our optional outputs to avoid stale real-source claims on offline reruns.
    for optional in ("gpt_nyc_sample.csv", "gpt_nyc_source_lineage.json", "gpt_nyc_download_receipt.json"):
        if (out/optional).exists():
            (out/optional).unlink()
    contents = {"gpt_train.csv": csv_bytes(train, FIELDS), "gpt_validation.csv": csv_bytes(validation, FIELDS),
                "gpt_tel_aviv_test.csv": csv_bytes(test, (*FIELDS, "expected_status")),
                "gpt_tel_aviv_demo.csv": csv_bytes(maps, DEMO_FIELDS)}
    if raw:
        contents["gpt_nyc_sample.csv"] = csv_bytes(raw, RAW_FIELDS)
        contents["gpt_nyc_source_lineage.json"] = (json.dumps(lineage, indent=2, ensure_ascii=False)+"\n").encode()
        if args.download_nyc or args.nyc_download_log:
            receipt = {"snapshot_sha256": digest_bytes(contents["gpt_nyc_sample.csv"]), "download": download}
            contents["gpt_nyc_download_receipt.json"] = (json.dumps(receipt, indent=2)+"\n").encode()
    for name, data in contents.items():
        (out/name).write_bytes(data)
    provenance = {
        "schema_version": "cityfix-data-v1", "mode": data_mode, "seed": SEED,
        "demo_anchor_utc": ANCHOR, "demo_default_window_days": 7,
        "brief": {"filename": "CityFix_02_Tel_Aviv_Claude_Product_and_Architecture_Brief.md",
                  "library_file_id": "libfile_a30ca282a87c8191a091271d57796121", "config_version": "demo-v1"},
        "sources": {"endpoint": ENDPOINT, "source_label_mapping": MAPPING,
                    "structured_source_used": used_source, "uploaded_csv": imported,
                    "requested_fields": list(RAW_FIELDS), "raw_rows_retained": len(raw),
                    "raw_rows_by_category": dict(collections.Counter(MAPPING.get(r["complaint_type"], "unmapped") for r in raw)),
                    "original_conversational_complaints_available": False,
                    "nyc_geography_transferred_to_tel_aviv": False},
        "download": download,
        "generation": {"method": "Assistant-authored fixed scenario catalog; five deterministic language templates per training family; validation/test authored separately; no external generative or prediction API at runtime.",
                       "synthetic_train_rows": 240, "synthetic_train_per_category": 60,
                       "train_families_per_category": 12, "variants_per_family": 5,
                       "ground_truth": "Canonical category assigned to curated scenarios before rendering text; real details labeled only by exact source-label mapping. No classifier predictions or pseudo-labeling.",
                       "synthetic_from_nyc_rows": 0,
                       "synthetic_from_nyc_reason": "The authored narratives are independent of the downloaded details, so every generated narrative is marked synthetic.",
                       "blank_source_key": "No real source record used for generated rows.",
                       "randomness": "random.Random(42) is used only for map jitter, placement and historical timestamps; catalogs are fixed.",
                       "test_expected_status": "routed is the desired policy outcome for 20 in-scope cases, not a measured model result; 3 review_required and 1 urgent_review policy fixtures have blank category.",
                       "limitations": ["Synthetic evaluation shares authoring and generation bias; it is not an estimate of real Tel Aviv accuracy.",
                                       "Illustrative geographic envelope is not an official city boundary; some scattered points may fall outside municipal land.",
                                       "Departments, real incidents, live municipal activity and achieved model metrics are not asserted."]},
        "actual_counts": {"train": counts(train), "validation": counts(validation), "tel_aviv_test": counts(test),
                          "tel_aviv_demo": counts(maps), "nyc_sample_raw_rows": len(raw)},
        "nyc_cleaning": {**source_checks,
                         "method": "Group identical primary descriptors regardless of location type; exclude conflicting source labels, generic details, unreviewed source/secondary families and held-out families. Keep one representative real detail per reviewed group.",
                         "reviewed_source_family_membership": SOURCE_FAMILIES},
        "split_checks": split_checks, "map_checks": map_checks,
        "demo": {"city_id": "il_tel_aviv", "map_center": [32.0853, 34.7818], "bbox": list(BBOX),
                 "fixture_manifest": fixture_manifest, "import_contract": {"status": "open", "routing_state": "routed", "category_source": "seed_fixture"},
                 "never_use_for_training_or_evaluation": True},
        "classifier_input": "text only; category, family_id, source, source_key and expected_status are metadata/targets, never features",
        "files_sha256": {name: digest_bytes(data) for name, data in sorted(contents.items())},
        "reproduction": {"python": "3.10+", "dependencies": "standard library only",
                         "offline": "python prepare_data.py",
                         "live": "python prepare_data.py --download-nyc",
                         "replay_real_snapshot": "python prepare_data.py --nyc-csv data/gpt_nyc_sample.csv --output-dir replay",
                         "determinism": "Offline files and provenance are byte-identical across runs. Live timing/status metadata and upstream data may differ; replay recreates CSV bytes from a retained real snapshot."}}
    (out/"gpt_data_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"mode": data_mode, "download_status": download["status"],
                      "counts": provenance["actual_counts"],
                      "five_training_examples": [next(r for r in synthetic if r["category"] == cat) for cat in CATEGORIES]+[useful[0] if useful else synthetic[5]],
                      "output_dir": str(out)}, indent=2))
    return 0

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_fetch":
        fetch_worker(sys.argv[2])
    else:
        raise SystemExit(main())
