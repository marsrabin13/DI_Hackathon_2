"""Prepare CityFix data files.

Outputs (all in data/):
  nyc_sample.csv        actual downloaded NYC 311 rows (only if the API is reachable)
  train.csv             deduplicated NYC details (if any) + 240 generated English narratives
  validation.csv        24 independently written synthetic cases (6/category)
  tel_aviv_test.csv     20 in-scope synthetic cases (5/category) + 4 policy cases
  tel_aviv_demo.csv     90 synthetic historical map reports (never used for training/eval)
  data_provenance.json  sources, counts, download status, seed, split checks

Usage:
  python prepare_data.py             # tries NYC API (bounded, <=3 min), else synthetic-only
  python prepare_data.py --offline   # skip the NYC API entirely
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
SEED = 42
CATEGORIES = ["roads", "waste", "lighting", "noise"]
NYC_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
NYC_LABELS = {
    "roads": ["Street Condition"],
    "waste": ["Dirty Condition"],
    "lighting": ["Street Light Condition"],
    "noise": ["Noise - Residential", "Noise - Street/Sidewalk"],
}
DEMO_ANCHOR_UTC = "2026-10-28T18:00:00Z"  # default manager-window end


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(text).lower())).strip()


# ---------------------------------------------------------------------------
# 1. NYC 311 bounded download (structured details only)
# ---------------------------------------------------------------------------
def fetch_nyc(budget_seconds: int = 180, per_call_timeout: int = 15):
    started = time.time()
    rows, status = [], {"attempted": True, "queries": []}
    for target, labels in NYC_LABELS.items():
        if time.time() - started > budget_seconds:
            status["queries"].append({"target": target, "error": "time budget exhausted"})
            continue
        label_list = ",".join("'" + l.replace("'", "''") + "'" for l in labels)
        params = {
            "$select": "unique_key,complaint_type,descriptor,descriptor_2",
            "$where": f"complaint_type in ({label_list}) AND descriptor IS NOT NULL",
            "$limit": "150",
        }
        url = NYC_ENDPOINT + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=per_call_timeout) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
            for r in batch:
                r["_target"] = target
            rows.extend(batch)
            status["queries"].append({"target": target, "rows": len(batch)})
        except Exception as exc:  # network blocked, timeout, HTTP error
            status["queries"].append({"target": target, "error": type(exc).__name__ + ": " + str(exc)[:120]})
    status["rows_downloaded"] = len(rows)
    status["succeeded"] = len(rows) > 0
    return rows, status


def nyc_to_training(rows):
    """Map, deduplicate and exclude conflicting descriptors. Text only."""
    if not rows:
        return pd.DataFrame(columns=["text", "category", "family_id", "source", "source_key"]), {}
    df = pd.DataFrame(rows)
    label_to_cat = {l: c for c, ls in NYC_LABELS.items() for l in ls}
    df["category"] = df["complaint_type"].map(label_to_cat)
    unmapped = int(df["category"].isna().sum())
    df = df.dropna(subset=["category", "descriptor"])
    df["text"] = (df["descriptor"].fillna("") + " " + df.get("descriptor_2", pd.Series("", index=df.index)).fillna("")).str.strip()
    df["key"] = df["text"].map(norm)
    cats_per_key = df.groupby("key")["category"].nunique()
    conflicting = set(cats_per_key[cats_per_key > 1].index)
    df = df[~df["key"].isin(conflicting)]
    dedup = df.drop_duplicates("key")
    out = pd.DataFrame({
        "text": dedup["text"],
        "category": dedup["category"],
        "family_id": "nyc_" + dedup["key"].map(lambda k: hashlib.md5(k.encode()).hexdigest()[:8]),
        "source": "nyc_structured",
        "source_key": dedup["unique_key"].astype(str),
    })
    stats = {"unmapped_rows_excluded": unmapped, "conflicting_descriptors_excluded": len(conflicting),
             "unique_details_kept": len(out), "rows_before_dedup": len(df)}
    return out, stats


# ---------------------------------------------------------------------------
# 2. Generated training narratives: 12 families/category x 5 variants
#    Each family = one scenario; variants differ in framing and location.
# ---------------------------------------------------------------------------
TRAIN_FAMILIES = {
    "roads": [
        ("a deep pothole in the middle of the road", "Cars keep swerving to avoid it."),
        ("a long crack running across the asphalt", "It is getting wider every week."),
        ("a manhole cover that is loose and rattles when cars drive over it", "It is loud and looks dangerous."),
        ("a sunken patch of road where the utility work was done", "The repair was never finished properly."),
        ("a broken curb stone sticking out into the street", "Strollers can't get past it."),
        ("raised sidewalk slabs that people trip over", "An elderly neighbor fell there."),
        ("pedestrian crossing paint that has completely faded", "Drivers don't see the crosswalk."),
        ("loose gravel left on the road after repaving", "Scooters are skidding on it."),
        ("a very bumpy and uneven road surface", "It damages car tires."),
        ("a speed bump that is broken and crumbling", "Pieces of asphalt are all over the lane."),
        ("the edge of the road collapsing next to the drain", "There is a hole forming under the asphalt."),
        ("broken paving stones on the bike path", "Cyclists have to ride into traffic."),
    ],
    "waste": [
        ("garbage bins overflowing with trash", "The bins have not been emptied for days."),
        ("trash bags piled up on the sidewalk", "Cats are tearing them open."),
        ("old furniture illegally dumped by the entrance", "Someone left a sofa and two chairs."),
        ("broken glass and litter all over the playground", "Kids can't play there safely."),
        ("a recycling container full and not emptied", "Bottles are piling up around it."),
        ("dog poop that nobody cleans up on the pavement", "The whole stretch smells."),
        ("construction debris dumped on the empty lot", "Bags of cement and bricks were left behind."),
        ("a dumpster with rotting food that smells terrible", "Flies are everywhere."),
        ("plastic bottles and wrappers littering the promenade path", "The litter blows onto the beach."),
        ("a mess of garbage spilled from the collection truck", "Nobody came back to clean it."),
        ("piles of rubbish left after the street market", "Cardboard boxes and rotten vegetables are everywhere."),
        ("an abandoned mattress dumped next to the bins", "It has been sitting there for a week."),
    ],
    "lighting": [
        ("a streetlight that is not working", "The sidewalk is completely dark at night."),
        ("a street lamp that keeps flickering on and off", "It is very distracting for drivers."),
        ("several lamps out so the whole street is dark", "People feel unsafe walking home."),
        ("a street light that stays on all day", "It is wasting electricity."),
        ("a lamp fixture with shattered glass", "Pieces of the cover fell on the ground."),
        ("a light pole leaning to one side", "It looks like it might fall over."),
        ("the path lights in the park not turning on", "Joggers can't see where they are going."),
        ("the light above the crosswalk burned out", "Drivers can't see pedestrians at night."),
        ("a street lamp that is extremely dim", "It barely lights up anything."),
        ("a lamp that buzzes and blinks all night", "The bulb seems to be failing."),
        ("the lights in the pedestrian underpass are out", "It is pitch black inside."),
        ("a lamp cover hanging loose from the pole", "It swings in the wind."),
    ],
    "noise": [
        ("loud music from a neighbor's apartment late at night", "It goes on past 2 am."),
        ("construction noise starting very early in the morning", "Drilling starts before 6 am."),
        ("a bar playing loud music until late", "The bass shakes our windows."),
        ("a car alarm going off again and again", "Nobody turns it off."),
        ("a dog barking all night long", "We can't sleep at all."),
        ("a loud party on the rooftop next door", "People are screaming and the music is blasting."),
        ("motorbikes revving their engines loudly", "They do it every evening."),
        ("a street performer with a very loud amplifier", "It goes on for hours."),
        ("a generator humming loudly day and night", "The noise never stops."),
        ("groups shouting outside the kiosk late at night", "It wakes up the whole building."),
        ("a leaf blower running very early in the morning", "It is incredibly loud."),
        ("a noisy air conditioning unit rattling on the wall", "The rattling is constant."),
    ],
}
LOCATIONS = ["on our street", "near the bus stop", "outside building 14",
             "at the corner by the school", "next to the park entrance"]
FRAMES = [
    "There is {issue} {loc}. {impact}",
    "Hi, I want to report {issue} {loc}.",
    "{Loc}, there is {issue}. {impact} Please look into it.",
    "For several days now there has been {issue} {loc}. {impact}",
    "Complaint: {issue} {loc}. {impact_l}",
]


def build_train_synthetic():
    rows = []
    for cat, fams in TRAIN_FAMILIES.items():
        for fi, (issue, impact) in enumerate(fams, start=1):
            for vi, frame in enumerate(FRAMES):
                loc = LOCATIONS[(fi + vi) % len(LOCATIONS)]
                text = frame.format(issue=issue, loc=loc, Loc=loc[0].upper() + loc[1:],
                                    impact=impact, impact_l=impact[0].lower() + impact[1:])
                rows.append({"text": text, "category": cat, "family_id": f"tr_{cat}_{fi:02d}",
                             "source": "synthetic", "source_key": f"tr_{cat}_{fi:02d}_v{vi+1}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Validation: independently written, different scenario families
# ---------------------------------------------------------------------------
VALIDATION = {
    "roads": [
        "My scooter wheel got stuck in a gap in the asphalt on Herzl Street.",
        "The lane markings at the roundabout are gone, drivers are confused about where to go.",
        "Tree roots have pushed the pavement up and it is cracked all along the block.",
        "The drain grate has sunk below the road level and cars bang into it.",
        "A steel plate left on the road after the works moves and clanks under every bus.",
        "The road has deep ruts from the heavy trucks and water collects in them.",
    ],
    "waste": [
        "Someone emptied a whole bag of household rubbish on the bench in the square.",
        "The bins behind the restaurant are always full and spilling onto the alley.",
        "Used diapers and food scraps are thrown over the fence into the shared yard.",
        "Nobody collected the green waste after the gardeners cut the hedges.",
        "There's an old fridge and a broken TV dumped near the parking entrance.",
        "Cigarette butts and cans cover the area around the bus shelter.",
    ],
    "lighting": [
        "The lamp outside the kindergarten has been dark for two weeks.",
        "Half the bulbs on the boulevard went out after the storm.",
        "The street light near my building turns off around midnight and never comes back on.",
        "One of the poles in the parking lot has no light at all.",
        "The lantern on the stairs down to the beach is broken.",
        "Every lamp in our alley is off, we use phone flashlights to get home.",
    ],
    "noise": [
        "The neighbors upstairs have been banging and stomping until three in the morning.",
        "A club downstairs blasts techno every weekend night.",
        "Workers are jackhammering on a Saturday morning next to our house.",
        "Someone plays drums in the building every night after eleven.",
        "Delivery trucks unload with loud beeping and shouting at 4 am.",
        "There's a karaoke event in the yard with speakers turned all the way up.",
    ],
}

# ---------------------------------------------------------------------------
# 4. Final test: independent families + 4 policy cases (category blank)
# ---------------------------------------------------------------------------
TEST = {
    "roads": [
        "There is a big hole in the pavement right in front of the pharmacy on Dizengoff.",
        "The asphalt on the bridge ramp is peeling off in large chunks.",
        "A section of the sidewalk collapsed and there's a drop of about 20 cm.",
        "Cobblestones in the old Jaffa alley are missing and people keep tripping.",
        "The road was dug up for pipes and left with a rough unpaved strip.",
    ],
    "waste": [
        "The trash has not been picked up on our street since last Sunday.",
        "People leave bags of garbage next to the full container every evening.",
        "Someone dumped building waste and tiles in the green area by the school.",
        "The public garbage can on the corner is overflowing with fast food wrappers.",
        "After the weekend the beach parking lot is covered in litter and bottles.",
    ],
    "lighting": [
        "The light post at the end of Ben Yehuda is not lighting at night.",
        "Two streetlamps on our block have been flickering for a week.",
        "The street is dark because the lamps were never switched back on after the repairs.",
        "A lamp in the playground is broken and the area is completely dark.",
        "The light at the pedestrian bridge keeps going out.",
    ],
    "noise": [
        "The pub across the street has loud music and shouting every night until 3.",
        "Building renovations next door start at 6:30 with hammering and drilling.",
        "My neighbor's dog howls for hours whenever they leave.",
        "Motorcycles race up and down the street at night making a terrible racket.",
        "An event hall nearby has extremely loud speakers on weeknights.",
    ],
}
TEST_POLICY = [
    ("Something is wrong here, please come and check.", "review_required", "policy_vague"),
    ("I need to renew my passport, which office should I go to?", "review_required", "policy_unsupported"),
    ("The streetlight is broken, there is garbage everywhere and the neighbors play loud music.",
     "review_required", "policy_multi_issue"),
    ("There is an exposed live wire hanging from the lamp post next to the playground.",
     "urgent_review", "policy_urgent"),
]

# ---------------------------------------------------------------------------
# 5. Map fixtures (seed 42) - illustrative points only
# ---------------------------------------------------------------------------
DEMO_TEXTS = {
    "roads": ["Pothole getting bigger near the junction.", "Cracked road surface by the crossing.",
              "Broken pavement outside the shops."],
    "waste": ["Overflowing bins on the corner.", "Rubbish bags left on the pavement.",
              "Litter piling up near the benches."],
    "lighting": ["Streetlight out on this block.", "Lamp flickering all night.", "Dark stretch, lamps not working."],
    "noise": ["Loud music late at night.", "Construction noise early morning.", "Shouting outside at night."],
}


def jitter(rng, lat, lon, max_m=60):
    r = max_m * math.sqrt(rng.random())
    theta = rng.random() * 2 * math.pi
    dlat = (r * math.cos(theta)) / 111_320
    dlon = (r * math.sin(theta)) / (111_320 * math.cos(math.radians(lat)))
    return round(lat + dlat, 6), round(lon + dlon, 6)


def build_demo():
    rng = random.Random(SEED)
    anchor = datetime.strptime(DEMO_ANCHOR_UTC, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    bbox = [32.04, 34.74, 32.15, 34.84]
    planted = [("roads", 12, (32.068, 34.774)), ("waste", 10, (32.080, 34.782)), ("lighting", 8, (32.103, 34.792))]
    rows, n = [], 0

    def ts():
        return (anchor - timedelta(seconds=rng.randint(0, 7 * 24 * 3600 - 60))).strftime("%Y-%m-%dT%H:%M:%SZ")

    for cat, count, (lat, lon) in planted:
        for _ in range(count):
            n += 1
            la, lo = jitter(rng, lat, lon)
            rows.append({"report_id": f"DEMO-{n:03d}", "city_id": "il_tel_aviv", "text": rng.choice(DEMO_TEXTS[cat]),
                         "category": cat, "latitude": la, "longitude": lo, "created_at_utc": ts(),
                         "source": "synthetic_demo"})
    for _ in range(60):
        n += 1
        cat = rng.choice(CATEGORIES)
        la = round(rng.uniform(bbox[0] + 0.005, bbox[2] - 0.005), 6)
        lo = round(rng.uniform(bbox[1] + 0.012, bbox[3] - 0.005), 6)
        rows.append({"report_id": f"DEMO-{n:03d}", "city_id": "il_tel_aviv", "text": rng.choice(DEMO_TEXTS[cat]),
                     "category": cat, "latitude": la, "longitude": lo, "created_at_utc": ts(),
                     "source": "synthetic_demo"})
    return pd.DataFrame(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip NYC API")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    if args.offline:
        nyc_rows, nyc_status = [], {"attempted": False, "succeeded": False, "reason": "--offline"}
    else:
        print("Trying NYC 311 API (bounded: 4 queries x <=150 rows, 3 min budget)...")
        nyc_rows, nyc_status = fetch_nyc()
    nyc_train, nyc_stats = nyc_to_training(nyc_rows)
    if nyc_rows:
        pd.DataFrame(nyc_rows).drop(columns=["_target"]).to_csv(DATA / "nyc_sample.csv", index=False)
        print(f"NYC: downloaded {len(nyc_rows)} rows, kept {len(nyc_train)} unique details")
    else:
        print("NYC API unavailable -> synthetic-only fallback (no NYC records invented).")

    train = pd.concat([nyc_train, build_train_synthetic()], ignore_index=True)
    val = pd.DataFrame([{"text": t, "category": c, "family_id": f"val_{c}_{i+1:02d}", "source": "synthetic",
                         "source_key": f"val_{c}_{i+1:02d}"} for c, ts_ in VALIDATION.items() for i, t in enumerate(ts_)])
    test = pd.DataFrame(
        [{"text": t, "category": c, "family_id": f"test_{c}_{i+1:02d}", "source": "synthetic",
          "source_key": f"test_{c}_{i+1:02d}", "expected_status": "routed"} for c, ts_ in TEST.items() for i, t in enumerate(ts_)]
        + [{"text": t, "category": "", "family_id": f"test_{k}", "source": "synthetic", "source_key": f"test_{k}",
            "expected_status": s} for t, s, k in TEST_POLICY])
    demo = build_demo()

    # ---- split isolation checks
    def fam(df): return set(df["family_id"])
    def txt(df): return set(df["text"].map(norm))
    checks = {
        "train_val_family_overlap": len(fam(train) & fam(val)),
        "train_test_family_overlap": len(fam(train) & fam(test)),
        "val_test_family_overlap": len(fam(val) & fam(test)),
        "train_val_text_overlap": len(txt(train) & txt(val)),
        "train_test_text_overlap": len(txt(train) & txt(test)),
        "val_test_text_overlap": len(txt(val) & txt(test)),
        "demo_vs_supervised_text_overlap": len(txt(demo) & (txt(train) | txt(val) | txt(test))),
        "train_duplicate_texts": int(train["text"].map(norm).duplicated().sum()),
    }
    assert all(v == 0 for v in checks.values()), checks

    train.to_csv(DATA / "train.csv", index=False)
    val.to_csv(DATA / "validation.csv", index=False)
    test.to_csv(DATA / "tel_aviv_test.csv", index=False)
    demo.to_csv(DATA / "tel_aviv_demo.csv", index=False)

    prov = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "nyc_download": {**nyc_status, **nyc_stats, "endpoint": NYC_ENDPOINT,
                         "fields": "unique_key,complaint_type,descriptor,descriptor_2",
                         "label_map": NYC_LABELS},
        "mode": "nyc_plus_synthetic" if len(nyc_train) else "synthetic_only",
        "generation_method": {
            "train": "Template generation: 12 hand-written scenario families per category x 5 framing/location "
                     "variants (all variants of a family stay in train). Plus deduplicated NYC descriptor text if downloaded.",
            "validation": "24 hand-written sentences, 6/category, scenario families not used in train.",
            "test": "20 hand-written in-scope sentences (5/category) + 4 policy cases, separate families; "
                    "untouched until train.py --evaluate-test.",
            "demo": "90 simulated Tel Aviv map reports (seed 42): 12 roads near 32.068,34.774; 10 waste near "
                    "32.080,34.782; 8 lighting near 32.103,34.792 (jitter <=60 m); 60 scattered. "
                    "Illustrative points, not verified incident locations. Never used for training/evaluation.",
        },
        "demo_anchor_utc": DEMO_ANCHOR_UTC,
        "counts": {
            "train": {"total": len(train), **train.groupby("source").size().to_dict(),
                      "by_category": train["category"].value_counts().to_dict(),
                      "families": int(train["family_id"].nunique())},
            "validation": {"total": len(val), "by_category": val["category"].value_counts().to_dict()},
            "test": {"total": len(test), "in_scope": int((test["category"] != "").sum()), "policy_cases": len(TEST_POLICY)},
            "demo": {"total": len(demo), "by_category": demo["category"].value_counts().to_dict()},
        },
        "split_checks": checks,
        "file_hashes": {f: file_hash(DATA / f) for f in ["train.csv", "validation.csv", "tel_aviv_test.csv", "tel_aviv_demo.csv"]},
        "caveats": [
            "Synthetic evaluations share generation bias with training data and do not establish real-city accuracy.",
            "Raw NYC records stay NYC records; no NYC coordinates are moved to Tel Aviv.",
            "Classifier input is text only (no complaint_type, agency, category code or city metadata).",
        ],
    }
    (DATA / "data_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"mode": prov["mode"], "counts": prov["counts"], "split_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
