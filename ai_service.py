"""AI drafts: reviewable text only. Never changes routing, counts or database state.

Providers (set AI_PROVIDER in .env):
  flan_t5   (default) local google/flan-t5-small via Hugging Face transformers - no API key.
            The small model writes only the human-sounding sentences; code fills in every fact
            (reference, department, fact IDs), so the model cannot invent a route or a deadline.
  anthropic Claude API (needs ANTHROPIC_API_KEY) - kept for later use.
  template  no AI; always the fixed template.

Validated model output -> "Live AI draft". Anything else -> "Template fallback".
Only approved facts are used; never raw complaint text, photos or contact details.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

LIVE = "Live AI draft"
FALLBACK = "Template fallback"
STATIC = "Static urgent notice"
TIMEOUT_S = 12
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_FLAN_MODEL = "google/flan-t5-small"

# Obvious promise/deadline/dispatch/completion claims -> reject draft
FORBIDDEN = re.compile(
    r"\b(within\s+\w+\s+(hours?|days?|weeks?)|by\s+(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|next\s+week)|deadline|guarantee[ds]?|will\s+be\s+(fixed|repaired|resolved|cleaned|"
    r"removed|dispatched|sent|handled|completed)|(has|have)\s+been\s+(fixed|repaired|resolved|dispatched|sent)|"
    r"dispatched|crew\s+(is|has|will)|on\s+(its|their)\s+way|we\s+will\s+(fix|repair|send|dispatch|resolve))\b",
    re.I)

TOPIC = {"roads": "road or sidewalk", "waste": "garbage and cleanliness", "lighting": "street light",
         "noise": "noise"}

_GENERATOR = None  # tests may inject a callable(prompt) -> str


def provider() -> str:
    return (os.getenv("AI_PROVIDER") or "flan_t5").strip().lower()


def flan_model_name() -> str:
    return os.getenv("FLAN_T5_MODEL") or DEFAULT_FLAN_MODEL


def model_name() -> str:
    return os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL


def has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def status_text() -> str:
    p = provider()
    if p == "flan_t5":
        return f"Local model {flan_model_name()} (no API key); falls back to template if unavailable"
    if p == "anthropic":
        return "Claude API · " + ("key configured" if has_credentials() else "no key → Template fallback")
    return "Template only (AI_PROVIDER=template)"


# ---------------------------------------------------------------- local FLAN-T5
@lru_cache(maxsize=1)
def _load_flan(name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # heavy import, only when used
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tok, model


def _generate(prompt: str, max_new_tokens: int = 40) -> str:
    if _GENERATOR is not None:
        return _GENERATOR(prompt)
    tok, model = _load_flan(flan_model_name())
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, no_repeat_ngram_size=3,
                         early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)


def _clean_sentence(raw: str, min_words: int = 3, max_words: int = 30) -> str:
    """Normalize one generated sentence and reject anything unsafe or unusable."""
    s = re.sub(r"\s+", " ", (raw or "")).strip().strip('"')
    words = s.split()
    if not (min_words <= len(words) <= max_words):
        raise ValueError(f"sentence length {len(words)} words")
    if re.search(r"\d", s) or "%" in s:
        raise ValueError("numbers in generated text")
    if FORBIDDEN.search(s):
        raise ValueError("promise/deadline/dispatch claim")
    if re.search(r"\(demo\)|department|municipality will|we will|i will", s, re.I):
        raise ValueError("names a department or makes a commitment")
    if s.lower().startswith(("write ", "rewrite ", "suggest ")):
        raise ValueError("echoed the instruction")
    if sum(c.isascii() for c in s) < 0.95 * len(s):
        raise ValueError("not plain English")
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!" else s + "."


# ---------------------------------------------------------------- Claude (optional)
ACK_SYSTEM = """ACKNOWLEDGMENT: Return JSON {"message":"..."}, at most 60 English words, using the
saved facts only. Say recorded in this demo. Name the proposed department only for
routed cases; otherwise say staff review is needed. Do not promise actual municipal
receipt, dispatch, repairs, completion, or deadlines. Urgent cases use the static notice.
Return only the JSON object."""

BRIEF_SYSTEM = """MANAGER BRIEF: Use Python-computed counts/hotspots for the selected city/window.
Return JSON with observations and recommended_actions arrays, at most three items each.
Each item has evidence_id and text; reference only supplied fact IDs. Leave numeric facts
to the displayed fact table; add no numbers/percentages in prose. Distinguish patterns
from hypotheses; suggest investigations. Mention simulated data. Promise no action.
Return only the JSON object."""


def _call(system: str, payload: dict, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0)
    msg = client.messages.create(model=model_name(), max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": json.dumps(payload)}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _parse_json(raw: str) -> dict:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = s.find("{"), s.rfind("}")
    return json.loads(s[start:end + 1])


# ---------------------------------------------------------------- acknowledgment
def ack_facts(report: dict) -> dict:
    """Approved facts only - no complaint text."""
    routed = report["routing_state"] == "routed"
    return {"reference": report["report_id"], "routing_state": report["routing_state"],
            "proposed_department": report["proposed_department"] if routed else None,
            "category": report.get("approved_category") if routed else None,
            "location_provided": report.get("location_status") == "valid"}


def _ack_facts_text(f: dict) -> str:
    if f["routing_state"] == "routed":
        return (f"Your report {f['reference']} was recorded in this demo and proposed for "
                f"{f['proposed_department']}. This is a demonstration only and was not forwarded to the municipality.")
    return (f"Your report {f['reference']} was recorded in this demo. Staff review is needed before a "
            f"department is proposed. This is a demonstration only and was not forwarded to the municipality.")


def ack_template(f: dict) -> str:
    return "Thank you. " + _ack_facts_text(f)


def validate_ack(obj, f: dict) -> str:
    if not isinstance(obj, dict) or set(obj) != {"message"} or not isinstance(obj["message"], str):
        raise ValueError("bad keys/types")
    m = obj["message"].strip()
    if not m or len(m.split()) > 60:
        raise ValueError("empty or over 60 words")
    if "demo" not in m.lower():
        raise ValueError("does not say recorded in this demo")
    if FORBIDDEN.search(m):
        raise ValueError("promise/deadline/dispatch claim")
    if f["routing_state"] != "routed" and "(demo)" in m:
        raise ValueError("names a department for a non-routed case")
    return m


TEMPLATE_SHORT = "Thank you — your report has been recorded."


def _flan_ack(f: dict) -> tuple[str, str]:
    if f["routing_state"] == "routed":
        prompt = (f"Write one short, polite sentence thanking a resident for reporting a "
                  f"{TOPIC.get(f['category'], 'street')} problem in their neighborhood.")
    else:
        prompt = ("Write one short, polite sentence thanking a resident for reporting a problem in their "
                  "neighborhood and saying a staff member will look at it.")
    courtesy = _clean_sentence(_generate(prompt), max_words=25)
    return validate_ack({"message": f"{courtesy} {_ack_facts_text(f)}"}, f), courtesy


def draft_acknowledgment(report: dict) -> dict:
    """Returns text (full acknowledgment), short (one line for the receipt), label and detail (diagnostics)."""
    from core import URGENT_NOTICE
    f = ack_facts(report)
    if f["routing_state"] == "urgent_review":
        return {"text": f"Your report {f['reference']} was recorded in this demo. {URGENT_NOTICE}",
                "short": "Your report has been recorded and flagged for urgent staff attention.",
                "label": STATIC, "detail": "Urgent cases always use the static notice."}
    p = provider()
    try:
        if p == "flan_t5":
            text, short = _flan_ack(f)
            return {"text": text, "short": short, "label": LIVE,
                    "detail": f"local {flan_model_name()} wrote the first sentence; facts filled by code"}
        if p == "anthropic":
            if not has_credentials():
                raise RuntimeError("No ANTHROPIC_API_KEY configured")
            msg = validate_ack(_parse_json(_call(ACK_SYSTEM, {"saved_facts": f}, 200)), f)
            return {"text": msg, "short": msg, "label": LIVE, "detail": f"Claude {model_name()}"}
        raise RuntimeError("AI_PROVIDER=template")
    except Exception as exc:
        return {"text": ack_template(f), "short": TEMPLATE_SHORT, "label": FALLBACK,
                "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


# ---------------------------------------------------------------- manager brief
def _hotspot_parts(fact: dict) -> tuple[str, str]:
    m = re.match(r"Hotspot: (\w+) cluster(?: near (.+?) \(approximate\))?", fact["fact"])
    cat, area = (m.group(1), m.group(2) or "") if m else ("", "")
    area = area.split("·")[-1].strip() if area else "one part of the city"
    return TOPIC.get(cat, cat or "street"), area


def brief_template(facts: list[dict]) -> dict:
    ids = {x["id"] for x in facts}
    hs = [x for x in facts if x["id"].startswith("H")]
    obs = [{"evidence_id": "F3", "text": "All figures come from simulated demo reports, not real municipal data."}]
    acts = []
    for h in hs[:2]:
        topic, area = _hotspot_parts(h)
        obs.append({"evidence_id": h["id"], "text": f"{topic.capitalize()} reports recur near {area}. This is a "
                                                    "pattern, not a confirmed single incident or cause."})
        acts.append({"evidence_id": h["id"], "text": f"Consider a site inspection near {area} to check whether these "
                                                     "reports describe one underlying problem."})
    if "F6" in ids:
        acts.append({"evidence_id": "F6", "text": "Review the staff-review queue and confirm or correct categories."})
    if not hs:
        obs.append({"evidence_id": "F8", "text": "No geographic hotspot met the clustering rule in this window."})
    return {"observations": obs[:3], "recommended_actions": acts[:3]}


def validate_brief(obj, facts: list[dict]) -> dict:
    ids = {x["id"] for x in facts}
    if not isinstance(obj, dict) or set(obj) != {"observations", "recommended_actions"}:
        raise ValueError("bad keys")
    all_text = []
    for key in ("observations", "recommended_actions"):
        items = obj[key]
        if not isinstance(items, list) or not 1 <= len(items) <= 3:
            raise ValueError(f"{key} must have 1-3 items")
        for it in items:
            if not isinstance(it, dict) or set(it) != {"evidence_id", "text"}:
                raise ValueError("item keys")
            if not isinstance(it["text"], str) or not isinstance(it["evidence_id"], str):
                raise ValueError("item types")
            if it["evidence_id"] not in ids:
                raise ValueError(f"unknown evidence_id {it['evidence_id']}")
            if len(it["text"]) > 300:
                raise ValueError("item too long")
            if re.search(r"\d", re.sub(r"\b[FH]\d+\b", "", it["text"])) or "%" in it["text"]:
                raise ValueError("numbers in prose")
            if FORBIDDEN.search(it["text"]):
                raise ValueError("promise/deadline/dispatch claim")
            all_text.append(it["text"].lower())
    if not any("simulat" in t for t in all_text):
        raise ValueError("does not mention simulated data")
    return obj


def _flan_brief(facts: list[dict]) -> tuple[dict, int, int]:
    """Template structure + evidence IDs from code; the local model rewrites hotspot sentences.
    Returns (brief, sentences_written_by_model, sentences_attempted)."""
    brief = brief_template(facts)
    hs = {h["id"]: h for h in facts if h["id"].startswith("H")}
    done = tried = 0
    last_err = ""
    for key, prompt_for in (
        ("observations", lambda t, a: f"Rewrite as one short sentence for a city manager, without numbers: "
                                      f"{t.capitalize()} complaints keep coming from near {a}."),
        ("recommended_actions", lambda t, a: f"Suggest one thing a city inspector could check about repeated "
                                             f"{t} complaints near {a}. Answer in one sentence."),
    ):
        for it in brief[key]:
            if it["evidence_id"] not in hs:
                continue  # data-provenance and queue items stay templated
            tried += 1
            topic, area = _hotspot_parts(hs[it["evidence_id"]])
            try:
                text = _clean_sentence(_generate(prompt_for(topic, area)), max_words=35)
                if key == "observations":
                    text += " This is a pattern to investigate, not a confirmed cause."
                it["text"] = text
                done += 1
            except Exception as exc:  # keep the template sentence for this item
                last_err = f"{type(exc).__name__}: {str(exc)[:100]}"
    if tried and not done:
        raise RuntimeError(f"local model produced no usable sentence ({last_err})")
    return validate_brief(brief, facts), done, tried


def generate_brief(facts: list[dict]) -> dict:
    p = provider()
    try:
        if p == "flan_t5":
            brief, done, tried = _flan_brief(facts)
            if done == 0:
                raise RuntimeError("no hotspot to describe")
            return {"brief": brief, "label": LIVE,
                    "detail": f"local {flan_model_name()} wrote {done} of {tried} hotspot sentences; "
                              "fact IDs and the rest filled by code"}
        if p == "anthropic":
            if not has_credentials():
                raise RuntimeError("No ANTHROPIC_API_KEY configured")
            obj = validate_brief(_parse_json(_call(BRIEF_SYSTEM, {"facts": facts}, 700)), facts)
            return {"brief": obj, "label": LIVE, "detail": f"Claude {model_name()}"}
        raise RuntimeError("AI_PROVIDER=template")
    except Exception as exc:
        return {"brief": brief_template(facts), "label": FALLBACK, "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


# ---------------------------------------------------------------- handover summary (Weekly Action Brief)
# Extra limits for handover text: no causes, trends, schedules, dispatch, promises or danger claims.
HANDOVER_FORBIDDEN = re.compile(
    r"\b(because|caus\w*|due to|result(s|ed)? (of|from)|increas\w*|ris(e|es|ing)|grow\w*|trend\w*|wors\w*|"
    r"schedul\w*|dispatch\w*|crew\w*|will|promis\w*|guarantee\w*|tomorrow|deadline\w*|danger\w*|unsafe|"
    r"emergenc\w*|hazard\w*)\b", re.I)
RANK_WORD = {1: "first", 2: "second", 3: "third"}


def _handover_sentence(raw: str) -> str:
    s = _clean_sentence(raw, min_words=4, max_words=30)
    if HANDOVER_FORBIDDEN.search(s):
        raise ValueError("cause/trend/schedule/dispatch/promise wording")
    return s


def handover_template(context: dict) -> list[dict]:
    items = []
    for p in context["priorities"]:
        items.append({"evidence_id": p["evidence_id"],
                      "why": (f"Ranked {RANK_WORD.get(p['rank'], str(p['rank']))} by active report count"
                              + (", the highest in this period." if p["rank"] == 1 else
                                 "; ties are broken by the oldest active report.")),
                      "verify": "Verify on site whether these reports describe one problem or several, and "
                                "link duplicates before assigning work."})
    return items


def generate_handover(context: dict) -> dict:
    """context: city, period, priorities [{evidence_id, rank, category, area, active_count, oldest_utc, department}],
    awaiting_review, oldest_active_days. No complaint text or attachments. Returns items keyed by evidence ID;
    numbers, dates, departments and ranking are added to the display by code, not by the model."""
    if not context.get("priorities"):
        return {"items": [], "label": FALLBACK, "detail": "no priority hotspot to summarise", "written": 0}
    p_ = provider()
    try:
        if p_ == "flan_t5":
            items, written, last_err = handover_template(context), 0, ""
            for it, pr in zip(items, context["priorities"]):
                topic, area = TOPIC.get(pr["category"], pr["category"]), (pr["area"] or "one part of the city").split("·")[-1].strip()
                for key, prompt in (
                    ("why", f"In one sentence, tell a city manager why repeated {topic} reports near {area} "
                            "deserve attention first. Do not mention numbers or causes."),
                    ("verify", f"In one sentence, suggest what city staff should check on site about repeated "
                               f"{topic} reports near {area}."),
                ):
                    try:
                        it[key] = _handover_sentence(_generate(prompt))
                        written += 1
                    except Exception as exc:
                        last_err = f"{type(exc).__name__}: {str(exc)[:100]}"
            if not written:
                raise RuntimeError(f"local model produced no usable sentence ({last_err})")
            return {"items": items, "label": LIVE, "written": written,
                    "detail": f"local {flan_model_name()} wrote {written} of {2 * len(items)} sentences; "
                              "counts, dates, departments and ranking added by code"}
        if p_ == "anthropic":
            if not has_credentials():
                raise RuntimeError("No ANTHROPIC_API_KEY configured")
            system = ("HANDOVER: Return JSON {\"items\":[{\"evidence_id\":\"...\",\"why\":\"...\",\"verify\":\"...\"}]}"
                      ", one item per supplied priority, same evidence_id values. 'why' explains in one sentence why "
                      "the item is on the priority list; 'verify' says what staff should check next. No numbers, "
                      "dates, departments, causes, trends, schedules, dispatch or promises.")
            obj = _parse_json(_call(system, context, 600))
            ids = [p["evidence_id"] for p in context["priorities"]]
            got = {i["evidence_id"]: i for i in obj.get("items", []) if isinstance(i, dict)}
            if set(got) != set(ids):
                raise ValueError("evidence IDs don't match the priorities")
            items = [{"evidence_id": i, "why": _handover_sentence(got[i]["why"]),
                      "verify": _handover_sentence(got[i]["verify"])} for i in ids]
            return {"items": items, "label": LIVE, "written": 2 * len(items), "detail": f"Claude {model_name()}"}
        raise RuntimeError("AI_PROVIDER=template")
    except Exception as exc:
        return {"items": handover_template(context), "label": FALLBACK, "written": 0,
                "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def check_access() -> str:
    """Provider check; downloads FLAN-T5 on first run. Never prints keys."""
    p = provider()
    if p == "flan_t5":
        try:
            sample = _generate("Write one short, polite sentence thanking a resident for reporting a street "
                               "light problem in their neighborhood.")
            return f"Local model {flan_model_name()} works. Sample output: {sample!r}"
        except Exception as exc:
            return f"Local model not available ({type(exc).__name__}: {str(exc)[:160]}) - drafts will use Template fallback."
    if p == "anthropic":
        if not has_credentials():
            return "No ANTHROPIC_API_KEY in environment/.env - drafts will use Template fallback."
        try:
            _call("Reply with the JSON {\"ok\":true} only.", {"ping": True}, 20)
            return f"Provider reachable with model {model_name()}."
        except Exception as exc:
            return f"Provider check failed: {type(exc).__name__}: {str(exc)[:120]}"
    return "AI_PROVIDER=template - no AI calls."


if __name__ == "__main__":
    print(status_text())
    print(check_access())
