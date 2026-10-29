"""Claude adapter: reviewable drafts only. Never changes routing, counts or database state.

Live call -> validated JSON -> "Live AI draft". Anything else -> "Template fallback".
Only approved facts are sent; never raw complaint text or contact details.
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

LIVE = "Live AI draft"
FALLBACK = "Template fallback"
STATIC = "Static urgent notice"
TIMEOUT_S = 12
DEFAULT_MODEL = "claude-sonnet-5-5"

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

# Obvious promise/deadline/dispatch/completion claims -> reject draft
FORBIDDEN = re.compile(
    r"\b(within\s+\w+\s+(hours?|days?|weeks?)|by\s+(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|next\s+week)|deadline|guarantee[ds]?|will\s+be\s+(fixed|repaired|resolved|cleaned|"
    r"removed|dispatched|sent|handled|completed)|(has|have)\s+been\s+(fixed|repaired|resolved|dispatched|sent)|"
    r"dispatched|crew\s+(is|has|will)|on\s+(its|their)\s+way|we\s+will\s+(fix|repair|send|dispatch|resolve))\b",
    re.I)


def model_name() -> str:
    return os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL


def has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _call(system: str, payload: dict, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0)
    msg = client.messages.create(model=model_name(), max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": json.dumps(payload)}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
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


def ack_template(f: dict) -> str:
    if f["routing_state"] == "routed":
        return (f"Thank you. Your report {f['reference']} was recorded in this demo and proposed for "
                f"{f['proposed_department']}. This is a demonstration only and was not forwarded to the municipality.")
    return (f"Thank you. Your report {f['reference']} was recorded in this demo. Staff review is needed before a "
            f"department is proposed. This is a demonstration only and was not forwarded to the municipality.")


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


def draft_acknowledgment(report: dict) -> dict:
    from core import URGENT_NOTICE
    f = ack_facts(report)
    if f["routing_state"] == "urgent_review":
        return {"text": f"Your report {f['reference']} was recorded in this demo. {URGENT_NOTICE}",
                "label": STATIC, "detail": "Urgent cases always use the static notice."}
    if not has_credentials():
        return {"text": ack_template(f), "label": FALLBACK, "detail": "No ANTHROPIC_API_KEY configured."}
    try:
        msg = validate_ack(_parse_json(_call(ACK_SYSTEM, {"saved_facts": f}, 200)), f)
        return {"text": msg, "label": LIVE, "detail": f"model={model_name()}"}
    except Exception as exc:
        return {"text": ack_template(f), "label": FALLBACK, "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


# ---------------------------------------------------------------- manager brief
def brief_template(facts: list[dict]) -> dict:
    ids = {x["id"] for x in facts}
    hs = [x for x in facts if x["id"].startswith("H")]
    obs = [{"evidence_id": "F3", "text": "All figures come from simulated Tel Aviv demo reports, not real municipal data."}]
    acts = []
    for h in hs[:2]:
        obs.append({"evidence_id": h["id"], "text": f"A recurring geographic pattern appears: {h['fact']}. "
                                                    "This is a pattern, not a confirmed single incident or cause."})
        acts.append({"evidence_id": h["id"], "text": "Consider a site inspection to check whether these reports "
                                                     "describe one underlying problem."})
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


def generate_brief(facts: list[dict]) -> dict:
    if not has_credentials():
        return {"brief": brief_template(facts), "label": FALLBACK, "detail": "No ANTHROPIC_API_KEY configured."}
    try:
        obj = validate_brief(_parse_json(_call(BRIEF_SYSTEM, {"facts": facts}, 700)), facts)
        return {"brief": obj, "label": LIVE, "detail": f"model={model_name()}"}
    except Exception as exc:
        return {"brief": brief_template(facts), "label": FALLBACK, "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def check_access() -> str:
    """Provider access check; never prints the key."""
    if not has_credentials():
        return "No ANTHROPIC_API_KEY in environment/.env - drafts will use Template fallback."
    try:
        _call("Reply with the JSON {\"ok\":true} only.", {"ping": True}, 20)
        return f"Provider reachable with model {model_name()}."
    except Exception as exc:
        return f"Provider check failed: {type(exc).__name__}: {str(exc)[:120]}"


if __name__ == "__main__":
    print(check_access())
