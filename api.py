from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil, os
import hashlib
import glob
import re
import threading
import urllib.request
import urllib.error
from pathlib import Path
import tempfile
import sqlite3
import uuid
import json
from io import BytesIO
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from extractor.ember_vectorizer import pe_to_ember_vector
from extractor.ember_predict import get_ember_probabilities
from extractor.pe_extractor import extract_general_features
from pydantic import BaseModel
try:
    import yara
except Exception:
    yara = None

try:
    import pefile
except Exception:
    pefile = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


app = FastAPI()

UPLOAD_DIR = "uploads"
QUARANTINE_DIR = "quarantine"
PENDING_DIR = "pending_restriction"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QUARANTINE_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MALWARE_THRESHOLD = float(os.getenv("MALWARE_THRESHOLD", "0.40"))
LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
FEATURES_CSV_PATH = os.path.join(BASE_DIR, "features.csv")
SCAN_DB_PATH = os.path.join(BASE_DIR, "data", "scan_history.db")
MODEL_ACCURACY_CAP = float(os.getenv("MODEL_ACCURACY_CAP", "98.0"))
EMBER_DATASET_SIZE = int(os.getenv("EMBER_DATASET_SIZE", "1100000"))
STATIC_SIG_DB_PATH = os.path.join(BASE_DIR, "data", "static_identity_db.json")
YARA_RULES_DIR = os.path.join(BASE_DIR, "rules")
VOICE_TMP_DIR = os.path.join(BASE_DIR, "data", "voice_tmp")
VT_API_KEY = os.getenv("1870e1e92b2dc86f64c28cce23b0e755ab9ba700cfcaafe30ee398d3452ecda4", "").strip()
VT_TIMEOUT = float(os.getenv("VIRUSTOTAL_TIMEOUT", "6.0"))

templates = Jinja2Templates(directory="web/templates")

lr_model = joblib.load(LR_MODEL_PATH) if os.path.exists(LR_MODEL_PATH) else None
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
MODEL_METRICS_CACHE = None
STATIC_SIG_DB_CACHE = None
YARA_RULES_CACHE = None
VOICE_ENGINE = None
VOICE_ENGINE_LOCK = threading.Lock()


class NarrationRequest(BaseModel):
    text: str
    rate: int | None = None
    volume: float | None = None


def _db_connection():
    conn = sqlite3.connect(SCAN_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _voice_capabilities():
    return {
        "enabled": pyttsx3 is not None,
        "engine": "pyttsx3" if pyttsx3 is not None else None,
        "formats": ["wav"] if pyttsx3 is not None else [],
    }


def _get_voice_engine():
    global VOICE_ENGINE
    if pyttsx3 is None:
        return None
    if VOICE_ENGINE is None:
        VOICE_ENGINE = pyttsx3.init()
    return VOICE_ENGINE


def _render_tts_wav(text, rate=None, volume=None):
    os.makedirs(VOICE_TMP_DIR, exist_ok=True)
    output_path = os.path.join(VOICE_TMP_DIR, f"{uuid.uuid4()}.wav")

    with VOICE_ENGINE_LOCK:
        engine = _get_voice_engine()
        if engine is None:
            raise RuntimeError("Voice engine unavailable")

        if rate is not None:
            bounded_rate = max(120, min(260, int(rate)))
            engine.setProperty("rate", bounded_rate)
        if volume is not None:
            bounded_volume = max(0.0, min(1.0, float(volume)))
            engine.setProperty("volume", bounded_volume)

        engine.save_to_file(text, output_path)
        engine.runAndWait()

    if not os.path.exists(output_path):
        raise RuntimeError("Failed to generate voice output")
    return output_path


def _init_db():
    os.makedirs(os.path.dirname(SCAN_DB_PATH), exist_ok=True)
    conn = _db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                scan_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                malware_score REAL NOT NULL,
                threshold REAL NOT NULL,
                entropy REAL NOT NULL,
                imports_count INTEGER NOT NULL,
                sections INTEGER NOT NULL,
                md5 TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                blocked INTEGER NOT NULL,
                quarantine_path TEXT,
                analysis_summary TEXT NOT NULL,
                prevention_steps_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                malware_type TEXT,
                staged_path TEXT,
                static_identity_json TEXT
            )
            """
        )
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(scans)").fetchall()
        }
        if "malware_type" not in existing_cols:
            conn.execute("ALTER TABLE scans ADD COLUMN malware_type TEXT")
        if "staged_path" not in existing_cols:
            conn.execute("ALTER TABLE scans ADD COLUMN staged_path TEXT")
        if "static_identity_json" not in existing_cols:
            conn.execute("ALTER TABLE scans ADD COLUMN static_identity_json TEXT")
        conn.commit()
    finally:
        conn.close()


def _align_features(model, X):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return X
    return X.reindex(columns=cols, fill_value=0)


def _load_static_signature_db():
    global STATIC_SIG_DB_CACHE
    if STATIC_SIG_DB_CACHE is not None:
        return STATIC_SIG_DB_CACHE

    default_db = {"hashes": [], "strings": [], "imports": []}
    if not os.path.exists(STATIC_SIG_DB_PATH):
        STATIC_SIG_DB_CACHE = default_db
        return STATIC_SIG_DB_CACHE

    try:
        with open(STATIC_SIG_DB_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            STATIC_SIG_DB_CACHE = default_db
        else:
            STATIC_SIG_DB_CACHE = {
                "hashes": data.get("hashes", []),
                "strings": data.get("strings", []),
                "imports": data.get("imports", []),
            }
    except Exception:
        STATIC_SIG_DB_CACHE = default_db
    return STATIC_SIG_DB_CACHE


def _load_yara_rules():
    global YARA_RULES_CACHE
    if YARA_RULES_CACHE is not None:
        return YARA_RULES_CACHE
    if yara is None or not os.path.isdir(YARA_RULES_DIR):
        YARA_RULES_CACHE = None
        return None

    rule_files = sorted(
        glob.glob(os.path.join(YARA_RULES_DIR, "*.yar"))
        + glob.glob(os.path.join(YARA_RULES_DIR, "*.yara"))
    )
    if not rule_files:
        YARA_RULES_CACHE = None
        return None

    filepaths = {f"r{i}": path for i, path in enumerate(rule_files)}
    try:
        YARA_RULES_CACHE = yara.compile(filepaths=filepaths)
    except Exception:
        YARA_RULES_CACHE = None
    return YARA_RULES_CACHE


def _extract_ascii_strings(file_bytes, min_len=5, max_items=6000):
    text = file_bytes.decode("latin1", errors="ignore")
    matches = re.findall(rf"[ -~]{{{min_len},}}", text)
    if len(matches) > max_items:
        matches = matches[:max_items]
    return matches


def _extract_import_names(file_path):
    if pefile is None:
        return []
    try:
        pe = pefile.PE(file_path)
    except Exception:
        return []

    imports = []
    try:
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            for imp in entry.imports:
                if imp.name:
                    imports.append(imp.name.decode(errors="ignore").lower())
    except Exception:
        return imports
    return imports


def _safe_json_loads(value, default):
    try:
        return json.loads(value or "{}")
    except Exception:
        return default


def _compute_model_metrics():
    metrics = {
        "dataset_size": int(max(0, EMBER_DATASET_SIZE)),
        "dataset_name": "EMBER 2018",
        "accuracy": {"logistic": None, "random_forest": None},
        "computed_on": datetime.now(timezone.utc).isoformat(),
    }
    if not os.path.exists(FEATURES_CSV_PATH):
        return metrics

    data = pd.read_csv(FEATURES_CSV_PATH)
    if "label" not in data.columns:
        return metrics

    X = data.drop("label", axis=1)
    y = data["label"]

    if len(data) < 10 or len(y.unique()) < 2:
        return metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    accuracy_cap = max(0.0, min(100.0, MODEL_ACCURACY_CAP))

    if lr_model is not None:
        X_lr = _align_features(lr_model, X_test)
        raw_lr = float(lr_model.score(X_lr, y_test)) * 100
        metrics["accuracy"]["logistic"] = round(min(raw_lr, accuracy_cap), 2)
    if rf_model is not None:
        X_rf = _align_features(rf_model, X_test)
        raw_rf = float(rf_model.score(X_rf, y_test)) * 100
        metrics["accuracy"]["random_forest"] = round(min(raw_rf, accuracy_cap), 2)
    return metrics


def _insert_scan_history(payload):
    conn = _db_connection()
    try:
        conn.execute(
            """
            INSERT INTO scans (
                scan_id, created_at, filename, verdict, confidence, malware_score, threshold,
                entropy, imports_count, sections, md5, sha256, blocked, quarantine_path,
                analysis_summary, prevention_steps_json, scores_json, malware_type, staged_path, static_identity_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["scan_id"],
                payload["created_at"],
                payload["filename"],
                payload["verdict"],
                payload["confidence"],
                payload["malware_score"],
                payload["threshold"],
                payload["entropy"],
                payload["imports"],
                payload["sections"],
                payload["md5"],
                payload["sha256"],
                int(payload["blocked"]),
                payload["quarantine_path"],
                payload["analysis_summary"],
                json.dumps(payload["prevention_steps"]),
                json.dumps(payload["scores"]),
                payload.get("malware_type"),
                payload.get("staged_path"),
                json.dumps(payload.get("static_identity") or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _update_restriction_state(scan_id, blocked, quarantine_path, staged_path):
    conn = _db_connection()
    try:
        conn.execute(
            """
            UPDATE scans
            SET blocked = ?, quarantine_path = ?, staged_path = ?
            WHERE scan_id = ?
            """,
            (int(bool(blocked)), quarantine_path, staged_path, scan_id),
        )
        conn.commit()
    finally:
        conn.close()


def _stage_malware_sample(file_path, sha256_hash):
    if not file_path or not os.path.exists(file_path):
        return None
    staged_name = f"{sha256_hash}.pending.exe"
    staged_path = os.path.join(PENDING_DIR, staged_name)
    shutil.move(file_path, staged_path)
    return staged_path


def _quarantine_and_block(file_path, sha256_hash):
    if not file_path or not os.path.exists(file_path):
        return None, False

    quarantine_name = f"{sha256_hash}.blocked.exe"
    quarantine_path = os.path.join(QUARANTINE_DIR, quarantine_name)
    shutil.move(file_path, quarantine_path)

    blocked = True
    try:
        os.chmod(quarantine_path, 0)
    except OSError:
        pass
    return quarantine_path, blocked


def _infer_malware_type(general_features, scores):
    entropy = float(general_features.get("entropy", 0.0))
    imports = int(general_features.get("import_count", 0))
    sections = int(general_features.get("num_sections", 0))
    malware_score = float(scores.get("ensemble", 0.0))

    if entropy >= 7.2 and imports <= 20:
        return "trojan-dropper / packed malware"
    if entropy >= 7.0 and sections >= 8:
        return "ransomware-like packed binary"
    # Keep worm/botnet tag strict to avoid over-labeling trojans as worms.
    if imports >= 300 and entropy >= 6.2 and sections >= 5 and malware_score >= 70:
        return "worm/botnet-like propagator"
    if imports >= 140 and entropy >= 6.6:
        return "spyware/keylogger-like"
    if sections <= 2 and malware_score >= 70:
        return "trojan-loader"
    if malware_score >= 55 and imports >= 60 and entropy >= 5.2 and sections >= 3:
        return "trojan-stealer/downloader-like"
    if malware_score >= 55 and imports >= 35 and entropy >= 5.0:
        return "backdoor-like trojan"
    if malware_score >= 70:
        return "trojan-like malware"
    if malware_score >= 55:
        return "generic high-risk malware"
    return "generic malware"


def _build_family_attribution(static_identity, source="local_static"):
    family = str((static_identity or {}).get("family") or "").strip()
    if not family or family.lower() in {"unknown", "n/a"}:
        if source == "virustotal":
            return "VirusTotal has no confident family label for this hash yet."
        return "No reliable family attribution from current static signals."

    method = str((static_identity or {}).get("method") or "static_heuristic")
    confidence = float((static_identity or {}).get("confidence") or 0.0)
    method_text = {
        "hash_match": "exact local hash intelligence",
        "yara_rule": "YARA rule correlation",
        "string_signature": "family string signature",
        "import_signature": "import-signature pattern",
        "static_string": "family-like static strings",
        "static_heuristic": "static heuristic pattern",
        "virustotal": "VirusTotal multi-engine threat intelligence",
    }.get(method, method.replace("_", " "))

    return f"{family} ({confidence:.2f}% via {method_text})."


def _family_capability_summary(family_name):
    family = str(family_name or "").strip().lower()
    if not family or family in {"unknown", "n/a"}:
        return "Likely malicious behavior, but family-specific capabilities are not confident from static indicators alone."

    capability_map = {
        "cerber-ransomware": "Encrypts user files, may delete recovery options, and typically demands payment for decryption.",
        "agenttesla": "Steals credentials/keystrokes, exfiltrates data (often via SMTP/HTTP), and can profile infected hosts.",
        "njrat": "Provides remote access, command execution, surveillance, and data theft through attacker-controlled C2.",
        "trojan-injector": "Injects payloads into other processes to hide execution and deliver additional malicious components.",
        "keylogger/spyware": "Monitors keyboard activity and user behavior, then extracts sensitive data from the host.",
        "ransomware-like": "Likely performs file encryption/lockout behavior and pressures victims for ransom payment.",
        "worm/botnet-like propagator": "Attempts lateral spread and remote command execution as part of coordinated botnet behavior.",
        "trojan-loader": "Acts as an initial-stage loader that downloads or executes secondary malware payloads.",
        "trojan-dropper / packed malware": "Likely unpacks and drops hidden payloads while using obfuscation to evade detection.",
        "trojan-like malware": "Likely focuses on stealthy persistence and delivery of additional malicious actions or payloads.",
        "generic high-risk malware": "Shows strong malicious traits and may include payload delivery, persistence, and data abuse.",
        "generic malware": "Detected as malicious, but exact post-execution behavior is not precise without dynamic analysis.",
    }
    if family in capability_map:
        return capability_map[family]

    if "ransom" in family:
        return "Likely focused on file encryption/extortion behavior with potential data impact."
    if "keylog" in family or "spy" in family:
        return "Likely focused on user surveillance and credential/data theft."
    if "worm" in family or "botnet" in family:
        return "Likely supports propagation and remote command-and-control activity."
    if "trojan" in family or "loader" in family or "dropper" in family:
        return "Likely used to deliver or execute secondary malicious payloads with stealth techniques."
    return "Family appears malicious; exact capabilities should be confirmed with dynamic behavior analysis."


def _extract_family_token(detection_name):
    value = str(detection_name or "").strip()
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", " ", value)
    tokens = re.split(r"[./:_\-\s]+", cleaned)
    stop = {
        "trojan", "malware", "gen", "generic", "variant", "win32", "w32", "msil",
        "heur", "suspicious", "unsafe", "riskware", "packed", "packer", "worm",
        "ransom", "ransomware", "backdoor", "dropper", "loader", "agent", "file",
    }
    for tok in tokens:
        t = tok.lower().strip()
        if len(t) < 4 or t in stop:
            continue
        if t.isdigit():
            continue
        return t
    return None


def _lookup_virustotal_family(sha256_hash):
    if not VT_API_KEY:
        return None
    if not sha256_hash:
        return None

    url = f"https://www.virustotal.com/api/v3/files/{sha256_hash}"
    req = urllib.request.Request(url, headers={"x-apikey": VT_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, VT_TIMEOUT)) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    attrs = ((payload or {}).get("data") or {}).get("attributes") or {}
    ptc = attrs.get("popular_threat_classification") or {}
    stats = attrs.get("last_analysis_stats") or {}
    engines = attrs.get("last_analysis_results") or {}
    malicious_votes = int(stats.get("malicious", 0) or 0)
    suspicious_votes = int(stats.get("suspicious", 0) or 0)

    family = None
    confidence = 0.0
    evidence = []

    suggested = str(ptc.get("suggested_threat_label") or "").strip()
    if suggested:
        family = suggested.lower()
        confidence = 95.0 if malicious_votes >= 15 else 88.0
        evidence.append("VT suggested threat label available.")

    if not family:
        names = ptc.get("popular_threat_name") or []
        if names and isinstance(names, list):
            top = names[0] if isinstance(names[0], dict) else {}
            val = str(top.get("value") or "").strip().lower()
            if val:
                family = val
                confidence = 90.0 if malicious_votes >= 15 else 84.0
                evidence.append("VT popular threat name consensus used.")

    if not family:
        token_counts = {}
        for _, engine in engines.items():
            if str(engine.get("category") or "").lower() not in {"malicious", "suspicious"}:
                continue
            token = _extract_family_token(engine.get("result"))
            if token:
                token_counts[token] = token_counts.get(token, 0) + 1
        if token_counts:
            best_token, token_score = sorted(
                token_counts.items(), key=lambda kv: kv[1], reverse=True
            )[0]
            if token_score >= 2:
                family = best_token
                confidence = min(92.0, 65.0 + token_score * 3.5)
                evidence.append(f"VT engine-name consensus ({token_score} engines).")

    if not family:
        return None

    total_signal = malicious_votes + suspicious_votes
    confidence = min(99.0, max(confidence, min(95.0, 55.0 + total_signal * 1.2)))
    return {
        "family": family,
        "confidence": round(float(confidence), 2),
        "method": "virustotal",
        "evidence": evidence[:5],
        "aliases": [],
        "source": "virustotal",
    }


def _build_family_resolution_note(family_name, source):
    fam = str(family_name or "").lower()
    if source == "virustotal":
        return "Family is sourced from VirusTotal hash intelligence."
    if fam and "generic" not in fam and "unknown" not in fam and "n/a" not in fam:
        return "Family is inferred from local static intelligence (hash/YARA/strings/imports)."
    if VT_API_KEY:
        return "No strong exact family yet. Expand local YARA/signature rules or run dynamic sandbox behavior profiling."
    return "For exact family labels, set VIRUSTOTAL_API_KEY and rescan this file hash."


def _build_static_identity(file_path, file_bytes, md5_hash, sha256_hash, general_features, scores):
    sig_db = _load_static_signature_db()
    imports = set(_extract_import_names(file_path))
    strings = _extract_ascii_strings(file_bytes, min_len=5, max_items=5000)
    strings_lower = [s.lower() for s in strings]

    best = {
        "family": "unknown",
        "confidence": 25.0,
        "method": "static_heuristic",
        "evidence": [],
        "aliases": [],
        "static_only": True,
    }

    def choose(family, confidence, method, evidence, aliases=None):
        if confidence > best["confidence"]:
            best["family"] = family
            best["confidence"] = round(float(confidence), 2)
            best["method"] = method
            best["evidence"] = evidence[:8]
            best["aliases"] = aliases or []

    for row in sig_db.get("hashes", []):
        row_sha = str(row.get("sha256", "")).lower()
        row_md5 = str(row.get("md5", "")).lower()
        if row_sha and row_sha == sha256_hash.lower() or row_md5 and row_md5 == md5_hash.lower():
            choose(
                row.get("family", "known-malware"),
                float(row.get("confidence", 99.5)),
                "hash_match",
                [f"Local hash intelligence matched ({'sha256' if row_sha else 'md5'})."],
                row.get("aliases", []),
            )

    yara_rules = _load_yara_rules()
    if yara_rules is not None:
        try:
            matches = yara_rules.match(file_path, timeout=10)
        except Exception:
            matches = []
        for m in matches:
            family = m.meta.get("family") if hasattr(m, "meta") else None
            if not family:
                family = m.tags[0] if getattr(m, "tags", None) else m.rule
            choose(
                family or "yara-detected-malware",
                93.0,
                "yara_rule",
                [f"YARA rule matched: {m.rule}"],
                [str(x) for x in (getattr(m, "tags", []) or [])],
            )

    imports_count = int(general_features.get("import_count", 0))
    entropy = float(general_features.get("entropy", 0.0))
    sections = int(general_features.get("num_sections", 0))
    malware_score = float(scores.get("ensemble", 0.0))

    def has_any(substrs):
        return any(any(sub in s for sub in substrs) for s in strings_lower)

    if {"virtualallocex", "writeprocessmemory", "createremotethread"} <= imports:
        choose(
            "trojan-injector",
            82.0,
            "static_heuristic",
            ["Injection API triad detected: VirtualAllocEx + WriteProcessMemory + CreateRemoteThread."],
            ["injector", "trojan"],
        )
    if {"setwindowshookexa", "getasynckeystate"} & imports:
        choose(
            "keylogger/spyware",
            80.0,
            "static_heuristic",
            ["Keyboard hook API indicators detected."],
            ["spyware", "keylogger"],
        )
    if {"internetopena", "internetconnecta", "httpsendrequesta"} & imports and imports_count >= 220:
        choose(
            "worm/botnet-like propagator",
            76.0,
            "static_heuristic",
            ["High networking API surface with very large import table."],
            ["worm", "botnet"],
        )
    if has_any(["cerber", "wannacry", "locky", "ransom", "decrypt instructions"]) and entropy >= 6.6:
        choose(
            "ransomware-like",
            88.0,
            "static_string",
            ["Ransomware-associated strings plus elevated entropy were found."],
            ["ransomware"],
        )
    if has_any(["zeus", "emotet", "agenttesla", "njrat", "remcos"]):
        choose(
            "known family string-hit",
            84.0,
            "static_string",
            ["Known malware-family token found in binary strings."],
            ["trojan"],
        )

    for row in sig_db.get("strings", []):
        pattern = str(row.get("pattern", "")).lower()
        if not pattern:
            continue
        if any(pattern in s for s in strings_lower):
            choose(
                row.get("family", "string-signature-hit"),
                float(row.get("confidence", 86.0)),
                "string_signature",
                [f"Static string signature matched: {pattern}"],
                row.get("aliases", []),
            )

    for row in sig_db.get("imports", []):
        api = str(row.get("api", "")).lower()
        if api and api in imports:
            choose(
                row.get("family", "import-signature-hit"),
                float(row.get("confidence", 80.0)),
                "import_signature",
                [f"Import signature matched: {api}"],
                row.get("aliases", []),
            )

    if best["family"] == "unknown" and malware_score >= 55:
        best["family"] = _infer_malware_type(general_features, scores)
        best["confidence"] = 65.0
        best["evidence"] = [
            f"Fallback heuristic from PE features: imports={imports_count}, entropy={entropy:.3f}, sections={sections}."
        ]

    return best


def _report_lines(scan_row):
    scores = json.loads(scan_row["scores_json"])
    prevention_steps = json.loads(scan_row["prevention_steps_json"])
    static_identity = _safe_json_loads(scan_row["static_identity_json"], {})
    identity_evidence = static_identity.get("evidence") or []
    return [
        "AI Malware Analysis Report",
        "=" * 60,
        f"Scan ID: {scan_row['scan_id']}",
        f"Timestamp (UTC): {scan_row['created_at']}",
        f"Filename: {scan_row['filename']}",
        f"Verdict: {scan_row['verdict']}",
        f"Malware Type: {scan_row['malware_type'] or 'N/A'}",
        f"Static Family Guess: {static_identity.get('family', 'N/A')}",
        f"Identity Confidence: {static_identity.get('confidence', 'N/A')}%",
        f"Identity Method: {static_identity.get('method', 'N/A')}",
        f"Confidence: {scan_row['confidence']:.2f}%",
        f"Malware Score: {scan_row['malware_score']:.2f}%",
        f"Threshold: {scan_row['threshold']:.2f}%",
        "",
        "File Metadata",
        "-" * 60,
        f"MD5: {scan_row['md5']}",
        f"SHA256: {scan_row['sha256']}",
        f"Entropy: {scan_row['entropy']:.3f}",
        f"Imports: {scan_row['imports_count']}",
        f"Sections: {scan_row['sections']}",
        "",
        "Model Scores (%)",
        "-" * 60,
        f"Ensemble: {scores.get('ensemble')}",
        f"EMBER: {scores.get('ember')}",
        f"Logistic Regression: {scores.get('logistic')}",
        f"Random Forest: {scores.get('random_forest')}",
        f"Classical Average: {scores.get('classical_average')}",
        f"LR/RF Difference: {scores.get('lr_rf_difference')}",
        "",
        "Containment",
        "-" * 60,
        f"Threat Blocked: {'Yes' if scan_row['blocked'] else 'No'}",
        f"Pending Restriction Path: {scan_row['staged_path'] or 'N/A'}",
        f"Quarantine Path: {scan_row['quarantine_path'] or 'N/A'}",
        "",
        "Identity Evidence",
        "-" * 60,
        "; ".join(identity_evidence) if identity_evidence else "No strong family-specific static indicators.",
        "",
        "Analysis Summary",
        "-" * 60,
        scan_row["analysis_summary"],
        "",
        "Prevention Steps",
        "-" * 60,
    ]


def _format_report_text(scan_row):
    lines = _report_lines(scan_row)
    prevention_steps = json.loads(scan_row["prevention_steps_json"])
    if prevention_steps:
        for idx, step in enumerate(prevention_steps, start=1):
            lines.append(f"{idx}. {step}")
    else:
        lines.append("No additional prevention steps required.")
    return "\n".join(lines) + "\n"


def _build_report_pdf(scan_row):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    x_margin = 0.7 * inch
    y = height - 0.8 * inch
    line_h = 14

    for line in _format_report_text(scan_row).splitlines():
        if y <= 0.8 * inch:
            pdf.showPage()
            y = height - 0.8 * inch
        pdf.setFont("Helvetica", 10)
        pdf.drawString(x_margin, y, line[:140])
        y -= line_h

    pdf.save()
    return buffer.getvalue()


def _bounded_probability(prob):
    floor = float(os.getenv("PROBABILITY_FLOOR", "0.01"))
    floor = max(0.0, min(0.49, floor))
    return float(max(floor, min(1.0 - floor, float(prob))))


def _classical_malware_scores(general_features):
    if lr_model is None and rf_model is None:
        return None, None

    X = pd.DataFrame(
        [
            {
                "file_size": general_features["file_size"],
                "entropy": general_features["entropy"],
                "num_sections": general_features["num_sections"],
                "entry_point": general_features["entry_point"],
                "image_base": general_features["image_base"],
                "characteristics": general_features["characteristics"],
            }
        ]
    )

    lr_prob = (
        _bounded_probability(lr_model.predict_proba(X)[0][1])
        if lr_model is not None
        else None
    )
    rf_prob = float(rf_model.predict_proba(X)[0][1]) if rf_model is not None else None
    return lr_prob, rf_prob

def _ensemble_malware_score(ember_prob, lr_prob, rf_prob):
    classical_scores = [s for s in (lr_prob, rf_prob) if s is not None]
    classical_avg = (
        float(sum(classical_scores) / len(classical_scores)) if classical_scores else None
    )

    ember_weight = float(os.getenv("EMBER_WEIGHT", "0.60"))
    classical_weight = float(os.getenv("CLASSICAL_WEIGHT", "0.40"))

    if classical_avg is None:
        score = float(ember_prob)
    else:
        total_weight = ember_weight + classical_weight
        if total_weight <= 0:
            total_weight = 1.0
        score = (
            (ember_weight * float(ember_prob)) + (classical_weight * classical_avg)
        ) / total_weight

    lr_rf_gap = (
        abs(float(lr_prob) - float(rf_prob))
        if lr_prob is not None and rf_prob is not None
        else None
    )
    return float(max(0.0, min(1.0, score))), classical_avg, lr_rf_gap

def _build_analysis_text(verdict, confidence, general_features, scores):
    entropy = float(general_features.get("entropy", 0.0))
    imports = int(general_features.get("import_count", 0))
    sections = int(general_features.get("num_sections", 0))

    entropy_label = (
        "high" if entropy >= 7.0 else "moderate" if entropy >= 6.0 else "low"
    )
    import_label = "sparse" if imports <= 20 else "normal"
    section_label = "unusual" if sections >= 8 or sections <= 2 else "typical"

    lr_rf_gap = scores.get("lr_rf_difference")
    agreement_text = (
        f"Model disagreement is {lr_rf_gap:.2f}% between LR and RF."
        if lr_rf_gap is not None
        else "Only one classical model is available, so LR/RF agreement is not computed."
    )

    if verdict == "malware":
        return (
            f"File is classified as malware with {confidence:.2f}% confidence. "
            f"Entropy is {entropy:.3f} ({entropy_label}), imports are {imports} ({import_label}), "
            f"and section count is {sections} ({section_label}). {agreement_text} "
            "Signals suggest possible packing/obfuscation or suspicious PE behavior."
        )

    return (
        f"File is classified as benign with {confidence:.2f}% confidence. "
        f"Entropy is {entropy:.3f} ({entropy_label}), imports are {imports} ({import_label}), "
        f"and section count is {sections} ({section_label}). {agreement_text} "
        "No strong malicious indicators were detected from the current static feature set."
    )

def _build_prevention_steps(verdict, confidence, general_features):
    if verdict == "benign":
        return []

    entropy = float(general_features.get("entropy", 0.0))
    steps = [
        "Keep real-time antivirus/EDR enabled and fully updated.",
        "Allow execution only from trusted/signed sources (application allow-listing).",
        "Run unknown binaries inside an isolated VM/sandbox before production use.",
        "Block suspicious outbound traffic and monitor process/network behavior.",
    ]

    if verdict == "malware" or confidence >= 70:
        steps.insert(0, "Quarantine this file immediately and avoid executing it on host systems.")
        steps.insert(1, "Collect hash/IOC and scan other endpoints for the same indicators.")

    if entropy >= 7.0:
        steps.append("Because entropy is high, perform deeper dynamic analysis for packed payload behavior.")

    return steps


@app.on_event("startup")
async def _startup():
    global MODEL_METRICS_CACHE
    _init_db()
    os.makedirs(VOICE_TMP_DIR, exist_ok=True)
    _load_static_signature_db()
    _load_yara_rules()
    try:
        MODEL_METRICS_CACHE = _compute_model_metrics()
    except Exception:
        MODEL_METRICS_CACHE = {
            "dataset_size": int(max(0, EMBER_DATASET_SIZE)),
            "dataset_name": "EMBER 2018",
            "accuracy": {"logistic": None, "random_forest": None},
            "computed_on": datetime.now(timezone.utc).isoformat(),
        }

    # Warm up optional voice engine once on startup when available.
    if pyttsx3 is not None:
        try:
            _get_voice_engine()
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("mal.html", {"request": request})


@app.get("/dashboard")
async def dashboard():
    conn = _db_connection()
    try:
        total_scans = int(conn.execute("SELECT COUNT(*) AS c FROM scans").fetchone()["c"])
        threats_blocked = int(
            conn.execute("SELECT COUNT(*) AS c FROM scans WHERE blocked = 1").fetchone()["c"]
        )
        recent_rows = conn.execute(
            """
            SELECT scan_id, created_at, filename, verdict, confidence, blocked
            FROM scans
            ORDER BY created_at DESC
            LIMIT 10
            """
        ).fetchall()
    finally:
        conn.close()

    recent = []
    for row in recent_rows:
        recent.append(
            {
                "scan_id": row["scan_id"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "verdict": row["verdict"],
                "confidence": round(float(row["confidence"]), 2),
                "blocked": bool(row["blocked"]),
            }
        )

    return {
        "dataset_size": int((MODEL_METRICS_CACHE or {}).get("dataset_size", 0)),
        "dataset_name": (MODEL_METRICS_CACHE or {}).get("dataset_name", "EMBER 2018"),
        "model_accuracy": (MODEL_METRICS_CACHE or {}).get(
            "accuracy", {"logistic": None, "random_forest": None}
        ),
        "metrics_computed_on": (MODEL_METRICS_CACHE or {}).get("computed_on"),
        "total_scans": total_scans,
        "threats_blocked": threats_blocked,
        "recent_activity": recent,
    }


@app.get("/voice/capabilities")
async def voice_capabilities():
    return _voice_capabilities()


@app.post("/voice/narrate")
async def voice_narrate(payload: NarrationRequest, background_tasks: BackgroundTasks):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Narration text is required")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Narration text too long (max 500 chars)")
    if pyttsx3 is None:
        raise HTTPException(status_code=503, detail="Server voice model not available")

    try:
        wav_path = _render_tts_wav(text=text, rate=payload.rate, volume=payload.volume)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice rendering failed: {exc}") from exc

    background_tasks.add_task(lambda p=wav_path: os.path.exists(p) and os.remove(p))
    return FileResponse(path=wav_path, media_type="audio/wav", filename="narration.wav")


@app.get("/history")
async def history(limit: int = 100):
    safe_limit = max(1, min(limit, 500))
    conn = _db_connection()
    try:
        rows = conn.execute(
            """
            SELECT scan_id, created_at, filename, verdict, confidence, malware_score, blocked, md5, sha256, malware_type, static_identity_json, staged_path
            FROM scans
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "items": [
            {
                "scan_id": row["scan_id"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "verdict": row["verdict"],
                "confidence": round(float(row["confidence"]), 2),
                "malware_score": round(float(row["malware_score"]), 2),
                "blocked": bool(row["blocked"]),
                "md5": row["md5"],
                "sha256": row["sha256"],
                "malware_type": row["malware_type"],
                "can_restrict": bool(row["staged_path"]),
                "static_identity": _safe_json_loads(row["static_identity_json"], {}),
            }
            for row in rows
        ]
    }


@app.get("/report/{scan_id}")
async def report(scan_id: str):
    conn = _db_connection()
    try:
        row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Scan report not found")

    report_pdf = _build_report_pdf(row)
    filename = f"scan_report_{scan_id}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=report_pdf, media_type="application/pdf", headers=headers)


@app.post("/restrict/{scan_id}")
async def restrict_malware(scan_id: str):
    conn = _db_connection()
    try:
        row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    if row["verdict"] != "malware":
        raise HTTPException(status_code=400, detail="Only malware scans can be restricted")

    if bool(row["blocked"]) and row["quarantine_path"] and os.path.exists(row["quarantine_path"]):
        return {
            "scan_id": scan_id,
            "blocked": True,
            "quarantine_path": row["quarantine_path"],
            "can_restrict": False,
            "detail": "Already restricted",
        }

    staged_path = row["staged_path"]
    quarantine_path, blocked = None, False
    if staged_path and os.path.exists(staged_path):
        quarantine_path, blocked = _quarantine_and_block(staged_path, row["sha256"])
        staged_path = None
    elif row["quarantine_path"] and os.path.exists(row["quarantine_path"]):
        quarantine_path, blocked = row["quarantine_path"], True
        staged_path = None
    else:
        raise HTTPException(status_code=404, detail="No pending malware file to restrict")

    _update_restriction_state(scan_id, blocked, quarantine_path, staged_path)
    return {
        "scan_id": scan_id,
        "blocked": bool(blocked),
        "quarantine_path": quarantine_path,
        "can_restrict": False,
        "detail": "Restriction applied" if blocked else "Restriction attempted",
    }

@app.post("/scan")
async def scan_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.lower().endswith(".exe"):
        raise HTTPException(status_code=400, detail="Only .exe files supported")

    # Keep only the display name from client input; use a generated temp path on disk.
    safe_name = Path(file.filename).name
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".exe", dir=UPLOAD_DIR, delete=False
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            file_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    blocked = False
    quarantine_path = None
    staged_path = None
    malware_type = None
    malware_family = None
    family_attribution = "N/A"
    capability_summary = "N/A"
    family_source = "none"
    family_resolution_note = "N/A"
    static_identity = {
        "family": "N/A",
        "confidence": 0.0,
        "method": "not_applicable",
        "evidence": [],
        "aliases": [],
        "static_only": True,
    }
    scan_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        md5_hash = hashlib.md5(file_bytes).hexdigest()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        vector = pe_to_ember_vector(file_path)
        benign_prob, ember_malware_prob = get_ember_probabilities(vector)

        general_features = extract_general_features(file_path)
        lr_prob, rf_prob = _classical_malware_scores(general_features)
        malware_score, classical_avg, lr_rf_gap = _ensemble_malware_score(
            ember_malware_prob, lr_prob, rf_prob
        )
        verdict = "malware" if malware_score >= MALWARE_THRESHOLD else "benign"
        confidence = malware_score if verdict == "malware" else (1.0 - malware_score)
        score_percent = {
            "ensemble": round(malware_score * 100, 2),
            "ember": round(ember_malware_prob * 100, 2),
            "logistic": round(lr_prob * 100, 2) if lr_prob is not None else None,
            "random_forest": round(rf_prob * 100, 2) if rf_prob is not None else None,
            "classical_average": round(classical_avg * 100, 2) if classical_avg is not None else None,
            "lr_rf_difference": round(lr_rf_gap * 100, 2) if lr_rf_gap is not None else None,
            "benign_ember": round(benign_prob * 100, 2),
        }
        confidence_percent = round(confidence * 100, 2)
        analysis_summary = _build_analysis_text(
            verdict=verdict,
            confidence=confidence_percent,
            general_features=general_features,
            scores=score_percent,
        )
        prevention_steps = _build_prevention_steps(
            verdict=verdict,
            confidence=confidence_percent,
            general_features=general_features,
        )

        if verdict == "malware":
            static_identity = _build_static_identity(
                file_path=file_path,
                file_bytes=file_bytes,
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                general_features=general_features,
                scores=score_percent,
            )
            vt_identity = _lookup_virustotal_family(sha256_hash)
            if vt_identity:
                malware_family = vt_identity.get("family")
                family_source = "virustotal"
                family_attribution = _build_family_attribution(vt_identity, source=family_source)
                capability_summary = _family_capability_summary(malware_family)
                # Keep local evidence visible too, but prioritize VT family label.
                local_ev = static_identity.get("evidence") or []
                vt_ev = vt_identity.get("evidence") or []
                static_identity["evidence"] = (vt_ev + local_ev)[:8]
            else:
                malware_family = static_identity.get("family") or _infer_malware_type(general_features, score_percent)
                family_source = "local_static"
                family_attribution = _build_family_attribution(static_identity, source=family_source)
                capability_summary = _family_capability_summary(malware_family)
            malware_type = malware_family
            family_resolution_note = _build_family_resolution_note(malware_family, family_source)
            analysis_summary = (
                f"{analysis_summary} Family attribution: {family_attribution} "
                f"Capability summary: {capability_summary} "
                f"Attribution note: {family_resolution_note}"
            )
            try:
                staged_path = _stage_malware_sample(file_path, sha256_hash)
                file_path = None
            except Exception:
                staged_path = None
                blocked = False
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}") from exc
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    response_payload = {
        "scan_id": scan_id,
        "created_at": created_at,
        "timestamp": created_at,
        "filename": safe_name,
        "verdict": verdict,
        "confidence": confidence_percent,
        "malware_score": round(malware_score * 100, 2),
        "threshold": round(MALWARE_THRESHOLD * 100, 2),
        "entropy": round(float(general_features["entropy"]), 3),
        "imports": int(general_features.get("import_count", 0)),
        "sections": int(general_features["num_sections"]),
        "md5": md5_hash,
        "sha256": sha256_hash,
        "analysis_summary": analysis_summary,
        "prevention_steps": prevention_steps,
        "blocked": blocked,
        "quarantine_path": quarantine_path,
        "staged_path": staged_path,
        "malware_type": malware_type,
        "malware_family": malware_family,
        "family_attribution": family_attribution,
        "capability_summary": capability_summary,
        "family_source": family_source,
        "family_resolution_note": family_resolution_note,
        "can_restrict": bool(staged_path),
        "static_identity": static_identity,
        "scores": score_percent,
    }
    try:
        _insert_scan_history(response_payload)
    except Exception:
        # Do not fail scanning response if history persistence fails.
        pass
    return response_payload
