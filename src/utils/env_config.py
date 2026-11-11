"""
Environment & API Key Configuration Manager (src/utils/env_config.py)
Loads `.env` credentials safely, masks sensitive values in logs/API responses,
and supports updating `.env` keys via the MLOps Control Plane API or CLI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.utils.storage import ROOT_DIR as PROJECT_ROOT


ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

ALLOWED_ENV_KEYS: dict[str, dict[str, str]] = {
    "ITIC_API_KEY": {
        "provider": "iTIC / Longdo Traffic Probe Feed",
        "category": "traffic_probe",
        "base_url_env": "ITIC_API_BASE_URL",
        "default_url": "https://traffic.longdo.com/api/v1",
        "register_url": "https://map.longdo.com/console/",
        "fallback_mode": "Deterministic 32-Segment BKK Bronze Parquet / Historical Feed",
    },
    "LONGDO_MAP_API_KEY": {
        "provider": "Longdo Map Tiles & Routing GIS",
        "category": "gis_tiles",
        "base_url_env": "ITIC_API_BASE_URL",
        "default_url": "https://ms.longdo.com/mmmap/",
        "register_url": "https://map.longdo.com/console/",
        "fallback_mode": "CartoDB DarkMatter / OpenStreetMap Tiles",
    },
    "SERPAPI_API_KEY": {
        "provider": "Google Maps Search & Places API (via SerpApi)",
        "category": "gis_google_maps",
        "base_url_env": "GOOGLE_MAPS_API_BASE_URL",
        "default_url": "https://serpapi.com/search?engine=google_maps",
        "register_url": "https://serpapi.com/manage-api-key",
        "fallback_mode": "Local 32-Corridor BKK Spatial Index + OpenStreetMap",
    },
    "TMD_API_KEY": {
        "provider": "Thai Meteorological Department (TMD NWP API)",
        "category": "weather",
        "base_url_env": "TMD_API_BASE_URL",
        "default_url": "https://data.tmd.go.th/nwpapi/v1/forecast/location/hourly/at",
        "register_url": "https://data.tmd.go.th/nwpapi/doc/",
        "fallback_mode": "Open-Meteo Public Archive + Observed Rain Lag-Only (Zero Leakage)",
    },
    "TMD_API_UID": {
        "provider": "Thai Meteorological Department User ID (UID)",
        "category": "weather",
        "base_url_env": "TMD_API_BASE_URL",
        "default_url": "https://data.tmd.go.th/nwpapi/v1/forecast/location/hourly/at",
        "register_url": "https://data.tmd.go.th/nwpapi/doc/",
        "fallback_mode": "Open-Meteo Public Archive",
    },
    "OPENMETEO_API_KEY": {
        "provider": "Open-Meteo Weather API (Commercial / High-Rate Tier)",
        "category": "weather",
        "base_url_env": "WEATHER_API_BASE_URL",
        "default_url": "https://api.open-meteo.com/v1/forecast",
        "register_url": "https://open-meteo.com/en/pricing",
        "fallback_mode": "Open-Meteo Free Non-Commercial Endpoint",
    },
    "TRAFFY_API_TOKEN": {
        "provider": "BMA Traffy Fondue Civic Hazard & Flood API",
        "category": "civic_incidents",
        "base_url_env": "TRAFFY_API_BASE_URL",
        "default_url": "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1",
        "register_url": "https://publicapi.traffy.in.th/",
        "fallback_mode": "Public Traffy GeoJSON Endpoint + Local Silver Cache",
    },
    "BKK_TRAFFIC_API_KEY": {
        "provider": "FastAPI Inference & MLOps Control Plane Auth (X-API-Key)",
        "category": "mlops_security",
        "base_url_env": "API_HOST",
        "default_url": "http://127.0.0.1:8050/api/v1",
        "register_url": "Local .env (BKK_TRAFFIC_API_KEY)",
        "fallback_mode": "Default Dev Key (bkk-tactical-dev-key-2026)",
    },
    "MLOPS_WEBHOOK_URL": {
        "provider": "5-Layer Drift & Circuit-Breaker Alert Webhook (Slack/Discord)",
        "category": "mlops_alerting",
        "base_url_env": "MLOPS_WEBHOOK_URL",
        "default_url": "https://hooks.slack.com/services/...",
        "register_url": "Slack / Discord Incoming Webhooks",
        "fallback_mode": "Local Audit Log (artifacts/monitoring/drift_report.json)",
    },
    "LINE_NOTIFY_TOKEN": {
        "provider": "LINE Notify Token for BMA Traffic Ops Alerts",
        "category": "mlops_alerting",
        "base_url_env": "LINE_NOTIFY_TOKEN",
        "default_url": "https://notify-api.line.me/api/notify",
        "register_url": "https://notify-bot.line.me/en/",
        "fallback_mode": "Local Audit Log",
    },
}


def load_dotenv_file(dotenv_path: Path | None = None, override: bool = False) -> dict[str, str]:
    """
    Zero-dependency `.env` loader (also compatible with python-dotenv).
    Loads variables into `os.environ` without printing any secret values.
    """
    path = dotenv_path or ENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded


def mask_secret(value: str | None) -> str:
    """Mask an API key or secret so it never leaks in plaintext."""
    if not value or not value.strip():
        return "NOT_SET"
    clean = value.strip()
    if len(clean) <= 6:
        return "*" * len(clean)
    return f"{clean[:4]}{'*' * (len(clean) - 6)}{clean[-2:]}"


def get_env(key: str, default: str = "") -> str:
    """Retrieve an environment variable after ensuring `.env` is loaded."""
    load_dotenv_file(override=False)
    return os.environ.get(key, default).strip()


def get_api_keys_status() -> dict[str, Any]:
    """
    Return safe, masked diagnostic status of all external & internal API keys.
    Safe to expose in Dashboard UI and `/api/v1/config/keys-status`.
    """
    load_dotenv_file(override=True)
    items: list[dict[str, Any]] = []
    configured_count = 0

    for key_name, meta in ALLOWED_ENV_KEYS.items():
        raw_val = os.environ.get(key_name, "").strip()
        is_set = bool(raw_val)
        if is_set:
            configured_count += 1
        base_url_key = meta.get("base_url_env", "")
        endpoint_url = os.environ.get(base_url_key, "").strip() or meta["default_url"]
        if key_name == "BKK_TRAFFIC_API_KEY":
            host = os.environ.get("API_HOST", "127.0.0.1")
            port = os.environ.get("API_PORT", "8050")
            endpoint_url = f"http://{host}:{port}/api/v1"

        items.append(
            {
                "key_name": key_name,
                "provider": meta["provider"],
                "category": meta["category"],
                "is_configured": is_set,
                "masked_preview": mask_secret(raw_val),
                "mode": "LIVE_API_KEY_ACTIVE" if is_set else "FALLBACK_READY",
                "endpoint_url": endpoint_url,
                "fallback_mode": meta["fallback_mode"],
                "register_url": meta["register_url"],
            }
        )

    return {
        "env_file_path": str(ENV_PATH.relative_to(PROJECT_ROOT)),
        "env_file_exists": ENV_PATH.exists(),
        "total_keys_tracked": len(ALLOWED_ENV_KEYS),
        "configured_keys_count": configured_count,
        "api_host": get_env("API_HOST", "127.0.0.1"),
        "api_port": int(get_env("API_PORT", "8050") or "8050"),
        "keys": items,
    }


def update_env_keys(updates: dict[str, str], dotenv_path: Path | None = None) -> dict[str, Any]:
    """
    Safely update one or more allowed API keys in `.env` and `os.environ`.
    Never logs raw values.
    """
    target_path = dotenv_path or ENV_PATH
    if not target_path.exists():
        if ENV_EXAMPLE_PATH.exists():
            target_path.write_text(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            target_path.write_text("# Local .env file\n", encoding="utf-8")

    content = target_path.read_text(encoding="utf-8")
    updated_keys: list[str] = []
    rejected_keys: list[str] = []

    allowed_all = set(ALLOWED_ENV_KEYS.keys()) | {
        "ITIC_API_BASE_URL",
        "GOOGLE_MAPS_API_BASE_URL",
        "TMD_API_BASE_URL",
        "WEATHER_API_BASE_URL",
        "TRAFFY_API_BASE_URL",
        "API_HOST",
        "API_PORT",
    }

    for k, v in updates.items():
        k_clean = k.strip()
        if k_clean not in allowed_all:
            rejected_keys.append(k_clean)
            continue

        v_sanitized = str(v).replace("\n", "").replace("\r", "").strip()
        pattern = re.compile(rf"^{re.escape(k_clean)}=.*$", re.MULTILINE)
        replacement_line = f'{k_clean}="{v_sanitized}"'

        if pattern.search(content):
            content = pattern.sub(replacement_line, content)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += f"{replacement_line}\n"

        os.environ[k_clean] = v_sanitized
        updated_keys.append(k_clean)

    target_path.write_text(content, encoding="utf-8")
    return {
        "updated_keys": updated_keys,
        "rejected_keys": rejected_keys,
        "status": get_api_keys_status(),
    }
