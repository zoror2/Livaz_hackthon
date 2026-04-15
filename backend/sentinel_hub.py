"""
sentinel_hub.py
---------------
Fetches real Sentinel-2 NDWI (water index) imagery using
Sentinel Hub Process API with OAuth client credentials.

NDWI = (Green - NIR) / (Green + NIR)
  > 0.1 → water / potential flood
Pixels above threshold are rendered as blue transparent PNG for Leaflet overlay.
"""

import json
import httpx
from datetime import datetime, timedelta

# ── OAuth Credentials ──────────────────────────────────────────────────────────
SH_CLIENT_ID     = "94881941-e74f-4ffe-a252-1db3785bc637"
SH_CLIENT_SECRET = "zvSbSkraAn3Qv6zakmMaZR35z67Colpr"
SH_TOKEN_URL     = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
SH_PROCESS_URL   = "https://services.sentinel-hub.com/api/v1/process"

# ── Evalscript: NDWI → transparent blue PNG ────────────────────────────────────
NDWI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08"], units: "DN" }],
    output: { bands: 4, sampleType: "UINT8" }
  };
}
function evaluatePixel(s) {
  var green = s.B03 / 10000.0;
  var nir   = s.B08 / 10000.0;
  var denom = green + nir;
  if (denom === 0) return [0, 0, 0, 0];
  var ndwi = (green - nir) / denom;

  if (ndwi > 0.2) {
    // Strong water — deep blue, high opacity
    var a = Math.min(220, Math.round((ndwi - 0.2) * 500 + 140));
    return [20, 100, 220, a];
  }
  if (ndwi > 0.05) {
    // Marginal water — lighter blue
    var a2 = Math.round((ndwi - 0.05) * 400 + 60);
    return [80, 160, 240, a2];
  }
  return [0, 0, 0, 0]; // transparent — dry land
}
"""

# ── Named bounding boxes for Tamil Nadu & AP coast ─────────────────────────────
BBOXES = {
    # [west, south, east, north]
    "chennai":      [79.90, 12.90, 80.40, 13.30],
    "tn_coast":     [79.50,  8.00, 80.70, 13.50],
    "andhra_coast": [79.80, 13.50, 81.00, 16.50],
}


async def _get_token() -> str | None:
    """Get OAuth2 bearer token using client credentials flow."""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                SH_TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     SH_CLIENT_ID,
                    "client_secret": SH_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                print(f"[SentinelHub] Token acquired successfully")
                return token
            print(f"[SentinelHub] Token error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[SentinelHub] Token request failed: {e}")
    return None


async def fetch_ndwi_overlay(
    bbox: list,
    width: int = 512,
    height: int = 512,
    days_back: int = 60,
) -> bytes | None:
    """
    Returns PNG bytes of NDWI water overlay for the given bbox.
    Uses leastCC mosaicking to pick the least cloudy Sentinel-2 scene.
    """
    token = await _get_token()
    if not token:
        print("[SentinelHub] No token — cannot fetch imagery")
        return None

    now   = datetime.utcnow()
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    end   = now.strftime("%Y-%m-%dT23:59:59Z")

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": start, "to": end},
                    "maxCloudCoverage": 40,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width":  width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": NDWI_EVALSCRIPT,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                SH_PROCESS_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type":  "application/json",
                    "Accept":        "image/png",
                },
                content=json.dumps(payload),
            )
            if resp.status_code == 200:
                print(f"[SentinelHub] NDWI image fetched: {len(resp.content)} bytes")
                return resp.content
            print(f"[SentinelHub] Process API {resp.status_code}: {resp.text[:400]}")
        except Exception as e:
            print(f"[SentinelHub] Process request failed: {e}")
    return None
