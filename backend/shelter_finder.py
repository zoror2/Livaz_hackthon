"""
shelter_finder.py
Finds nearest emergency shelters using OpenStreetMap Overpass API.
Returns schools, hospitals, government buildings, places of worship,
and police stations near a given coordinate.
"""

import math
import httpx

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Shelter types to search for (OSM tags)
SHELTER_QUERIES = {
    "school":      'node["amenity"="school"]',
    "hospital":    'node["amenity"="hospital"]',
    "worship":     'node["amenity"="place_of_worship"]',
    "police":      'node["amenity"="police"]',
    "community":   'node["amenity"="community_centre"]',
    "govt":        'node["office"="government"]',
    "fire_station":'node["amenity"="fire_station"]',
}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two GPS points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _google_maps_link(lat, lon):
    return f"https://maps.google.com/?q={lat},{lon}"


async def find_nearby_shelters(lat: float, lon: float, radius_m: int = 5000, limit: int = 3):
    """
    Queries OpenStreetMap for nearby shelters within radius_m meters.
    Returns up to `limit` nearest shelters sorted by distance.
    """
    # Build Overpass query for all shelter types within radius
    amenity_parts = []
    for tag_query in SHELTER_QUERIES.values():
        amenity_parts.append(f'{tag_query}(around:{radius_m},{lat},{lon});')

    query = f"""
    [out:json][timeout:10];
    (
      {''.join(amenity_parts)}
    );
    out body;
    """

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.post(OVERPASS_URL, data={"data": query})
            if r.status_code != 200:
                print(f"[SHELTER] Overpass API returned {r.status_code}")
                return []

            elements = r.json().get("elements", [])
    except Exception as e:
        print(f"[SHELTER] Overpass API error: {e}")
        return []

    # Parse results and calculate distances
    shelters = []
    for el in elements:
        if "lat" not in el or "lon" not in el:
            continue

        name = el.get("tags", {}).get("name", "Unnamed Shelter")
        amenity = el.get("tags", {}).get("amenity", el.get("tags", {}).get("office", "shelter"))
        dist_km = _haversine_km(lat, lon, el["lat"], el["lon"])

        # Friendly type label
        type_labels = {
            "school": "🏫 School",
            "hospital": "🏥 Hospital",
            "place_of_worship": "⛪ Place of Worship",
            "police": "👮 Police Station",
            "community_centre": "🏛️ Community Centre",
            "government": "🏛️ Govt Office",
            "fire_station": "🚒 Fire Station",
        }
        type_label = type_labels.get(amenity, "📍 Shelter")

        shelters.append({
            "name": name,
            "type": type_label,
            "amenity": amenity,
            "lat": el["lat"],
            "lon": el["lon"],
            "distance_km": round(dist_km, 2),
            "maps_link": _google_maps_link(el["lat"], el["lon"]),
        })

    # Sort by distance and return closest
    shelters.sort(key=lambda x: x["distance_km"])
    return shelters[:limit]


def format_shelter_sms(score: int, shelters: list, forecast_mm: float = 0) -> str:
    """Format shelter info into an SMS message."""
    lines = [
        f"⚠️ FLOOD ALERT - Advaya Risk Engine",
        f"Risk Score: {score}/100 CRITICAL",
        f"Predicted rainfall: {forecast_mm:.0f}mm in next 12h",
        "",
        "Nearest safe shelters:",
    ]

    for i, s in enumerate(shelters, 1):
        lines.append(f"{i}. {s['type']} - {s['name']} ({s['distance_km']}km)")
        lines.append(f"   {s['maps_link']}")

    lines.append("")
    lines.append("Evacuate NOW. Do not wait for the flood.")
    lines.append("- Advaya Climate AI (NASA Prithvi EO-2.0)")

    return "\n".join(lines)
