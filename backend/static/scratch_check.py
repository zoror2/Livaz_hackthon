import json, math
from pathlib import Path

manifest = json.loads(Path('backend/static/overlays/manifest.json').read_text())

def nearest(lat, lon):
    best, bd = None, float('inf')
    for o in manifest:
        clat = (o['bounds'][0][0]+o['bounds'][1][0])/2
        clon = (o['bounds'][0][1]+o['bounds'][1][1])/2
        d = math.sqrt((clat-lat)**2+(clon-lon)**2)
        if d < bd:
            bd, best = d, o
    return best, bd

for name, lat, lon in [('Chennai',13.08,80.27),('Hyderabad',17.38,78.47),('Mumbai',19.07,72.87),('Kolkata',22.57,88.36)]:
    tile, dist = nearest(lat, lon)
    score = min(60, int(tile['flood_pct']*6))
    print(f"{name}: tile={tile['image_id']} flood={tile['flood_pct']}% dist={dist:.1f} score={score}")
