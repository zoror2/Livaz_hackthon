import asyncio, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from backend.shelter_finder import find_nearby_shelters, format_shelter_sms

async def test():
    shelters = await find_nearby_shelters(13.08, 80.27, radius_m=5000, limit=3)
    print(f"Found {len(shelters)} shelters near Chennai:")
    for s in shelters:
        print(f"  {s['amenity']} - {s['name']} ({s['distance_km']}km)")
        print(f"    {s['maps_link']}")
    print()
    sms = format_shelter_sms(85, shelters, 45.0)
    print("--- SMS MESSAGE ---")
    print(sms)

asyncio.run(test())
