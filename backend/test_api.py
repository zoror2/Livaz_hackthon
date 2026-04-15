import httpx, asyncio

async def test():
    r = await httpx.AsyncClient().post(
        "http://127.0.0.1:8000/api/predict_live",
        json={"lat": 13.08, "lon": 80.27},
        timeout=60,
    )
    d = r.json()
    print("Score:", d.get("composite_score"))
    print("Call:", d.get("call_alert"))
    print("SMS:", d.get("sms_alert"))
    print("Shelters:", len(d.get("shelters", [])))
    print("Breakdown:", d.get("breakdown"))

asyncio.run(test())
