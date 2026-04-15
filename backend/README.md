# Backend API

## Install

```powershell
cd D:\AdvayaHakcathon
venv\Scripts\activate
pip install -r backend\requirements.txt
```

## Run

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `POST /predict`
- `GET /results/{result_id}`

## Sample request

```json
{
  "district_name": "Chennai",
  "rainfall_30d": [12.0, 13.5, 9.0, 3.2, 0.0, 4.2, 8.8, 1.0, 0.0, 0.0, 7.0, 6.1, 2.2, 0.0, 3.4, 5.6, 12.3, 20.1, 4.9, 0.5, 0.0, 1.2, 3.8, 2.7, 10.4, 11.2, 0.0, 0.0, 5.5, 8.9],
  "windspeed_30d": [14.0, 15.1, 12.2, 9.8, 8.4, 10.5, 11.1, 13.4, 12.7, 10.0, 9.9, 14.2, 16.0, 15.5, 14.8, 12.2, 11.3, 10.4, 9.6, 8.9, 7.8, 8.2, 9.1, 10.7, 11.9, 12.6, 13.0, 13.7, 14.4, 15.0]
}
```

## Notes

- If `satellite_path` is omitted, bundled sample TIFF is used.
- If Prithvi inference fails, API automatically falls back to weather-only score.
