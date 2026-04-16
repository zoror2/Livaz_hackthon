import React, { useState } from 'react';
import { Activity, ShieldAlert, CloudRain, Wind, Droplets, Loader2, MapPin } from 'lucide-react';
import MapComponent from './components/MapComponent';
import './index.css';

export default function App() {
  const [lang, setLang] = useState('en');
  const [monsoon, setMonsoon] = useState(false);

  const [weatherData, setWeatherData] = useState({
    rainfall: '-- mm',
    windSpeed: '-- km/h',
    humidity: '--%',
    forecast12h: '--',
  });

  const [riskData, setRiskData] = useState({
    score: '--',
    level: 'NO DATA',
    alerts: {
      en: 'Click anywhere on the map to run real-time risk analysis.',
      ta: 'நேரடி ஆபத்து பகுப்பாய்வை இயக்க வரைபடத்தில் எங்கும் கிளிக் செய்யவும்.',
      te: 'రియల్-టైమ్ రిస్క్ విశ్లేషణ నడపడానికి మ్యాప్‌పై ఎక్కడైనా క్లిక్ చేయండి.',
    },
  });

  const [modelMetrics, setModelMetrics] = useState(null);

  const getRiskColor = (level) => {
    const map = {
      CRITICAL: '#ef4444',
      HIGH: '#f97316',
      MODERATE: '#eab308',
      LOW: '#22c55e',
      'NO DATA': '#94a3b8',
    };
    return map[level] || '#94a3b8';
  };

  const [tileData, setTileData] = useState(null);

  const handlePrediction = (data) => {
    if (data.status !== 'success') return;
    setWeatherData({
      rainfall:    data.weather.rainfall,
      windSpeed:   data.weather.windSpeed,
      humidity:    data.weather.humidity,
      forecast12h: data.weather.forecast_12h ?? 0,
    });
    setRiskData({
      score:  data.composite_score,
      level:  data.risk_level,
      alerts: data.alerts,
    });
    if (data.nearest_overlay) setTileData(data.nearest_overlay);
    if (data.prediction)      setTileData(data.prediction);
  };

  const riskColor = getRiskColor(riskData.level);
  const isReady   = riskData.level !== 'NO DATA';

  return (
    <div className="dashboard-container">
      {/* Alert Banner */}
      <header
        className="glass-panel header-span"
        style={{
          borderColor: riskColor,
          background: isReady ? `${riskColor}12` : 'rgba(255,255,255,0.9)',
          boxShadow: isReady ? `0 4px 15px ${riskColor}25` : undefined,
          transition: 'all 0.4s ease',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <ShieldAlert color={riskColor} size={26} />
          <p style={{ fontSize: '1rem', fontWeight: 700, color: riskColor }}>
            {riskData.alerts[lang]}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {['en', 'ta', 'te'].map(l => (
            <button
              key={l}
              onClick={() => setLang(l)}
              style={{
                background: lang === l ? 'white' : 'transparent',
                color: lang === l ? '#2563eb' : '#64748b',
                border: `1px solid ${lang === l ? '#2563eb' : '#e2e8f0'}`,
                padding: '0.25rem 0.8rem',
                borderRadius: 8,
                cursor: 'pointer',
                fontWeight: 700,
                fontSize: '0.8rem',
                textTransform: 'uppercase',
              }}
            >
              {l}
            </button>
          ))}
        </div>
      </header>

      {/* LEFT — Risk Panel */}
      <aside className="sidebar-left">
        <div className="glass-panel" style={{ padding: '1.5rem', background: '#fff', height: '100%' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <Activity size={20} color="#0284c7" />
            <span className="text-gradient" style={{ fontWeight: 800, fontSize: '1.2rem' }}>
              Advaya Risk Engine
            </span>
          </h3>
          {/* Monsoon Simulation Toggle */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.75rem 1rem',
              borderRadius: 12,
              marginBottom: '1rem',
              background: monsoon
                ? 'linear-gradient(135deg, #dc2626 0%, #ea580c 100%)'
                : '#f1f5f9',
              transition: 'all 0.3s ease',
              cursor: 'pointer',
            }}
            onClick={() => setMonsoon(!monsoon)}
          >
            <div>
              <div style={{
                fontSize: '0.78rem',
                fontWeight: 700,
                color: monsoon ? '#fff' : '#475569',
              }}>
                🌧️ Monsoon Simulation
              </div>
              <div style={{
                fontSize: '0.65rem',
                color: monsoon ? '#fecaca' : '#94a3b8',
                marginTop: 2,
              }}>
                {monsoon ? 'Cyclone Michaung (2023) active' : 'Click to simulate heavy rainfall'}
              </div>
            </div>
            <div
              style={{
                width: 44,
                height: 24,
                borderRadius: 12,
                background: monsoon ? '#fff' : '#cbd5e1',
                position: 'relative',
                transition: 'background 0.3s',
              }}
            >
              <div
                style={{
                  width: 18,
                  height: 18,
                  borderRadius: '50%',
                  background: monsoon ? '#dc2626' : '#fff',
                  position: 'absolute',
                  top: 3,
                  left: monsoon ? 23 : 3,
                  transition: 'all 0.3s',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                }}
              />
            </div>
          </div>

          {/* Score Card */}
          <div
            className="glass-panel"
            style={{
              padding: '1.25rem',
              background: '#f8fafc',
              border: `1.5px solid ${riskColor}40`,
              marginBottom: '1rem',
              transition: 'border-color 0.4s',
            }}
          >
            <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: 600, marginBottom: 4 }}>
              Composite Risk Score
            </div>
            <div style={{ fontSize: '3rem', fontWeight: 800, color: riskColor, lineHeight: 1 }}>
              {riskData.score}
              <span style={{ fontSize: '1.1rem', color: '#94a3b8', fontWeight: 600 }}>/100</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: riskColor, fontWeight: 700, marginTop: 6, letterSpacing: 0.5 }}>
              {riskData.level} RISK
            </div>
          </div>

          {/* Weather Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            {[
              { icon: <CloudRain size={20} color="#2563eb" />, label: 'Rainfall',     val: weatherData.rainfall },
              { icon: <Wind      size={20} color="#64748b" />, label: 'Wind Speed',   val: weatherData.windSpeed },
              { icon: <Droplets  size={20} color="#0284c7" />, label: 'Humidity',     val: weatherData.humidity },
              { icon: <CloudRain size={20} color="#f97316" />, label: '12h Forecast', val: `${weatherData.forecast12h} mm` },
            ].map(({ icon, label, val }) => (
              <div
                key={label}
                className="glass-panel"
                style={{ padding: '1rem', textAlign: 'center', background: '#fff' }}
              >
                <div style={{ marginBottom: 6 }}>{icon}</div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>{label}</div>
                <div style={{ fontWeight: 800, marginTop: 4, color: '#0f172a', fontSize: '1rem' }}>{val}</div>
              </div>
            ))}
          </div>

          {/* Instructions */}
          <div
            style={{
              marginTop: '1.5rem',
              background: '#f1f5f9',
              borderRadius: 12,
              padding: '0.75rem 1rem',
              fontSize: '0.78rem',
              color: '#475569',
              lineHeight: 1.6,
            }}
          >
            <MapPin size={13} style={{ display: 'inline', marginRight: 4 }} />
            <strong>How to use:</strong> Click on any point on the map. The system will fetch live weather for
            that location and fuse it with the nearest Prithvi-EO-2.0 satellite
            prediction to compute a real-time risk score.
          </div>
        </div>
      </aside>

      {/* CENTRE — Interactive Map */}
      <main className="map-container">
        <MapComponent onPrediction={handlePrediction} monsoon={monsoon} />
      </main>

      {/* RIGHT — History */}
      <aside className="sidebar-right">
        <div className="glass-panel" style={{ padding: '1.5rem', background: '#fff', height: '100%' }}>
          <h3 style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '1.25rem', color: '#0f172a' }}>
            Model Performance
          </h3>

          {(tileData ? [
            { label: 'Dataset',     val: 'Sen1Floods11' },
            { label: 'Tile ID',     val: tileData.image_id || '—' },
            { label: 'Flood Detected', val: `${tileData.flood_pct ?? '—'}%` },
            { label: 'Accuracy',    val: tileData.accuracy != null ? (tileData.accuracy*100).toFixed(1)+'%' : '—' },
            { label: 'F1 Score',    val: tileData.f1  != null ? tileData.f1.toFixed(3)  : '—' },
            { label: 'IoU',         val: tileData.iou != null ? tileData.iou.toFixed(3) : '—' },
            { label: 'Model',       val: 'Prithvi-EO-2.0-300M' },
          ] : [
            { label: 'Dataset',     val: 'Sen1Floods11' },
            { label: 'Test Images', val: '67 tiles' },
            { label: 'Accuracy',    val: '92.94 %' },
            { label: 'Avg F1',      val: '0.461' },
            { label: 'Avg IoU',     val: '0.368' },
            { label: 'Model',       val: 'Prithvi-EO-2.0-300M' },
          ]).map(({ label, val }) => (
            <div
              key={label}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '0.6rem 0',
                borderBottom: '1px solid #f1f5f9',
                fontSize: '0.85rem',
              }}
            >
              <span style={{ color: '#64748b', fontWeight: 600 }}>{label}</span>
              <span style={{ color: '#0f172a', fontWeight: 700 }}>{val}</span>
            </div>
          ))}

          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '1rem', color: '#0f172a' }}>
              Live Satellite Overlays
            </h3>
            <div className="glass-panel" style={{
              padding: '0.75rem 1rem',
              background: '#eff6ff',
              border: '1px solid #bfdbfe',
              fontSize: '0.8rem',
              color: '#1e40af',
              lineHeight: 1.6,
              marginBottom: '0.75rem',
            }}>
              🛰️ <strong>Sentinel-2 NDWI</strong> — Real satellite water
              detection over Tamil Nadu &amp; Andhra Pradesh coast (last 60 days,
              least-cloud mosaic).
            </div>
            <div className="glass-panel" style={{
              padding: '0.75rem 1rem',
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              fontSize: '0.8rem',
              color: '#166534',
              lineHeight: 1.6,
            }}>
              🤖 <strong>67 Prithvi-EO-2.0 overlays</strong> from global test
              split — real model predictions across 6 continents.
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
