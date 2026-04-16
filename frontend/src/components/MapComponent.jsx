import React, { useEffect, useState } from 'react';
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  Marker,
  Popup,
  LayersControl,
  useMapEvents,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix default leaflet icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl:       'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl:     'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const API = 'http://127.0.0.1:8000';

// Sentinel region bounds matching backend/sentinel_hub.py BBOXES
// Format: [[south, west], [north, east]] for Leaflet ImageOverlay
const SENTINEL_REGIONS = {
  chennai:      [[12.90, 79.90], [13.30, 80.40]],
  tn_coast:     [[ 8.00, 79.50], [13.50, 80.70]],
  andhra_coast: [[13.50, 79.80], [16.50, 81.00]],
};

// Inner component to handle map click events
function ClickHandler({ onMapClick }) {
  useMapEvents({ click: (e) => onMapClick(e.latlng) });
  return null;
}

export default function MapComponent({ onPrediction, monsoon = false }) {
  const [overlays, setOverlays]       = useState([]);   // pre-computed Prithvi overlays
  const [sentinel, setSentinel]       = useState([]);   // real Sentinel-2 NDWI overlays
  const [clickedPin, setClickedPin]   = useState(null);
  const [highlighted, setHighlighted] = useState(null);
  const [isLoading, setIsLoading]     = useState(false);

  // Load all 67 pre-computed Prithvi overlays
  useEffect(() => {
    fetch(`${API}/api/overlays`)
      .then(r => r.json())
      .then(d => setOverlays(d.overlays || []))
      .catch(e => console.warn('Could not load overlays:', e));
  }, []);

  // Load real Sentinel-2 NDWI overlays for TN & AP coast
  useEffect(() => {
    const regions = Object.keys(SENTINEL_REGIONS);
    Promise.all(
      regions.map(r =>
        fetch(`${API}/static/sentinel/${r}.png`, { method: 'HEAD' })
          .then(res => res.ok ? { region: r, url: `${API}/static/sentinel/${r}.png` } : null)
          .catch(() => null)
      )
    ).then(results => {
      setSentinel(results.filter(Boolean));
      console.log(`Loaded ${results.filter(Boolean).length} Sentinel NDWI overlays`);
    });
  }, []);

  const [elapsed, setElapsed] = useState(0);
  const timerRef = React.useRef(null);

  const handleMapClick = async (latlng) => {
    setClickedPin(latlng);
    setIsLoading(true);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    try {
      // Real-time Prithvi model inference (~28s on CPU)
      const res = await fetch(`${API}/api/predict_live`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ lat: latlng.lat, lon: latlng.lng, monsoon }),
      });
      const data = await res.json();

      if (data.prediction) setHighlighted(data.prediction.image_id);
      else if (data.nearest_overlay) setHighlighted(data.nearest_overlay.image_id);
      if (onPrediction) onPrediction(data);
    } catch (e) {
      console.error('Live inference error:', e);
    } finally {
      clearInterval(timerRef.current);
      setIsLoading(false);
      setElapsed(0);
    }
  };

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      {/* Loading indicator */}
      {isLoading && (
        <div style={{
          position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
          background: 'white', borderRadius: 10, padding: '10px 20px',
          boxShadow: '0 4px 20px rgba(37,99,235,0.25)', zIndex: 1000,
          fontSize: '0.85rem', fontWeight: 700, color: '#2563eb',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
          minWidth: 220,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="spin-icon" style={{ display: 'inline-block', fontSize: '1.1rem' }}>⟳</span>
            Running Prithvi-EO-2.0 inference…
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 500 }}>
            Elapsed: {elapsed}s — may take up to 30s on CPU
          </div>
          <div style={{
            width: '100%', height: 4, background: '#e2e8f0', borderRadius: 2, overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: `${Math.min(100, (elapsed / 30) * 100)}%`,
              background: 'linear-gradient(90deg, #2563eb, #0ea5e9)',
              borderRadius: 2,
              transition: 'width 1s linear'
            }} />
          </div>
        </div>
      )}

      {/* Click hint */}
      {overlays.length > 0 && !clickedPin && (
        <div style={{
          position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(37,99,235,0.9)', borderRadius: 20, padding: '6px 18px',
          boxShadow: '0 4px 12px rgba(37,99,235,0.3)', zIndex: 1000,
          fontSize: '0.8rem', fontWeight: 600, color: 'white',
          pointerEvents: 'none',
        }}>
          Click anywhere on the map to run risk analysis
        </div>
      )}

      <MapContainer
        center={[13.0827, 80.2707]}
        zoom={6}
        style={{ height: '100%', width: '100%' }}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Clean Map">
            <TileLayer
              attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              attribution='&copy; Esri'
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        <ClickHandler onMapClick={handleMapClick} />

        {/* Real Sentinel-2 NDWI water overlays for Tamil Nadu & AP coast */}
        {sentinel.map(({ region, url }) => (
          <ImageOverlay
            key={`sentinel-${region}`}
            url={url}
            bounds={SENTINEL_REGIONS[region]}
            opacity={0.75}
            zIndex={300}
          />
        ))}

        {/* Pre-computed Prithvi flood prediction overlays (67 global tiles) */}
        {overlays.map(o => (
          <ImageOverlay
            key={o.image_id}
            url={`${API}${o.png_url}`}
            bounds={o.bounds}
            opacity={highlighted === o.image_id ? 0.92 : 0.55}
            zIndex={highlighted === o.image_id ? 500 : 200}
          />
        ))}

        {/* Live Prithvi prediction overlay after a click */}
        {highlighted && overlays.find(o => o.image_id === highlighted) === undefined && (
          <ImageOverlay
            key={`live-${highlighted}`}
            url={`${API}/static/live/${highlighted}.png`}
            bounds={clickedPin ? [
              [clickedPin.lat - 0.025, clickedPin.lng - 0.025],
              [clickedPin.lat + 0.025, clickedPin.lng + 0.025],
            ] : [[0,0],[1,1]]}
            opacity={0.9}
            zIndex={600}
          />
        )}

        {/* Clicked pin with risk popup */}
        {clickedPin && (
          <Marker position={[clickedPin.lat, clickedPin.lng]}>
            <Popup>
              <strong>Clicked Location</strong><br />
              Lat: {clickedPin.lat.toFixed(4)}<br />
              Lon: {clickedPin.lng.toFixed(4)}<br />
              <em style={{ fontSize: '0.75rem', color: '#64748b' }}>
                Risk fused from nearest satellite tile + live weather
              </em>
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}
