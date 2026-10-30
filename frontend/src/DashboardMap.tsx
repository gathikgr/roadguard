import { useEffect, useMemo, useRef, useState } from 'react';
import { LocateFixed, MapPin, Minus, Plus, Route as RouteIcon, X } from 'lucide-react';

export type MapPoint = { latitude: number; longitude: number };
export type MapHazard = MapPoint & { id: string; hazard_type: string; severity: string; status: string; source?: string; evidence_url?: string | null; description?: string };

type Props = {
  location: MapPoint | null;
  route: MapPoint[] | null;
  hazards: MapHazard[];
  selectedId?: string;
  onSelectHazard: (hazard: MapHazard) => void;
  onClosePreview: () => void;
  onRequestLocation: () => void;
};

const TILE_SIZE = 256;
const API_BASE = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:8001` : 'http://localhost:8001');
const MAP_TILE_URL = import.meta.env.VITE_MAP_TILE_URL || 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const clampLatitude = (latitude: number) => Math.max(-85.05112878, Math.min(85.05112878, latitude));
function project(point: MapPoint, zoom: number) {
  const worldSize = TILE_SIZE * 2 ** zoom;
  const latitude = clampLatitude(point.latitude) * Math.PI / 180;
  return {
    x: (point.longitude + 180) / 360 * worldSize,
    y: (1 - Math.asinh(Math.tan(latitude)) / Math.PI) / 2 * worldSize,
  };
}

export default function DashboardMap({ location, route, hazards, selectedId, onSelectHazard, onClosePreview, onRequestLocation }: Props) {
  const root = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [zoomOverride, setZoomOverride] = useState<number | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  useEffect(() => {
    if (!root.current) return;
    const observer = new ResizeObserver(entries => {
      const box = entries[0]?.contentRect;
      if (box) setSize({ width: Math.round(box.width), height: Math.round(box.height) });
    });
    observer.observe(root.current);
    return () => observer.disconnect();
  }, []);

  const viewPoints = useMemo(() => route?.length ? route : [...(location ? [location] : []), ...hazards.map(({ latitude, longitude }) => ({ latitude, longitude }))], [route, location, hazards]);
  const view = useMemo(() => {
    if (!viewPoints.length || size.width < 1 || size.height < 1) return null;
    const projectedByZoom = (zoom: number) => viewPoints.map(point => project(point, zoom));
    let fittedZoom = 14;
    if (route && route.length > 1) {
      for (let z = 18; z >= 4; z--) {
        const projected = projectedByZoom(z);
        const xs = projected.map(p => p.x), ys = projected.map(p => p.y);
        if (Math.max(...xs) - Math.min(...xs) <= size.width * 0.82 && Math.max(...ys) - Math.min(...ys) <= size.height * 0.78) { fittedZoom = z; break; }
        fittedZoom = z;
      }
    } else {
      fittedZoom = 15;
      for (let z = 18; z >= 4; z--) {
        const projected = projectedByZoom(z);
        const xs = projected.map(p => p.x), ys = projected.map(p => p.y);
        if (Math.max(...xs) - Math.min(...xs) <= size.width * 0.82 && Math.max(...ys) - Math.min(...ys) <= size.height * 0.78) { fittedZoom = z; break; }
        fittedZoom = z;
      }
      fittedZoom = Math.min(15, fittedZoom);
    }
    const zoom = Math.max(3, Math.min(19, zoomOverride ?? fittedZoom));
    const projected = projectedByZoom(zoom);
    const xs = projected.map(p => p.x), ys = projected.map(p => p.y);
    const center = { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: (Math.min(...ys) + Math.max(...ys)) / 2 };
    const origin = { x: center.x - size.width / 2, y: center.y - size.height / 2 };
    const minTileX = Math.floor(origin.x / TILE_SIZE), maxTileX = Math.floor((origin.x + size.width) / TILE_SIZE);
    const minTileY = Math.max(0, Math.floor(origin.y / TILE_SIZE)), maxTileY = Math.min(2 ** zoom - 1, Math.floor((origin.y + size.height) / TILE_SIZE));
    const tiles = [];
    for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
      for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
        const wrappedX = ((tileX % 2 ** zoom) + 2 ** zoom) % 2 ** zoom;
        const url = MAP_TILE_URL.replace('{z}', String(zoom)).replace('{x}', String(wrappedX)).replace('{y}', String(tileY));
        tiles.push({ key: `${zoom}/${wrappedX}/${tileY}`, url, left: tileX * TILE_SIZE - origin.x, top: tileY * TILE_SIZE - origin.y });
      }
    }
    return { zoom, origin, tiles };
  }, [viewPoints, route, size, zoomOverride]);

  const screen = (point: MapPoint) => {
    if (!view) return { x: 0, y: 0 };
    const p = project(point, view.zoom);
    const x = p.x - view.origin.x, y = p.y - view.origin.y;
    return { x, y, left: x, top: y };
  };
  const routePath = route?.length ? route.map((point, index) => { const p = screen(point); return `${index ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(' ') : '';

  return <div className="road-map-root" ref={root}>
    {view?.tiles.map(tile => <img className="road-map-tile" key={tile.key} src={tile.url} alt="" draggable={false} style={{ left: tile.left, top: tile.top }} />)}
    {view && <svg className="road-map-overlay" viewBox={`0 0 ${size.width} ${size.height}`} aria-label={routePath ? 'Planned driving route and nearby hazards' : 'Live map with nearby hazards'}>
      {routePath && <path d={routePath} className="road-map-route-casing" />}
      {routePath && <path d={routePath} className="road-map-route" />}
    </svg>}
    {location && view && <div className="road-map-user" style={screen(location)} aria-label="Your live location" title="Your live location"><span /></div>}
    {route?.length && view && <>
      <div className="road-map-endpoint route-origin" style={screen(route[0])} title="Route start">A</div>
      <div className="road-map-endpoint route-destination" style={screen(route[route.length - 1])} title="Destination">B</div>
    </>}
    {hazards.filter(h => h.status === 'PUBLISHED' || (h.status === 'PENDING_REVIEW' && h.source === 'AI_DETECTION_REVIEW')).map(hazard => {
      const point = screen(hazard);
      const suggestion = hazard.status === 'PENDING_REVIEW';
      const isPreviewed = previewId === hazard.id;
      const previewX = Math.max(105, Math.min(size.width - 105, point.x));
      const previewY = Math.max(150, point.y);
      const previewPoint = { x: previewX, y: previewY, left: previewX, top: previewY };
      const evidenceSrc = hazard.evidence_url?.startsWith('/') ? `${API_BASE}${hazard.evidence_url}` : hazard.evidence_url;
      return <div key={hazard.id}>
        <button key={hazard.id} type="button" className={`road-map-hazard ${hazard.severity.toLowerCase()} ${suggestion ? 'suggestion' : ''} ${selectedId === hazard.id || isPreviewed ? 'selected' : ''} ${isPreviewed ? 'pinging' : ''}`} style={point} title={`${hazard.hazard_type} · ${suggestion ? 'AI suggestion at approximate photo GPS' : hazard.severity}`} aria-label={`Show ${hazard.hazard_type}${suggestion ? ', unverified AI suggestion' : `, ${hazard.severity}`}`} onClick={() => { setPreviewId(hazard.id); onSelectHazard(hazard); }}><MapPin size={17} fill="currentColor" /></button>
        {isPreviewed && <div className="road-map-photo-preview" style={previewPoint}>
          <button className="road-map-photo-close" type="button" aria-label="Close hazard preview" onClick={() => { setPreviewId(null); onClosePreview(); }}><X size={13}/></button>
          {evidenceSrc ? <img src={evidenceSrc} alt={`${hazard.hazard_type} road evidence`} /> : <div className="road-map-photo-empty"><MapPin size={19}/><span>Photo preview unavailable</span></div>}
          <div className="road-map-photo-caption"><b>{hazard.hazard_type}</b><span>{suggestion ? 'Unverified map suggestion' : `${hazard.severity} · Published`}</span></div>
          <span className="road-map-photo-tip" />
        </div>}
      </div>;
    })}
    {!location && !route?.length && hazards.length === 0 && <div className="road-map-empty"><LocateFixed size={22}/><b>No geotagged hazards yet</b><span>Map pins will appear when located reports or detections are available.</span><button type="button" className="button primary" onClick={onRequestLocation}>Enable live location</button></div>}
    <div className="road-map-controls"><button type="button" aria-label="Zoom in" onClick={() => setZoomOverride(Math.min(19, (view?.zoom ?? 14) + 1))}><Plus size={15}/></button><button type="button" aria-label="Zoom out" onClick={() => setZoomOverride(Math.max(3, (view?.zoom ?? 14) - 1))}><Minus size={15}/></button>{route?.length ? <button type="button" aria-label="Fit full route" onClick={() => setZoomOverride(null)}><RouteIcon size={15}/></button> : <button type="button" aria-label="Center live location" onClick={() => { setZoomOverride(null); onRequestLocation(); }}><LocateFixed size={15}/></button>}</div>
    <div className="road-map-attribution"><a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a></div>
    {route?.length ? <div className="road-map-status">A–B · {route.length} route points · zoom {view?.zoom ?? '…'}</div> : <div className="road-map-status">Live location · zoom {view?.zoom ?? '…'}</div>}
  </div>;
}
