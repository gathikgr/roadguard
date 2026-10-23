import { useRef, useState } from 'react';

export type RoadDetection = {
  frame_second: number;
  model_class: string;
  source_model?: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
};

export type MediaPreviewFile = {
  url: string;
  kind: 'image' | 'video';
  name: string;
};

type Props = {
  media: MediaPreviewFile;
  job: {
    status: string;
    error?: string;
    result?: { detections?: RoadDetection[]; model?: string; location?: { latitude:number; longitude:number; source:string; accuracy_m:number|null } | null; metadata?: { captured_at?:string } };
  };
};

function DetectionBoxes({ detections, width, height }: { detections: RoadDetection[]; width: number; height: number }) {
  if (!width || !height || !detections.length) return null;
  const labelSize = Math.max(14, Math.min(28, width / 28));
  return <svg className="detection-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label={`${detections.length} model detection boxes`}>
    {detections.map((d, index) => {
      const [x1, y1, x2, y2] = d.bbox_xyxy;
      const label = `${d.model_class.replaceAll('_', ' ')}${d.source_model?.includes('YOLO-World') ? ' candidate' : ''} · ${Math.round(d.confidence * 100)}%`;
      return <g key={`${d.frame_second}-${index}`}>
        <rect x={x1} y={y1} width={Math.max(0, x2 - x1)} height={Math.max(0, y2 - y1)} rx={4} className="detection-box" />
        <text x={Math.max(0, x1)} y={Math.max(labelSize, y1 - 8)} fontSize={labelSize} className="detection-label">{label}</text>
      </g>;
    })}
  </svg>;
}

export default function DetectionPreview({ media, job }: Props) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const all = (job?.result?.detections || []) as RoadDetection[];
  const detections = media.kind === 'image'
    ? all
    : all.filter(d => Math.abs(d.frame_second - currentTime) <= 0.55);
  const frames = [...new Set(all.map(d => d.frame_second))].sort((a, b) => a - b);
  const completed = job?.status === 'COMPLETED';

  return <section className="media-preview-panel" aria-live="polite">
    <header className="media-preview-head">
      <div><span className="eyebrow">MODEL EVIDENCE</span><h2>{media.kind === 'image' ? 'Image detections' : 'Video detections'}</h2><p>{media.name} · {job?.result?.model || 'Road damage model'}</p></div>
      <span className={`pill ${completed ? 'preview-ready' : 'pending'}`}>{job?.status || 'UPLOADING'}</span>
    </header>
    <div className="media-preview-stage">
      {media.kind === 'image' ? <div className="preview-media-frame">
        <img src={media.url} alt={`Uploaded road image ${media.name}`} onLoad={event => setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
        {completed && <DetectionBoxes detections={detections} width={imageSize.width} height={imageSize.height} />}
      </div> : <div className="preview-media-frame">
        <video ref={videoRef} src={media.url} controls playsInline preload="metadata" onLoadedMetadata={event => setVideoSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight })} onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)} />
        {completed && <DetectionBoxes detections={detections} width={videoSize.width} height={videoSize.height} />}
      </div>}
      {completed && detections.length === 0 && <div className="no-boxes">{media.kind === 'video' ? `No detections at ${currentTime.toFixed(1)}s. Boxes appear at sampled video frames.` : 'The model found no road-damage boxes in this image.'}</div>}
      {!completed && <div className="preview-processing">{job?.status === 'FAILED' ? `Processing failed: ${job.error || 'worker error'}` : 'Running road-damage detection…'}</div>}
    </div>
    <div className="media-preview-footer"><span>{completed ? `${all.length} AI ${media.kind === 'image' ? 'image detections' : 'detections across sampled frames'}${job.result?.location ? ` · GPS ${job.result.location.latitude.toFixed(5)}, ${job.result.location.longitude.toFixed(5)}` : media.kind === 'image' ? ' · no EXIF GPS found' : ''}${job.result?.metadata?.captured_at ? ` · captured ${job.result.metadata.captured_at}` : ''}` : 'Video detections are sampled once per second.'}</span>
      {media.kind === 'video' && completed && frames.length > 0 && <div className="frame-jumps" aria-label="Detected video timestamps">{frames.map(second => <button key={second} className="button small" onClick={() => { if (videoRef.current) videoRef.current.currentTime = second; setCurrentTime(second); }} type="button">{second.toFixed(1)}s · {all.filter(d => d.frame_second === second).length} boxes</button>)}</div>}
    </div>
  </section>;
}
