import React from "react";
import { Link } from "react-router-dom";

const cameras = [0, 1, 2, 3].map((index) => ({
  id: index + 1,
  src: `http://localhost:8000/video/${index}`,
}));

function VideoPage() {
  return (
    <div className="page-stack">
      <section className="hero-band compact-hero">
        <div>
          <p className="eyebrow">Live Camera Grid</p>
          <h2>Processed lane feeds</h2>
          <p>
            These streams come from the Flask server inside the Python detector.
            Bounding boxes are drawn before each lane frame is published.
          </p>
        </div>

        <Link className="secondary-action" to="/">
          Back to Dashboard
        </Link>
      </section>

      <section className="camera-grid">
        {cameras.map((camera) => (
          <article className="camera-card" key={camera.id}>
            <div className="camera-heading">
              <h3>Lane {camera.id}</h3>
              <span>MJPEG stream</span>
            </div>
            <div className="camera-frame">
              <img src={camera.src} alt={`Live processed feed for lane ${camera.id}`} />
            </div>
          </article>
        ))}
      </section>

      <section className="flow-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Stream Source</p>
            <h3>How the camera view is produced</h3>
          </div>
        </div>

        <div className="pipeline">
          <span>OpenCV reads lane video</span>
          <span>YOLO draws detections</span>
          <span>Frame saved in memory</span>
          <span>Flask serves /video/lane</span>
        </div>
      </section>
    </div>
  );
}

export default VideoPage;
