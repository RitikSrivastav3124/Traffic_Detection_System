import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000", {
  transports: ["websocket"],
});

const emptyTraffic = {
  lane1: 0,
  lane2: 0,
  lane3: 0,
  lane4: 0,
  green_lane: 1,
  wait: [0, 0, 0, 0],
  emergency: false,
  mode: "LOADING",
  backendStatus: {
    pythonRunning: false,
    lastUpdate: null,
    restartCount: 0,
    lastError: "",
  },
};

function formatTime(value) {
  if (!value) return "Waiting for model";
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function Dashboard() {
  const [traffic, setTraffic] = useState(emptyTraffic);
  const [connected, setConnected] = useState(socket.connected);

  useEffect(() => {
    const onTraffic = (newData) => setTraffic({ ...emptyTraffic, ...newData });
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);

    socket.on("traffic-update", onTraffic);
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);

    fetch("http://localhost:5000/api/traffic")
      .then((response) => response.json())
      .then((data) => setTraffic({ ...emptyTraffic, ...data }))
      .catch(() => {});

    return () => {
      socket.off("traffic-update", onTraffic);
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
    };
  }, []);

  const lanes = useMemo(
    () =>
      [1, 2, 3, 4].map((lane, index) => ({
        id: lane,
        vehicles: Number(traffic[`lane${lane}`] || 0),
        wait: Number(traffic.wait?.[index] || 0),
        signal: traffic.green_lane === lane ? "GREEN" : "RED",
      })),
    [traffic]
  );

  const totalVehicles = lanes.reduce((sum, lane) => sum + lane.vehicles, 0);
  const maxWait = Math.max(...lanes.map((lane) => lane.wait));
  const activeLane = lanes.find((lane) => lane.signal === "GREEN") || lanes[0];
  const backendStatus = traffic.backendStatus || emptyTraffic.backendStatus;

  return (
    <div className="page-stack">
      <section className="hero-band">
        <div>
          <p className="eyebrow">Live Control Room</p>
          <h2>Adaptive traffic signal monitoring</h2>
          <p>
            The system reads four lane feeds, detects vehicles and ambulances,
            then selects the signal lane using emergency, wait time, and traffic
            density priority.
          </p>
        </div>

        <div className="hero-actions">
          <Link className="primary-action" to="/video">
            Open Camera Grid
          </Link>
          <span className={`connection-pill ${connected ? "online" : "offline"}`}>
            {connected ? "Socket connected" : "Socket offline"}
          </span>
        </div>
      </section>

      <section className="metric-grid" aria-label="Traffic summary">
        <article className="metric-tile">
          <span>Total Vehicles</span>
          <strong>{totalVehicles}</strong>
          <p>Across all four lanes</p>
        </article>
        <article className="metric-tile accent-green">
          <span>Green Lane</span>
          <strong>Lane {activeLane.id}</strong>
          <p>{activeLane.vehicles} vehicles detected</p>
        </article>
        <article className="metric-tile">
          <span>Maximum Wait</span>
          <strong>{maxWait}s</strong>
          <p>Longest current red-lane wait</p>
        </article>
        <article className={`metric-tile ${traffic.emergency ? "accent-alert" : ""}`}>
          <span>Emergency</span>
          <strong>{traffic.emergency ? "Detected" : "Clear"}</strong>
          <p>Ambulance priority status</p>
        </article>
      </section>

      <section className="content-grid">
        <div className="wide-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Signal State</p>
              <h3>Lane decision board</h3>
            </div>
            <span className="mode-chip">{traffic.mode}</span>
          </div>

          <div className="lane-grid">
            {lanes.map((lane) => (
              <article
                className={`lane-tile ${lane.signal === "GREEN" ? "is-green" : ""}`}
                key={lane.id}
              >
                <div className="lane-title-row">
                  <h4>Lane {lane.id}</h4>
                  <span>{lane.signal}</span>
                </div>
                <div className="lane-count">{lane.vehicles}</div>
                <p>Vehicles detected</p>
                <div className="wait-bar">
                  <span style={{ width: `${Math.min(lane.wait, 60) * 1.66}%` }} />
                </div>
                <small>Waiting {lane.wait}s</small>
              </article>
            ))}
          </div>
        </div>

        <aside className="side-panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Runtime</p>
              <h3>Model bridge</h3>
            </div>
          </div>

          <dl className="status-list">
            <div>
              <dt>Python detector</dt>
              <dd>{backendStatus.pythonRunning ? "Running" : "Starting"}</dd>
            </div>
            <div>
              <dt>Last update</dt>
              <dd>{formatTime(backendStatus.lastUpdate)}</dd>
            </div>
            <div>
              <dt>Restarts</dt>
              <dd>{backendStatus.restartCount || 0}</dd>
            </div>
            <div>
              <dt>Video server</dt>
              <dd>Port 8000</dd>
            </div>
          </dl>

          {backendStatus.lastError ? (
            <div className="error-box">{backendStatus.lastError}</div>
          ) : null}
        </aside>
      </section>

      <section className="flow-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Project Flow</p>
            <h3>Backend and model pipeline</h3>
          </div>
        </div>

        <div className="pipeline">
          <span>Node backend starts Python</span>
          <span>YOLO scans lane frames</span>
          <span>Priority logic selects green lane</span>
          <span>JSON reaches Socket.IO</span>
          <span>Dashboard updates live</span>
        </div>
      </section>
    </div>
  );
}

export default Dashboard;
