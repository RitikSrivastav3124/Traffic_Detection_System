import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./Dashboard";
import VideoPage from "./VideoPage";
import "./App.css";

function Shell() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">TD</div>
          <div>
            <p className="eyebrow">Major Project</p>
            <h1>Smart Traffic Detection</h1>
          </div>
        </div>

        <nav className="nav-list" aria-label="Main navigation">
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/video">Live Cameras</NavLink>
        </nav>

        <div className="sidebar-note">
          <span>Pipeline</span>
          <strong>YOLO detection + adaptive signal priority</strong>
          <p>Vehicle counts, emergency detection, waiting time, and green lane decisions update in real time.</p>
        </div>
      </aside>

      <main className="main-panel">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/video" element={<VideoPage />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}

export default App;
