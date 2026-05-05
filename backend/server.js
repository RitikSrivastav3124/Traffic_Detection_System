const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);

const PORT = Number(process.env.PORT || 5000);
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || "http://localhost:3000";
const PYTHON_COMMAND = process.env.PYTHON_COMMAND || "python";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const MODEL_SCRIPT = path.join(PROJECT_ROOT, "Model", "detect_details2.py");
const MODEL_WEIGHTS = path.join(
  PROJECT_ROOT,
  "Model",
  "runs",
  "train",
  "ambulance_v1",
  "weights",
  "best.pt"
);
const LANE_SOURCES = [1, 2, 3, 4].map((lane) =>
  path.join(PROJECT_ROOT, "videos", `lane${lane}.mp4`)
);

const io = new Server(server, {
  cors: {
    origin: FRONTEND_ORIGIN,
    methods: ["GET", "POST"],
  },
});

app.use(cors({ origin: FRONTEND_ORIGIN }));
app.use(express.json());

const defaultTraffic = {
  lane1: 0,
  lane2: 0,
  lane3: 0,
  lane4: 0,
  green_lane: 1,
  wait: [0, 0, 0, 0],
  emergency: false,
  mode: "LOADING",
};

let latestData = { ...defaultTraffic };
let lastSentData = "";
let stdoutBuffer = "";
let pythonProcess = null;
let pythonRunning = false;
let startTime = null;
let lastUpdate = null;
let restartTimer = null;
let restartCount = 0;
let lastError = "";

function getLaneSummary(data = latestData) {
  return [1, 2, 3, 4].map((lane, index) => ({
    id: lane,
    vehicles: Number(data[`lane${lane}`] || 0),
    wait: Number(data.wait?.[index] || 0),
    signal: data.green_lane === lane ? "GREEN" : "RED",
    cameraUrl: `http://localhost:8000/video/${index}`,
  }));
}

function buildStatus() {
  return {
    backend: true,
    pythonRunning,
    modelScript: MODEL_SCRIPT,
    modelWeights: MODEL_WEIGHTS,
    sources: LANE_SOURCES,
    latestData,
    lanes: getLaneSummary(),
    lastUpdate,
    startTime,
    restartCount,
    lastError,
    videoServer: "http://localhost:8000",
    socketEvent: "traffic-update",
  };
}

function emitSnapshot() {
  io.emit("traffic-update", {
    ...latestData,
    lanes: getLaneSummary(),
    backendStatus: {
      pythonRunning,
      lastUpdate,
      restartCount,
      lastError,
    },
  });
}

function handlePythonLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return;

  try {
    const parsed = JSON.parse(trimmed);
    const dataString = JSON.stringify(parsed);

    if (dataString !== lastSentData) {
      latestData = parsed;
      lastSentData = dataString;
      lastUpdate = new Date().toISOString();
      emitSnapshot();
    }
  } catch (error) {
    if (trimmed.includes("Traceback") || trimmed.includes("Error")) {
      lastError = trimmed.slice(0, 500);
      console.log("Python:", lastError);
    }
  }
}

function startPython() {
  if (pythonRunning) return;

  clearTimeout(restartTimer);
  stdoutBuffer = "";
  lastError = "";
  startTime = new Date().toISOString();

  const args = [
    MODEL_SCRIPT,
    "--weights",
    MODEL_WEIGHTS,
    "--source",
    ...LANE_SOURCES,
    "--imgsz",
    "416",
  ];

  pythonProcess = spawn(PYTHON_COMMAND, args, {
    cwd: path.join(PROJECT_ROOT, "backend"),
    windowsHide: true,
  });

  pythonRunning = true;
  console.log(`Python model started: ${MODEL_SCRIPT}`);
  emitSnapshot();

  pythonProcess.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk.toString();
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || "";
    lines.forEach(handlePythonLine);
  });

  pythonProcess.stderr.on("data", (chunk) => {
    const message = chunk.toString();
    if (message.includes("Traceback") || message.includes("Error")) {
      lastError = message.slice(0, 500);
      console.log("Python error:", lastError);
      emitSnapshot();
    }
  });

  pythonProcess.on("close", (code) => {
    pythonRunning = false;
    pythonProcess = null;
    restartCount += 1;
    lastError = code === 0 ? "" : `Python exited with code ${code}`;
    console.log(lastError || "Python exited normally");
    emitSnapshot();

    restartTimer = setTimeout(startPython, 2500);
  });
}

function stopPython() {
  clearTimeout(restartTimer);

  if (!pythonProcess) {
    pythonRunning = false;
    return;
  }

  pythonProcess.kill();
  pythonProcess = null;
  pythonRunning = false;
}

app.get("/api/health", (req, res) => {
  res.json({ ok: true, service: "traffic-detection-backend" });
});

app.get("/api/traffic", (req, res) => {
  res.json({ ...latestData, lanes: getLaneSummary() });
});

app.get("/api/status", (req, res) => {
  res.json(buildStatus());
});

app.get("/api/config", (req, res) => {
  res.json({
    lanes: 4,
    minGreenSeconds: 15,
    maxWaitSeconds: 60,
    jsonIntervalFrames: 10,
    modelScript: MODEL_SCRIPT,
    modelWeights: MODEL_WEIGHTS,
    sources: LANE_SOURCES,
    videoStreams: getLaneSummary().map((lane) => lane.cameraUrl),
  });
});

app.post("/api/restart", (req, res) => {
  stopPython();
  restartTimer = setTimeout(startPython, 1000);
  res.json({ ok: true, message: "Python detector restart requested" });
});

io.on("connection", (socket) => {
  socket.emit("traffic-update", {
    ...latestData,
    lanes: getLaneSummary(),
    backendStatus: {
      pythonRunning,
      lastUpdate,
      restartCount,
      lastError,
    },
  });
});

startPython();

server.listen(PORT, () => {
  console.log(`Traffic backend running on http://localhost:${PORT}`);
});

process.on("SIGINT", () => {
  stopPython();
  process.exit(0);
});

process.on("SIGTERM", () => {
  stopPython();
  process.exit(0);
});
