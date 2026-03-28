import { useEffect, useState } from "react";
import "./styles/global.css";

// ✅ Correct Imports (CASE FIXED)
import VideoPlayer from "./components/VideoPlayer";
import VLMControls from "./components/VLMControls";
import RadarPanel from "./components/RadarPanel";
import AITerminal from "./components/AITerminal";
import TelemetryPanel from "./components/TelemetryPanel";
import AlertsBanner from "./components/AlertsBanner";
import MissionControl from "./components/MissionControl";
import BottomPanels from "./components/BottomPanels";


export default function App() {

  const sounds = {
    emergency: new Audio("/sounds/siren.mp3"),
    target: new Audio("/sounds/targetlock.mp3"),
    warning: new Audio("/sounds/warning.mp3"),
  };

  const [persons, setPersons] = useState([]);
  const [alert, setAlert] = useState("NORMAL");
  const [audioEnabled, setAudioEnabled] = useState(false);

  const enableAudio = () => {
    Object.values(sounds).forEach((s) => {
      s.volume = 0.3;
      s.play().then(() => s.pause()).catch(() => {});
    });
    setAudioEnabled(true);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setPersons([
        {
          id: "P-001",
          status: Math.random() > 0.5 ? "Standing" : "Injured",
          score: Math.floor(Math.random() * 150),
        },
      ]);

        const [logs, setLogs] = useState([
       "[SYSTEM] Boot sequence initiated...",
        "[AI] Awaiting commands..."
      ]);
      setLogs((prev) => [
  ...prev,
  "[AI] Hazard scan complete...",
]);
setLogs((prev) => [
  ...prev.slice(-5), // keep last 5 logs only
  "[AI] New update received..."
]);

      const states = ["NORMAL", "WARNING", "EMERGENCY"];
      const random = states[Math.floor(Math.random() * 3)];
      setAlert(random);

      if (audioEnabled) {
        if (random === "EMERGENCY") sounds.emergency.play();
        if (random === "WARNING") sounds.warning.play();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [audioEnabled]);

  return (
    <div className="app">

      {!audioEnabled && (
        <button onClick={enableAudio}>🔊 Enable Audio</button>
      )}

      <div className="header">
        🚁 GUARDIAN EYE - EDGE COMMAND DECK
      </div>

      <AlertsBanner alert={alert} />

      <div className="top-section">
        <VLMControls />
        <VideoPlayer />
        <RadarPanel />
      </div>

      <TelemetryPanel persons={persons} />

      <AITerminal />
      <MissionControl />
      <BottomPanels />
      <AITerminal logs={logs} />

    </div>
  );
}