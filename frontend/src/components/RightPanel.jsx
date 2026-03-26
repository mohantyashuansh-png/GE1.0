import Radar from "./Radar";

export default function RightPanel() {
  return (
    <div className="panel">

      <div className="panel-title">TACTICAL RADAR</div>
      <Radar />

      <div className="panel-title">ENVIRONMENT STATUS</div>
      <div className="terminal">
        Status: UNKNOWN<br />
        Visibility: 0%<br />
        Conditions: Clear
      </div>

      <div className="panel-title">MISSION TIMELINE</div>
      <div className="terminal">
        [SYSTEM] Guardian Eye Initialized...
      </div>

    </div>
  );
}