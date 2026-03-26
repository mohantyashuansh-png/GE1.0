export default function LeftPanel() {
  return (
    <div className="panel">
      <div className="panel-title">MISSION SETUP</div>

      <button className="btn green">💻 USE PC WEBCAM</button>

      <input placeholder="http://192.168.x.x/video" />
      <button className="btn green">CONNECT</button>

      <button className="btn green">📁 UPLOAD DRONE FEED</button>

      <div className="panel-title">MOONDREAM VLM (OFFLINE)</div>

      <button className="btn blue">🌍 SCAN HAZARDS</button>
      <button className="btn orange">📋 TRIAGE VICTIM</button>

      <div className="panel-title">VIP HUNTER</div>
      <input placeholder="Shirt Color" />
      <input placeholder="Pants Color" />

      <button className="btn blue">TEXT LOCK</button>
      <button className="btn purple">AUTO-EXTRACT</button>

      <p>Survivors Found: 0</p>

      <button className="btn red">⛔ END MISSION</button>
    </div>
  );
}