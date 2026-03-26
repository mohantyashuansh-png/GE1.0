import "./styles/global.css";
import LeftPanel from "./components/LeftPanel";
import CenterPanel from "./components/CenterPanel";
import RightPanel from "./components/RightPanel";
import BottomPanel from "./components/BottomPanel";

export default function App() {
  return (
    <div className="app">

      <div className="header">
        🚁 GUARDIAN EYE - EDGE COMMAND DECK
      </div>

      <div className="top-section">
        <LeftPanel />
        <CenterPanel />
        <RightPanel />
      </div>

      <BottomPanel />

    </div>
  );
}