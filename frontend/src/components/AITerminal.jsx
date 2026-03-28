import { useEffect, useRef } from "react";
import TypeWriter from "react-typewriter-effect";

export default function AITerminal({ logs = [] }) {
  const terminalRef = useRef();

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="panel">
      <div className="panel-title">AI TERMINAL</div>

      <div className="terminal" ref={terminalRef}>
        {logs.map((log, i) => (
          <div key={i}>
            <TypeWriter text={log} typeSpeed={30} />
          </div>
        ))}
      </div>
    </div>
  );
}