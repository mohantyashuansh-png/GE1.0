export default function VoiceControl() {
  const startListening = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.start();

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript.toLowerCase();

      if (text.includes("scan hazards")) {
        alert("Scanning hazards...");
      }

      if (text.includes("lock target")) {
        alert("Target locked!");
      }
    };
  };

  return (
    <button className="mic-btn" onClick={startListening}>
      🎤 ACTIVATE VOICE
    </button>
  );
}