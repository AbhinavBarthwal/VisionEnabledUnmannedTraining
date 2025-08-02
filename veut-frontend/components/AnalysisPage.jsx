import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import './AnalysisPage.css';

// === Camera Toggle Button Component ===
const CameraToggleButton = ({ onClick }) => (
  <button
    className="camera-switch-btn"
    onClick={onClick}
    title="Switch Camera"
  >
    <RefreshCw size={30} />
  </button>
);

const AnalysisPage = () => {
  const videoRef = useRef(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [facingMode, setFacingMode] = useState('environment');

  useEffect(() => {
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing webcam:', err);
      }
    };

    getVideo();
  }, [facingMode]);

  return (
    <div className="analysis-container">
      <div className="left-panel">
        <video ref={videoRef} autoPlay playsInline muted className="live-feed" />

        {/* Red toggle button for mobile */}
        <button
          className="center-toggle-btn"
          onClick={() => setIsAnalyzing(!isAnalyzing)}
        />

        {/* Camera switcher */}
        <CameraToggleButton
          onClick={() =>
            setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'))
          }
        />
      </div>

      <div className="right-panel">
        <div className="object-list">
          <h2>objects detected</h2>
          <p>Toolbox: 0</p>
          <p>Oxygen Tank: 0</p>
          <p>Fire Extinguisher: 0</p>
        </div>

        {/* Shown only on desktops */}
        <div className="controls">
          <button
            className="start-btn"
            onClick={() => setIsAnalyzing(true)}
            disabled={isAnalyzing}
          >
            START
          </button>
          <button
            className="stop-btn"
            onClick={() => setIsAnalyzing(false)}
            disabled={!isAnalyzing}
          >
            STOP
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;
