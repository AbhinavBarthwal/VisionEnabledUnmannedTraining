import React, { useEffect, useRef, useState } from 'react'
import './AnalysisPage.css'

const AnalysisPage = () => {
  const videoRef = useRef(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  useEffect(() => {
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (err) {
        console.error('Error accessing webcam:', err)
      }
    }

    getVideo()
  }, [])

  return (
    <div className="analysis-container">
      <div className="left-panel">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="live-feed"
        />
      </div>

      <div className="right-panel">
        <div className="object-list">
          <h2>objects detected</h2>
          <p>Toolbox: 0</p>
          <p>Oxygen Tank: 0</p>
          <p>Fire Extinguisher: 0</p>
        </div>

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
  )
}

export default AnalysisPage
