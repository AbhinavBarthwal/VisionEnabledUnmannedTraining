// components/CameraFeed.jsx
import React, { useEffect, useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import CameraToggleButton from "./cameratogglebutton";

const CameraFeed = ({ isCameraOn = true, isAnalyzing, onDetections }) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!isAnalyzing) {
      clearInterval(intervalRef.current);
      return;
    }

    intervalRef.current = setInterval(async () => {
      if (!webcamRef.current) return;
      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      const blob = await (await fetch(screenshot)).blob();
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const res = await axios.post("http://localhost:8000/detect", formData);
        const detections = res.data.detections;
        drawDetections(detections);
        onDetections(detections);
      } catch (err) {
        console.error("Detection error:", err);
      }
    }, 1000);

    return () => clearInterval(intervalRef.current);
  }, [isAnalyzing]);

  const drawDetections = (detections) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(({ label, bbox }) => {
      const [x, y, w, h] = bbox;
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.font = "16px Arial";
      ctx.fillStyle = "red";
      ctx.fillText(label, x, y - 5);
    });
  };

  return (
    <div className="relative w-full h-full">
      {isCameraOn && (
        <>
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="w-full h-full object-cover"
            videoConstraints={{ facingMode: "environment" }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            width={640}
            height={480}
          />
          <CameraToggleButton />
        </>
      )}
    </div>
  );
};

export default CameraFeed;
