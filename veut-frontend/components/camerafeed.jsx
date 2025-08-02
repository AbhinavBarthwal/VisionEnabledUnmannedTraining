import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const CameraFeed = ({ shouldAnalyze, onDetections, onImageReceived, onDone }) => {
  const webcamRef = useRef(null);
  const [facingMode, setFacingMode] = useState("environment"); // default to back camera

  // Capture and send image when shouldAnalyze changes to true
  useEffect(() => {
    const captureAndSend = async () => {
      if (!shouldAnalyze || !webcamRef.current) return;

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      try {
        const blob = await (await fetch(screenshot)).blob();
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const res = await axios.post("http://192.168.1.5:8000/detect", formData);

        onDetections(res.data.detections);
        onImageReceived(`data:image/jpeg;base64,${res.data.annotated_image}`);
      } catch (error) {
        console.error("❌ Detection error:", error);
      } finally {
        onDone();
      }
    };

    captureAndSend();
  }, [shouldAnalyze, onDetections, onImageReceived, onDone]);

  const toggleCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  return (
    <div className="relative w-full h-full overflow-hidden rounded-lg bg-black">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode }}
        className="w-full h-full object-cover rounded-lg"
      />

      {/* Camera Switch Button */}
      <button
        onClick={toggleCamera}
        className="absolute top-4 right-4 z-30 bg-white/70 rounded-full p-2 shadow-md"
        title="Switch Camera"
      >
        🔄
      </button>
    </div>
  );
};

export default CameraFeed;
