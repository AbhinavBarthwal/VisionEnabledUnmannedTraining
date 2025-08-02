import React, { useEffect, useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import CameraToggleButton from "./cameratogglebutton";

const CameraFeed = ({ shouldAnalyze, onDetections, isCameraOn = true, onImageReceived, onDone }) => {
  const webcamRef = useRef(null);

  useEffect(() => {
    const captureAndSend = async () => {
      if (!shouldAnalyze || !webcamRef.current) return;

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      try {
        const blob = await (await fetch(screenshot)).blob();
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const res = await axios.post("http://localhost:8000/detect", formData);

        onDetections(res.data.detections);
        onImageReceived(`data:image/jpeg;base64,${res.data.annotated_image}`);
      } catch (error) {
        console.error("❌ Detection error:", error);
      } finally {
        onDone(); // reset `shouldAnalyze`
      }
    };

    captureAndSend();
  }, [shouldAnalyze, onDetections, onImageReceived, onDone]);

  return (
    <div className="relative w-full h-full overflow-hidden rounded-lg bg-black">
      {isCameraOn && (
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ facingMode: "environment" }}
          className="w-full h-full object-cover rounded-lg"
        />
      )}
    </div>
  );
};

export default CameraFeed;
