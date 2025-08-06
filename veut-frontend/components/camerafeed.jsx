import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import CameraToggleButton from "./cameratogglebutton";

const CameraFeed = ({ shouldAnalyze, onDetections, onImageReceived, onDone, isCameraOn = true }) => {
  const [videoDevices, setVideoDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(null);
  const [isFrontCamera, setIsFrontCamera] = useState(false);
  const webcamRef = useRef(null);

  const checkIfFrontCamera = (label) => /front|user/i.test(label);

  useEffect(() => {
    const getDevices = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");

        const preferredCam =
          videoInputs.find((d) => /back|rear|environment/i.test(d.label)) || videoInputs[0];

        setVideoDevices(videoInputs);
        setDeviceId(preferredCam.deviceId);
        setIsFrontCamera(checkIfFrontCamera(preferredCam.label));

        stream.getTracks().forEach((track) => track.stop());
      } catch (err) {
        console.error("Failed to access camera devices:", err);
      }
    };

    getDevices();
  }, []);

  useEffect(() => {
    const device = videoDevices.find((d) => d.deviceId === deviceId);
    if (device) setIsFrontCamera(checkIfFrontCamera(device.label));
  }, [deviceId, videoDevices]);

  const switchCamera = () => {
    if (videoDevices.length < 2) return;
    const currentIndex = videoDevices.findIndex((d) => d.deviceId === deviceId);
    const nextIndex = (currentIndex + 1) % videoDevices.length;
    setDeviceId(videoDevices[nextIndex].deviceId);
  };

  // Capture & send image when shouldAnalyze turns true
  useEffect(() => {
    const captureAndSend = async () => {
      if (!shouldAnalyze || !webcamRef.current) return;

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      try {
        const blob = await (await fetch(screenshot)).blob();
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        // ✅ Use Vercel proxy API instead of calling backend directly
        const res = await fetch("/api/detect", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        onDetections(data.detections);
        onImageReceived(`data:image/jpeg;base64,${data.annotated_image}`);
      } catch (error) {
        console.error("❌ Detection error:", error);
      } finally {
        onDone();
      }
    };

    captureAndSend();
  }, [shouldAnalyze, onDetections, onImageReceived, onDone]);

  return (
    <div className="relative w-full h-full overflow-hidden rounded-lg bg-black">
      {isCameraOn && (
        <>
          <Webcam
            key={deviceId}
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ deviceId }}
            className={`w-full h-full object-cover ${isFrontCamera ? "scale-x-[-1]" : ""} rounded-lg`}
          />
          <CameraToggleButton onClick={switchCamera} />
        </>
      )}
    </div>
  );
};

export default CameraFeed;
