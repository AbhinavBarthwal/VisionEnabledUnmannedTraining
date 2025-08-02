// components/CameraFeed.jsx
import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import CameraToggleButton from "./cameratogglebutton";

const CameraFeed = ({ isCameraOn = true }) => {
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
            className={`w-full h-full object-cover ${
              isFrontCamera ? "scale-x-[-1]" : ""
            } rounded-lg`}
          />
          <CameraToggleButton onClick={switchCamera} />
        </>
      )}
    </div>
  );
};

export default CameraFeed;
