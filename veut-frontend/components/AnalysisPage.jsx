import React, { useState, useRef } from "react";
import CameraFeed from "./camerafeed";

const labelMap = {
  "ToolBox": "Toolbox",
  "OxygenTank": "Oxygen Tank",
  "FireExtinguisher": "Fire Extinguisher",
};

const AnalysisPage = () => {
  const [shouldAnalyze, setShouldAnalyze] = useState(false);
  const [counts, setCounts] = useState({
    Toolbox: 0,
    "Oxygen Tank": 0,
    "Fire Extinguisher": 0,
  });
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const cameraFeedRef = useRef(null);

  const handleDetections = (detections) => {
    const newCounts = {
      Toolbox: 0,
      "Oxygen Tank": 0,
      "Fire Extinguisher": 0,
    };

    detections.forEach(({ label }) => {
      const mappedLabel = labelMap[label];
      if (mappedLabel && mappedLabel in newCounts) {
        newCounts[mappedLabel]++;
      }
    });

    setCounts(newCounts);
    setShouldAnalyze(false);
  };

  return (
    <div className="relative w-screen h-screen bg-[#001524] overflow-hidden">
      <CameraFeed
        ref={cameraFeedRef}
        shouldAnalyze={shouldAnalyze}
        onDetections={handleDetections}
        onImageReceived={setAnnotatedImage}
        onDone={() => setShouldAnalyze(false)}
      />

      {/* Annotated Image preview */}
      {annotatedImage && (
        <img className="w-[40%] h-[20vh] "
          src={annotatedImage}
          alt="Annotated Preview"
          style={{
            position: "absolute",
            top: 10,
            left: 10,
            width: 300,
            height: 250,
            border: "2px solid white",
            borderRadius: 8,
            objectFit: "cover",
            zIndex: 1000,
          }}
        />
      )}

      {/* Counts panel */}
      <div className="absolute md:bottom-10 md:right-6 bottom-45 left-1/2 md:left-auto transform md:transform-none -translate-x-1/2 md:translate-x-0 w-[90%] md:w-[300px] bg-white/10 backdrop-blur-lg text-gray-900 rounded-xl p-4 space-y-2  md:text-base shadow-xl">
        <h2 className="font-semibold text-lg">Objects Detected</h2>
        <p className="flex justify-between">
          <span>Toolbox:</span> <span>{counts.Toolbox}</span>
        </p>
        <p className="flex justify-between">
          <span>Oxygen Tank:</span> <span>{counts["Oxygen Tank"]}</span>
        </p>
        <p className="flex justify-between">
          <span>Fire Extinguisher:</span> <span>{counts["Fire Extinguisher"]}</span>
        </p>
      </div>

      {/* Red button triggers immediate capture */}
      <button
        className="absolute md:hidden bottom-20 left-1/2 transform -translate-x-1/2 w-[75px] h-[75px] bg-red-600 border-4 border-white rounded-full z-20 shadow-lg"
        onClick={() => setShouldAnalyze(true)}
      />

      {/* Desktop buttons */}
      <div className="hidden md:flex flex-col space-y-4 absolute bottom-6 left-6 z-20">
        <button
          className="bg-green-500 text-white py-2 px-6 rounded-lg text-lg shadow-md"
          onClick={() => setShouldAnalyze(true)}
        >
          START
        </button>
        <button
          className="bg-red-500 text-white py-2 px-6 rounded-lg text-lg shadow-md"
          
        >
          STOP
        </button>
      </div>
    </div>
  );
};

export default AnalysisPage;
