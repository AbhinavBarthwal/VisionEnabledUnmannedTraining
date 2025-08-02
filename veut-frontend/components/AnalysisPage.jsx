import React, { useState } from "react";
import CameraFeed from "./camerafeed";

const AnalysisPage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="flex flex-col md:flex-row h-screen bg-[#001524]">
      {/* Left Panel */}
      <div className="flex-1 flex items-center justify-center relative">
        <CameraFeed />
        {/* Mobile toggle button */}
        <button
          className="absolute bottom-6 left-1/2 transform -translate-x-1/2 w-[75px] h-[75px] bg-red-600 border-4 border-white rounded-full z-20"
          onClick={() => setIsAnalyzing((prev) => !prev)}
        ></button>
      </div>

      {/* Right Panel */}
      <div className="w-full md:w-[20%] bg-[#001524] p-4 md:p-6 flex flex-col items-center justify-center space-y-6">
        <div className="w-full max-w-xs bg-white text-black p-4 rounded-xl text-center">
          <h2 className="text-lg font-semibold mb-2">Objects Detected</h2>
          <p className="flex justify-between px-4">Toolbox: 0</p>
          <p className="flex justify-between px-4">Oxygen Tank: 0</p>
          <p className="flex justify-between px-4">Fire Extinguisher: 0</p>
        </div>

        {/* Only visible on desktop */}
        <div className="hidden md:flex flex-col space-y-4">
          <button
            className="bg-green-500 text-white py-3 px-6 rounded-lg text-xl"
            onClick={() => setIsAnalyzing(true)}
            disabled={isAnalyzing}
          >
            START
          </button>
          <button
            className="bg-red-500 text-white py-3 px-6 rounded-lg text-xl"
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
