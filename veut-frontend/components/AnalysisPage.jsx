import React, { useState } from "react";
import CameraFeed from "./camerafeed";

const AnalysisPage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="relative w-screen h-screen bg-[#001524] overflow-hidden">
      {/* CAMERA FEED FULL SCREEN */}
      <CameraFeed />

      {/* OBJECT DETECTED PANEL */}
      <div
        className="absolute md:bottom-10 md:right-6 bottom-45 left-1/2 md:left-auto transform md:transform-none -translate-x-1/2 md:translate-x-0 w-[90%] md:w-[300px] bg-white/10 backdrop-blur-lg text-white rounded-xl p-4 space-y-2 text-sm md:text-base shadow-xl"
      >
        <h2 className="font-semibold text-lg">Objects Detected</h2>
        <p className="flex justify-between">
          <span>Toolbox:</span> <span>0</span>
        </p>
        <p className="flex justify-between">
          <span>Oxygen Tank:</span> <span>0</span>
        </p>
        <p className="flex justify-between">
          <span>Fire Extinguisher:</span> <span>0</span>
        </p>
      </div>

      {/* MOBILE ONLY: RED TOGGLE BUTTON */}
      <button
        className="absolute md:hidden bottom-20 left-1/2 transform -translate-x-1/2 w-[75px] h-[75px] bg-red-600 border-4 border-white rounded-full z-20 shadow-lg"
        onClick={() => setIsAnalyzing((prev) => !prev)}
      />

      {/* DESKTOP ONLY: START/STOP BUTTONS */}
      <div className="hidden md:flex flex-col space-y-4 absolute bottom-6 left-6 z-20">
        <button
          className="bg-green-500 text-white py-2 px-6 rounded-lg text-lg shadow-md"
          onClick={() => setIsAnalyzing(true)}
          disabled={isAnalyzing}
        >
          START
        </button>
        <button
          className="bg-red-500 text-white py-2 px-6 rounded-lg text-lg shadow-md"
          onClick={() => setIsAnalyzing(false)}
          disabled={!isAnalyzing}
        >
          STOP
        </button>
      </div>
    </div>
  );
};

export default AnalysisPage;
