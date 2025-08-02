// components/CameraToggleButton.jsx
import { RefreshCw } from "lucide-react";

const CameraToggleButton = ({ onClick }) => (
  <button
    onClick={onClick}
    title="Switch Camera"
    className="absolute top-4 right-4 w-10 h-10 md:hidden bg-white rounded-full p-2 flex items-center justify-center shadow-md z-10"
  >
    <RefreshCw className="text-black" size={24} />
  </button>
);

export default CameraToggleButton;
