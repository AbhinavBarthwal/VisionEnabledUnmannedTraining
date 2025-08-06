export const config = {
  api: {
    bodyParser: false, // Important for streaming file uploads
  },
};

export default async function handler(req, res) {
  try {
    const response = await fetch("http://10.60.88.63:8000/detect", {
      method: req.method,
      headers: req.headers,
      body: req, // Stream the form-data directly to backend
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    console.error("Proxy error:", error);
    res.status(500).json({ error: "Failed to reach backend" });
  }
}
