# Oil Spill Detection using Computer Vision

Journey into the unseen world of satellite imagery where machine learning meets environmental science. This interactive full-stack web app uncovers oil spills hidden in SAR (Synthetic Aperture Radar) and satellite images, turning complex data into clear, actionable insights for climate responders, marine researchers, and curious minds alike.

### Technologies Used:
- Frontend: React 18 + TypeScript · Vite · Tailwind CSS · Axios · Lucide React
- Backend: Flask (Python) · TensorFlow · OpenCV · Matplotlib
- Deployment: Vercel (frontend) · Render (backend)
- Additional Tools: CORS · REST API · Cloud model hosting (Google Drive, AWS S3, etc.)

### Features:
- Real-Time Detection – Upload SAR or Infrared images and instantly receive visual segmentation of oil spills.
- Deep Learning Models – Includes U-Net, DeepLabv3, and custom CNN architectures trained on annotated datasets.
- Advanced Metrics – Dice coefficient, IoU, precision, recall, density, area, intensity... all in one detailed response.
- Visual Output – Returns both processed images and density graphs to help interpret spill severity and spread.
- Modern UX – Built with a sleek UI using Tailwind CSS and icon-rich design powered by Lucide React.

### Who Benefits:
- Environmental Analysts needing fast, AI-powered spill assessments
- Oceanographers and climate researchers looking for pixel-level segmentation
- Students & Developers curious about applied computer vision in real-world contexts
- Disaster Response Teams needing instant visual data for rapid decision-making

### Future Goals:
- Integrating graph-based segmentation overlays
- Enabling video stream detection for real-time surveillance
- Supporting mobile-friendly uploads
- Expanding to detect chemical leaks and algal blooms
