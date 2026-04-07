# Frontend Setup Guide

The Hand Sign Detection frontend is a simple vanilla HTML/CSS/JavaScript application that provides real-time webcam-based hand sign detection.

## Quick Start

### Option 1: Using npm/Node.js

```bash
# Install development server
npm install

# Start the frontend (serves on http://localhost:3000)
npm start
```

### Option 2: Using Python

```bash
# Simple HTTP server
python -m http.server 3000
```

### Option 3: Direct File

Open `index.html` directly in a browser. Note: Some features may require serving via HTTP due to browser security restrictions (especially webcam access).

## Architecture

```
Frontend (index.html)
    │
    ├── Upload image → POST /predict → Display result
    │
    └── Webcam stream → Capture frames → POST /predict → Real-time display
```

## Files

| File | Purpose |
|------|---------|
| `index.html` | Main application (HTML + CSS + JS) |
| `package.json` | Node.js package configuration |

## Configuration

The frontend expects the backend API at `http://localhost:8000`. To change this:

1. Open `index.html`
2. Find the `API_BASE_URL` constant
3. Update to your backend URL

```javascript
const API_BASE_URL = 'http://localhost:8000';
// Change to:
const API_BASE_URL = 'https://your-api.example.com';
```

## Features

- **Image Upload**: Drag-and-drop or click to upload hand sign images
- **Webcam Detection**: Real-time hand sign detection from webcam feed
- **Prediction Display**: Shows detected sign with confidence score
- **Responsive Design**: Works on desktop and tablet devices

## Development

### Running with Hot Reload

For development with auto-reload:

```bash
# Using browser-sync
npx browser-sync start --server --files "*.html, *.js, *.css"
```

### Customizing Styles

All styles are in the `<style>` block within `index.html`. Key CSS variables:

```css
/* Main gradient colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent color */
.panel h2 { color: #667eea; }
```

## Building for Production

Since this is a static HTML file, no build step is required. For production:

1. Copy `index.html` to your web server
2. Update `API_BASE_URL` to production backend
3. Ensure CORS is configured on the backend

### Docker Deployment

If you want to containerize the frontend:

```dockerfile
# Dockerfile.frontend
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

```nginx
# nginx.conf
server {
    listen 80;
    root /usr/share/nginx/html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome 80+ | ✅ Full |
| Firefox 75+ | ✅ Full |
| Safari 13+ | ✅ Full |
| Edge 80+ | ✅ Full |
| IE 11 | ❌ Not supported |

## Troubleshooting

### Webcam Not Working

1. Ensure HTTPS or localhost (HTTP) - browsers require secure context
2. Check browser permissions for camera access
3. Try a different browser if issues persist

### API Connection Failed

1. Verify backend is running: `curl http://localhost:8000/health/live`
2. Check CORS configuration in backend
3. Verify `API_BASE_URL` matches backend address

### Prediction Errors

1. Check browser console for error messages
2. Verify image format (JPEG/PNG supported)
3. Ensure model is loaded: `GET /health/ready`
