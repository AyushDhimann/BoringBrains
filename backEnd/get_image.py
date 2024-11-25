import os
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime
import json
# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Directory to store images
UPLOAD_FOLDER = 'faiss'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store the latest two screenshots
recent_screenshots = []

# Serve images from the uploads directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Function to save base64 image
def save_image(data_url, metadata):
    # Decode base64 image
    image_data = base64.b64decode(data_url.split(",")[1])

    # Generate filename based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the image to the uploads directory
    with open(filepath, 'wb') as f:
        f.write(image_data)

    # Add metadata to the list
    metadata["filename"] = filename
    metadata["path"] = f"/uploads/{filename}"

    # Keep only the last two screenshots
    recent_screenshots.append(metadata)
    if len(recent_screenshots) > 2:
        recent_screenshots.pop(0)

    return metadata


@app.route("/", methods=["GET"])
def index():
    # Render the webpage with the latest two screenshots
    return render_template("index.html", screenshots=recent_screenshots)


@app.route("/", methods=["POST"])
def receive_screenshot():
    try:
        # Parse JSON data from request
        data = request.json
        screenshot = data.get("screenshot")
        url = data.get("url")
        title = data.get("title")
        timestamp = data.get("timestamp")

        if not screenshot:
            return jsonify({"error": "Screenshot data is required"}), 400

        # Save the image and metadata
        metadata = {"url": url, "title": title, "timestamp": timestamp}
        saved_metadata = save_image(screenshot, metadata)

        return jsonify({"message": "Screenshot received", "metadata": saved_metadata}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
