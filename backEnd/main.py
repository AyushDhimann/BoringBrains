import asyncio
import secrets
import bcrypt
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import datetime
import subprocess

# File to store token and registered devices
DATA_FILE = Path("data/state.json")
DATA_FILE.parent.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_server():
    """Initialize or load the server state from a JSON file."""
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            if not isinstance(data.get("devices"), dict):
                data["devices"] = {}
    else:
        # Generate a new token
        token = secrets.token_urlsafe(27)
        hashed_token = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
        data = {"token": hashed_token, "devices": {}}
        print(f"\nYour login token (SAVE THIS): {token}\n")

    with open(DATA_FILE, "w") as f:
        json.dump(data, f)
    return data


# Load initial server state
data = init_server()


@app.post("/login")
async def login(request: Request):
    """Endpoint to register a device."""
    try:
        body = await request.json()
        device_id = body.get("device_id")
        token = body.get("token")
        device_name = body.get("device_name", "Unknown Device")

        if not device_id or not token:
            raise HTTPException(status_code=400, detail="Device ID and Token are required.")

        if not bcrypt.checkpw(token.encode(), data["token"].encode()):
            raise HTTPException(status_code=403, detail="Invalid token.")

        data["devices"][device_id] = {
            "name": device_name,
            "registered_at": str(datetime.datetime.now()),
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)

        return {"status": "success", "message": "Device registered successfully."}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/devices")
async def get_devices():
    """Get a list of all registered devices."""
    return {"devices": data["devices"]}


@app.get("/status")
async def status():
    """Health-check endpoint."""
    return {"status": "running", "devices_registered": len(data["devices"])}


def run_script(script, command=None):
    """Run a script and print its output."""
    try:
        print(f"Starting {script}...")
        if command:
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            process = subprocess.Popen(
                ['python', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        print(f"Successfully started {script}")
        return process
    except Exception as e:
        print(f"Error starting {script}: {e}")
        return None


if __name__ == "__main__":
    print("Starting all services...")

    # Start other scripts first
    processes = []
    scripts = [
        ('get_image.py', None),
        ('faiss_always_updater.py', None),
        ('query.py', 'uvicorn query:app --host 0.0.0.0 --port 9000')
    ]

    # Start each script
    for script, command in scripts:
        process = run_script(script, command)
        if process:
            processes.append(process)

    # Start the login server
    print("\nStarting login server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)