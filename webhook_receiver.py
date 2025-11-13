# api.py - Add this at the very top
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# webhook_receiver.py - Simple webhook receiver
from fastapi import FastAPI
import argparse
import json
from datetime import datetime

app = FastAPI(title="Webhook Receiver", version="1.0.0")

@app.post("/webhook")
async def receive_webhook(data: dict):
    """Receive webhook notifications"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🔔 WEBHOOK RECEIVED at {timestamp}")
    print(f"{'='*60}")
    
    # Pretty print the received data
    print(json.dumps(data, indent=2))
    print(f"{'='*60}\n")
    
    return {"status": "received", "timestamp": timestamp}

@app.get("/health")
async def health():
    return {"status": "listening"}

def main():
    parser = argparse.ArgumentParser(description='Webhook receiver')
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    args = parser.parse_args()
    
    import uvicorn
    print(f"🎯 Starting webhook receiver on port {args.port}")
    print(f"📡 Listening for webhooks at http://localhost:{args.port}/webhook")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
