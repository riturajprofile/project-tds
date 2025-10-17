#!/usr/bin/env python3
"""
Test server to capture evaluation URL callbacks.
Run this to see what data the evaluation server sends to your evaluation_url.

Usage:
    python test.py

Then use this URL as your evaluation_url:
    http://localhost:8001/evaluation-callback
    or
    http://your-server-ip:8001/evaluation-callback
"""

import json
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

app = FastAPI(title="Evaluation URL Test Server")

# Store all received callbacks
callbacks_received: List[Dict] = []


@app.post("/evaluation-callback")
async def evaluation_callback(request: Request):
    """
    This endpoint captures what the evaluation server sends.
    Use this URL when you need to test evaluation callbacks.
    """
    try:
        # Capture all request details
        body = await request.json()
        headers = dict(request.headers)
        
        callback_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "body": body,
            "headers": headers,
            "client_ip": request.client.host if request.client else "unknown",
            "method": request.method,
            "url": str(request.url)
        }
        
        # Store the callback
        callbacks_received.append(callback_data)
        
        # Print to console for immediate visibility
        print("\n" + "=" * 80)
        print(f"🔔 EVALUATION CALLBACK RECEIVED at {callback_data['timestamp']}")
        print("=" * 80)
        print(f"Client IP: {callback_data['client_ip']}")
        print(f"\nBody Data:")
        print(json.dumps(body, indent=2))
        print("\nHeaders:")
        for key, value in headers.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Callback received and logged",
                "timestamp": callback_data['timestamp'],
                "data_received": body
            }
        )
    
    except Exception as e:
        print(f"❌ Error processing callback: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/")
async def root():
    """Home page with instructions"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation URL Test Server</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
            }}
            .endpoint {{
                background: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                font-family: monospace;
            }}
            .info {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #2196F3;
            }}
            .count {{
                font-size: 24px;
                color: #4CAF50;
                font-weight: bold;
            }}
            pre {{
                background: #263238;
                color: #aed581;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Evaluation URL Test Server</h1>
            
            <div class="info">
                <h3>📍 Your Evaluation Callback URL:</h3>
                <div class="endpoint">
                    POST http://localhost:8001/evaluation-callback
                </div>
                <p>Use this URL as your <code>evaluation_url</code> when testing tasks.</p>
            </div>
            
            <div class="info">
                <h3>📊 Callbacks Received:</h3>
                <div class="count">{len(callbacks_received)}</div>
                <p><a href="/callbacks">View all callbacks →</a></p>
            </div>
            
            <h3>🚀 How to Use:</h3>
            <ol>
                <li>This server is running on port 8001</li>
                <li>When submitting a task, use the callback URL above as your <code>evaluation_url</code></li>
                <li>Monitor the console for incoming callbacks</li>
                <li>Visit <a href="/callbacks">/callbacks</a> to see all received data</li>
            </ol>
            
            <h3>📝 Example Task Submission:</h3>
            <pre>{{
  "task": "test-task",
  "email": "your@email.com",
  "round": 1,
  "brief": "Test brief",
  "evaluation_url": "http://localhost:8001/evaluation-callback",
  "nonce": "test-nonce-123",
  "secret": "your-secret"
}}</pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/callbacks")
async def get_callbacks():
    """View all received callbacks"""
    return JSONResponse(
        status_code=200,
        content={
            "total_callbacks": len(callbacks_received),
            "callbacks": callbacks_received
        }
    )


@app.get("/callbacks/latest")
async def get_latest_callback():
    """Get the most recent callback"""
    if not callbacks_received:
        return JSONResponse(
            status_code=404,
            content={"message": "No callbacks received yet"}
        )
    
    return JSONResponse(
        status_code=200,
        content=callbacks_received[-1]
    )


@app.get("/clear")
async def clear_callbacks():
    """Clear all stored callbacks"""
    global callbacks_received
    count = len(callbacks_received)
    callbacks_received = []
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Cleared {count} callbacks",
            "remaining": len(callbacks_received)
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "callbacks_received": len(callbacks_received)
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 Evaluation URL Test Server Starting...")
    print("=" * 80)
    print(f"📍 Callback URL: http://localhost:8001/evaluation-callback")
    print(f"🌐 Web Interface: http://localhost:8001")
    print(f"📊 View Callbacks: http://localhost:8001/callbacks")
    print(f"🔍 Latest Callback: http://localhost:8001/callbacks/latest")
    print(f"🧹 Clear Callbacks: http://localhost:8001/clear")
    print("=" * 80)
    print("Waiting for evaluation callbacks...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
