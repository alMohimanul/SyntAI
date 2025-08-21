#!/usr/bin/env python3
"""
SyntAI Application Launcher
Starts the new HTML/Tailwind frontend with FastAPI backend
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn

        print("✅ FastAPI dependencies found")
        return True
    except ImportError:
        print("❌ Missing FastAPI dependencies")
        print("Please install: pip install fastapi uvicorn[standard] python-multipart")
        return False


def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting SyntAI Backend Server...")

    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Start the server
    try:
        import uvicorn

        uvicorn.run(
            "backend_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    print("=" * 60)
    print("🧠 SyntAI - Your AI-Powered Research Ops Agent")
    print("=" * 60)
    print("🌟 New HTML/Tailwind CSS Frontend")
    print("📡 FastAPI Backend with existing services integration")
    print("=" * 60)

    if not check_dependencies():
        return

    print("\n📊 Frontend will be available at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("\n" + "=" * 60)

    # Wait a moment for user to read
    time.sleep(2)

    # Open browser automatically after a delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8000")

    import threading

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start the server
    start_server()


if __name__ == "__main__":
    main()
