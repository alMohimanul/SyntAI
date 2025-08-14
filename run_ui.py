#!/usr/bin/env python3
"""
PaperWhisperer - New Clean UI Launcher

Launches the clean, minimalist interface for PaperWhisperer.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the new UI"""
    ui_path = Path(__file__).parent / "frontend" / "ui.py"

    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        return

    print("🚀 Launching PaperWhisperer Clean UI...")
    print("🌐 The interface will open in your browser")
    print("🔄 Press Ctrl+C to stop the server")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_path),
                "--server.headless=true",
                "--server.address=localhost",
                "--server.port=8501",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down PaperWhisperer...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")


if __name__ == "__main__":
    main()
