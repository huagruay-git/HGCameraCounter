"""
Test Real-time Dashboard Integration

Run this to test the controller GUI with real-time updates
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from controller.main import main

if __name__ == "__main__":
    print("HG Camera Counter - Controller")
    print("=" * 50)
    print("Starting GUI...")
    print()
    print("Tips:")
    print("1. Click 'Start Service' to begin real-time monitoring")
    print("2. Check 'Dashboard' tab for live updates")
    print("3. Camera status updates every 2 seconds")
    print("4. Event counts update every 5 seconds")
    print("=" * 50)
    print()
    
    main()
