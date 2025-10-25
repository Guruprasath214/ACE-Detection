#!/usr/bin/env python3
"""
Simple script to run the ACE Detection API server
"""

import uvicorn
import sys
import socket
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def find_free_port(start_port=8000, max_port=8010):
    """Find a free port starting from start_port."""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Run the API server."""
    try:
        from api.serve import app
        
        # Find a free port
        port = find_free_port()
        if port is None:
            print("Error: No free ports available between 8000-8010")
            return False
        
        print("Starting ACE Detection API server...")
        print(f"Server will be available at: http://127.0.0.1:{port}")
        print(f"API documentation: http://127.0.0.1:{port}/docs")
        print("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
