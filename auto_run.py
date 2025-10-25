#!/usr/bin/env python3
"""
Automatically run the ACE Detection Framework
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def kill_existing_servers():
    """Kill any existing servers on ports 8000-8010"""
    try:
        # Kill processes on common ports
        for port in range(8000, 8011):
            try:
                subprocess.run(f'netstat -ano | findstr :{port}', shell=True, capture_output=True)
                subprocess.run(f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}\') do taskkill /f /pid %a', shell=True, capture_output=True)
            except:
                pass
    except:
        pass

def run_demo():
    """Run the demo first"""
    print("=" * 60)
    print("RUNNING ACE DETECTION DEMO")
    print("=" * 60)
    
    try:
        # Run demo directly in current directory
        result = subprocess.run([sys.executable, "demo.py"], 
                              capture_output=True, text=True, cwd=".")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Demo error: {e}")
        return False

def run_server():
    """Run the server automatically"""
    print("\n" + "=" * 60)
    print("STARTING ACE DETECTION SERVER")
    print("=" * 60)
    
    try:
        # Kill any existing servers
        kill_existing_servers()
        time.sleep(2)
        
        # Run the server
        print("Starting server...")
        process = subprocess.Popen([sys.executable, "run_server.py"], 
                                 cwd=".",
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if process.poll() is None:
            print("Server started successfully!")
            print("Server is running in the background")
            print("Open your browser and go to: http://127.0.0.1:8000/docs")
            print("To stop the server, close this window or press Ctrl+C")
            
            # Try to open browser automatically
            try:
                webbrowser.open("http://127.0.0.1:8000/docs")
                print("Browser opened automatically!")
            except:
                print("Please manually open: http://127.0.0.1:8000/docs")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def main():
    """Main function to run everything automatically"""
    print("ACE DETECTION FRAMEWORK - AUTO RUNNER")
    print("=" * 60)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run demo first
    demo_success = run_demo()
    
    if demo_success:
        print("\nDemo completed successfully!")
        
        # Ask user if they want to start the server
        print("\n" + "=" * 60)
        print("READY TO START SERVER")
        print("=" * 60)
        
        try:
            # Start server
            server_success = run_server()
            
            if server_success:
                print("\nACE Detection Framework is now running!")
                print("Framework Features:")
                print("   - Text preprocessing and risk scoring")
                print("   - Emotion detection")
                print("   - Harassment classification")
                print("   - Real-time API server")
                print("\nAccess the API at: http://127.0.0.1:8000/docs")
                
                # Keep the script running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nServer stopped by user")
            else:
                print("Failed to start server")
                
        except KeyboardInterrupt:
            print("\nStopped by user")
    else:
        print("Demo failed, please check the setup")

if __name__ == "__main__":
    main()
