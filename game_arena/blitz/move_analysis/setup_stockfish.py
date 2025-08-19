#!/usr/bin/env python3
"""Setup script for Stockfish chess engine installation and configuration.

This script helps users install and configure Stockfish for move analysis.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def detect_platform() -> str:
    """Detect the current platform."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"


def check_stockfish_installed(path: str) -> bool:
    """Check if Stockfish is installed at the given path."""
    if not os.path.exists(path):
        return False
    
    try:
        result = subprocess.run([path, "--help"], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def find_existing_stockfish() -> Optional[str]:
    """Try to find an existing Stockfish installation."""
    platform_name = detect_platform()
    
    if platform_name == "macos":
        candidates = [
            "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
            "/usr/local/bin/stockfish",     # Homebrew on Intel
            "/usr/bin/stockfish"            # System installation
        ]
    elif platform_name == "linux":
        candidates = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/usr/games/stockfish"
        ]
    elif platform_name == "windows":
        candidates = [
            "C:\\engines\\stockfish.exe",
            "C:\\Program Files\\stockfish\\stockfish.exe",
            "stockfish.exe"  # In PATH
        ]
    else:
        candidates = ["/usr/local/bin/stockfish"]
    
    for candidate in candidates:
        if check_stockfish_installed(candidate):
            return candidate
    
    # Try finding in PATH
    try:
        result = subprocess.run(["which", "stockfish"], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            if check_stockfish_installed(path):
                return path
    except FileNotFoundError:
        pass
    
    return None


def install_stockfish_instructions() -> None:
    """Print installation instructions for each platform."""
    platform_name = detect_platform()
    
    print(f"\n📋 Stockfish Installation Instructions for {platform_name.title()}:")
    print("=" * 60)
    
    if platform_name == "macos":
        print("Option 1 - Using Homebrew (recommended):")
        print("  brew install stockfish")
        print()
        print("Option 2 - Download manually:")
        print("  1. Visit: https://stockfishchess.org/download/")
        print("  2. Download macOS version")
        print("  3. Extract and copy to /usr/local/bin/stockfish")
        print("  4. Make executable: chmod +x /usr/local/bin/stockfish")
        
    elif platform_name == "linux":
        print("Option 1 - Using package manager:")
        print("  Ubuntu/Debian: sudo apt-get install stockfish")
        print("  CentOS/RHEL:   sudo yum install stockfish")
        print("  Arch:          sudo pacman -S stockfish")
        print()
        print("Option 2 - Download manually:")
        print("  1. Visit: https://stockfishchess.org/download/")
        print("  2. Download Linux version")
        print("  3. Extract and copy to /usr/local/bin/stockfish")
        print("  4. Make executable: chmod +x /usr/local/bin/stockfish")
        
    elif platform_name == "windows":
        print("Option 1 - Download from official site:")
        print("  1. Visit: https://stockfishchess.org/download/")
        print("  2. Download Windows version")
        print("  3. Extract to C:\\engines\\stockfish.exe")
        print()
        print("Option 2 - Using chocolatey:")
        print("  choco install stockfish")
        
    else:
        print("Please visit https://stockfishchess.org/download/ for download instructions.")
    
    print()


def test_stockfish(path: str) -> bool:
    """Test that Stockfish is working correctly."""
    print(f"🧪 Testing Stockfish at {path}...")
    
    try:
        # Test basic functionality
        result = subprocess.run([path], 
                              input="uci\nquit\n", 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if "uciok" in result.stdout:
            print("✅ Stockfish is working correctly!")
            
            # Try to get version info
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith("id name"):
                    print(f"   Version: {line}")
                    break
            
            return True
        else:
            print("❌ Stockfish responded but didn't provide expected UCI output")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Stockfish test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing Stockfish: {e}")
        return False


def create_config_file(stockfish_path: str) -> None:
    """Create a configuration file with the Stockfish path."""
    config_dir = Path.home() / ".game_arena"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "stockfish_config.txt"
    with open(config_file, 'w') as f:
        f.write(f"STOCKFISH_PATH={stockfish_path}\n")
    
    print(f"💾 Saved Stockfish path to: {config_file}")


def main():
    """Main setup function."""
    print("🏁 Game Arena - Stockfish Setup")
    print("=" * 40)
    
    # First, try to find existing installation
    existing_path = find_existing_stockfish()
    
    if existing_path:
        print(f"✅ Found existing Stockfish installation: {existing_path}")
        
        if test_stockfish(existing_path):
            create_config_file(existing_path)
            print(f"\n🎉 Setup complete! Stockfish is ready at: {existing_path}")
            
            # Show next steps
            print("\n📖 Next Steps:")
            print("1. Install python-chess if not already installed:")
            print("   pip install chess")
            print("2. Test the move analyzer:")
            print("   python -m game_arena.blitz.move_analysis.move_analyzer <match_directory>")
            
            return 0
        else:
            print("❌ Found Stockfish but it's not working properly")
    
    # If not found or not working, show installation instructions
    print("❌ Stockfish not found or not working")
    install_stockfish_instructions()
    
    print("\n🔄 After installing Stockfish, run this script again to verify the installation.")
    
    return 1


if __name__ == "__main__":
    exit(main())
