#!/usr/bin/env python3
"""Stockfish setup and configuration utilities."""

import os
import platform
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
    return os.path.exists(path) and os.access(path, os.X_OK)


def find_existing_stockfish() -> Optional[str]:
    """Try to find an existing Stockfish installation."""
    system = detect_platform()
    
    if system == "macos":
        paths = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish"
        ]
    elif system == "linux":
        paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/usr/games/stockfish"
        ]
    elif system == "windows":
        paths = [
            "C:\\engines\\stockfish.exe",
            "C:\\Program Files\\stockfish\\stockfish.exe"
        ]
    else:
        paths = []
    
    for path in paths:
        if check_stockfish_installed(path):
            return path
    
    return None


def install_stockfish_instructions() -> None:
    """Print instructions for installing Stockfish."""
    system = detect_platform()
    
    print("\n📦 STOCKFISH INSTALLATION INSTRUCTIONS")
    print("=" * 50)
    
    if system == "macos":
        print("""
For macOS:
  Using Homebrew (recommended):
    brew install stockfish
    
  The binary will be installed at /opt/homebrew/bin/stockfish
  (Apple Silicon) or /usr/local/bin/stockfish (Intel)
""")
    elif system == "linux":
        print("""
For Linux:
  Using apt (Debian/Ubuntu):
    sudo apt update
    sudo apt install stockfish
    
  Using yum/dnf (Fedora/RHEL):
    sudo dnf install stockfish
    
  The binary will typically be at /usr/bin/stockfish or /usr/games/stockfish
""")
    elif system == "windows":
        print("""
For Windows:
  1. Download Stockfish from: https://stockfishchess.org/download/
  2. Extract to C:\\engines\\stockfish.exe
  3. Or add the directory to your PATH
""")
    else:
        print("""
Platform not recognized. Please:
  1. Download Stockfish from: https://stockfishchess.org/download/
  2. Extract and note the path to the executable
  3. Set the path when initializing MoveQualityAnalyzer
""")
    
    print("=" * 50)


def test_stockfish(path: str) -> bool:
    """Test if Stockfish works at the given path."""
    try:
        import chess.engine
        with chess.engine.SimpleEngine.popen_uci(path) as engine:
            # Simple test position
            board = chess.Board()
            result = engine.analyse(board, chess.engine.Limit(depth=1))
            return True
    except Exception as e:
        print(f"Stockfish test failed: {e}")
        return False


def main():
    """Run setup checks and provide guidance."""
    print("🔧 STOCKFISH SETUP CHECK")
    print("=" * 50)
    
    # Try to find existing installation
    existing_path = find_existing_stockfish()
    
    if existing_path:
        print(f"✅ Found Stockfish at: {existing_path}")
        
        if test_stockfish(existing_path):
            print("✅ Stockfish is working correctly!")
            print(f"\nTo use in your code:")
            print(f'  analyzer = MoveQualityAnalyzer(engine_path="{existing_path}")')
        else:
            print("❌ Stockfish found but not working correctly")
            install_stockfish_instructions()
    else:
        print("❌ Stockfish not found in standard locations")
        install_stockfish_instructions()


if __name__ == "__main__":
    main()

