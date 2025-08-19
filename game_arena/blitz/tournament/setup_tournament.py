#!/usr/bin/env python3
"""Setup script for blitz chess tournaments."""

import os
import sys
from pathlib import Path

def setup_tournament_environment():
    """Set up the necessary directories and check dependencies."""
    
    print("🏆 Setting up Blitz Chess Tournament Environment")
    print("=" * 50)
    
    # Create directories
    dirs_to_create = [
        "tournament_data",
        "_data",
        "tournament_data/analysis",
        "tournament_data/reports"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Check Python dependencies
    required_packages = [
        "pandas",
        "matplotlib", 
        "seaborn",
        "numpy",
        "jupyter"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Run a single match: python blitz_match.py")
    print("2. Run a tournament: python run_tournament.py")
    print("3. Analyze results in the generated Jupyter notebooks")
    
    return True

def list_available_models():
    """List commonly available models for tournaments."""
    models = {
        "OpenAI": ["gpt-4.1", "gpt-4.0", "gpt-3.5-turbo"],
        "Google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"],
        "Anthropic": ["claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku"],
        "Meta": ["llama-3.1-70b", "llama-3.1-8b"],
        "Mistral": ["mistral-large", "mistral-medium"]
    }
    
    print("\n🤖 Available Models for Tournaments:")
    print("=" * 40)
    
    for provider, model_list in models.items():
        print(f"\n{provider}:")
        for model in model_list:
            print(f"  - {model}")
    
    print("\nTo run a tournament with custom models:")
    print("python run_tournament.py --models='model1,model2,model3,model4,model5,model6,model7,model8'")

if __name__ == "__main__":
    if setup_tournament_environment():
        list_available_models()
    else:
        sys.exit(1) 