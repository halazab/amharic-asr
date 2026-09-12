#!/usr/bin/env python3
"""
Setup script for Amharic ASR project.

This script helps set up the project environment securely.
"""

import os
import sys
import json
from pathlib import Path


def setup_kaggle_credentials():
    """Setup Kaggle credentials securely."""
    print("\n=== Kaggle Setup ===")
    print("To use Kaggle datasets, you need to provide your API credentials.")
    print("Get your API token from: https://www.kaggle.com/settings")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print(f"\nKaggle credentials already exist at: {kaggle_json}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            return True
    
    username = input("Enter Kaggle username: ").strip()
    key = input("Enter Kaggle API key: ").strip()
    
    if not username or not key:
        print("Error: Username and key are required")
        return False
    
    # Create directory
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write credentials
    credentials = {"username": username, "key": key}
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    
    print(f"\nKaggle credentials saved to: {kaggle_json}")
    print("File permissions set to 600 (owner read/write only)")
    
    return True


def setup_environment():
    """Setup environment variables."""
    print("\n=== Environment Setup ===")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print(".env file already exists")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            return True
    
    if not env_example.exists():
        print("Error: .env.example not found")
        return False
    
    # Copy example to .env
    with open(env_example) as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print(f".env file created from .env.example")
    print("Please edit .env to add your credentials")
    
    return True


def verify_installation():
    """Verify Python packages are installed."""
    print("\n=== Verifying Installation ===")
    
    required_packages = [
        "numpy",
        "torch",
        "onnxruntime",
        "soundfile",
        "fastapi",
        "uvicorn",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll required packages installed!")
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("Amharic ASR Setup")
    print("=" * 60)
    
    steps = [
        ("Verify Installation", verify_installation),
        ("Setup Environment", setup_environment),
        ("Setup Kaggle", setup_kaggle_credentials),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if not step_func():
                print(f"\n{step_name} failed!")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env file with your credentials")
    print("2. Run: python scripts/train.py --config configs/training.json")
    print("3. Run: python scripts/export.py --model checkpoints/amharic_asr_v1/model.pt --export-onnx --quantize-int8")
    print("4. Run: uvicorn api.server:app --reload")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
