#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install visualization dependencies for the trading-strategy-backtester.

This script helps users install the required dependencies for enhanced
visualizations in the trading-strategy-backtester.
"""
import sys
import subprocess
import os

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def main():
    """Install the required packages for enhanced visualizations."""
    packages = ["plotly>=5.14.0", "kaleido>=0.2.1"]
    
    print("This script will install the required packages for enhanced visualizations.")
    print("Packages to be installed:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    confirmation = input("Do you want to continue? (y/n): ")
    if confirmation.lower() not in ["y", "yes"]:
        print("Installation cancelled.")
        return
    
    success = True
    for pkg in packages:
        if not install_package(pkg):
            success = False
    
    if success:
        print("\nAll packages installed successfully!")
        print("You can now use enhanced Plotly visualizations in your backtests.")
        print("Just set 'plot': true in your configuration files.")
    else:
        print("\nSome packages could not be installed.")
        print("Please try installing them manually:")
        for pkg in packages:
            print(f"  pip install {pkg}")

if __name__ == "__main__":
    main()