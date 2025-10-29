#!/usr/bin/env bash

echo "Detecting GDAL version from system..."

# Check if gdal-config is available
if ! command -v gdal-config &> /dev/null; then
  echo "❌ gdal-config not found. Please install GDAL development libraries first."
  echo "   On Ubuntu: sudo apt install gdal-bin libgdal-dev"
  echo "   On macOS:  brew install gdal"
  exit 1
fi

# Extract the GDAL version
GDAL_VERSION=$(gdal-config --version)
echo "✅ Found GDAL version: $GDAL_VERSION"

echo "Installing Python package: gdal==$GDAL_VERSION"
pip install "gdal==$GDAL_VERSION"

echo "Installing other dependencies from pyproject.toml"
pip install -e .

echo "✅ Installation complete."
