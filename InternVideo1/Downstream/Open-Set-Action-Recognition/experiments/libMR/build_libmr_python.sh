#!/bin/bash

# This script creates a clean temporary environment Python, and then
# builds LibMR's python bindings.

if [ '!' -f setup.py ]; then
  echo Put this script into the same folder as setup.py
  exit 1
fi

echo Step 1: Download virtualenv
wget -O virtualenv-1.9.1.tar.gz --no-check-certificate https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.9.1.tar.gz
tar xvf virtualenv-1.9.1.tar.gz

echo Step 2: Create virtualenv
python virtualenv-1.9.1/virtualenv.py --system-site-packages venv

echo Step 3: Entering virtualenv and installing dependencies
source venv/bin/activate
pip install cython==0.19.1

echo Step 5: Build the extension
rm -f python/libmr.cpp
python setup.py build_ext -i

deactivate

echo The .so should be built in the current folder.
