#!/bin/bash -e
python setup.py sdist
python setup.py bdist_wheel
cd dist
pip install --user --force-reinstall vllm*.whl
pip uninstall torch -y
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
