#!/bin/bash -e
clean=$1
if [ "$clean" == "clean" ]; then
    rm -rf build dist vllm.egg-info
fi
reset
pip uninstall vllm -y
python setup.py sdist
python setup.py bdist_wheel
cd dist
pip install --user --force-reinstall vllm*.whl
pip uninstall torch -y
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

