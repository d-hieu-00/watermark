# watermark
&lt;TBD>

## Use WSL and Nivida GPU
- Configuration:
    + python:   3.12
    + OS:       WSL-Ubuntu-24.04
    + Devices:  require one Nivida GPU

## Setting up
```
# Create virtual environment & install packages
python3 -m venv venv
source venv/bin/active

pip install 'tensorflow[and-cuda]' tqdm

# Check CPU devices
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
