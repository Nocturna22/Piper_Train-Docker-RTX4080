# Piper_Train-Docker-RTX4080

# Run Docker container "https://hub.docker.com/r/thedeem/piper-training"
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/humanity/piper:/workspace thedeem/piper-training:1.0.0

# Go into virtual environment
source .venv/bin/activate

# Install Pip
python3 -m pip install pip==23.3.1

# Install numpy
pip install numpy==1.24.4

# install pytorch this version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# change requirements.txt
cython>=0.29.0,<1
librosa>=0.9.2,<1
piper-phonemize~=1.1.0
numpy>=1.19.0
onnxruntime>=1.11.0
pytorch-lightning~=1.9.0
onnx

# run
pip3 install --upgrade wheel setuptools

# run
pip3 install -e .

# Install tochmetrics
pip install torchmetrics==0.11.4



# Start training
python3 -m piper_train --dataset-dir /workspace/output-train/ --accelerator 'gpu' --gpus 1 --batch-size 20 --validation-split 0.0 --num-test-examples 0 --max_epochs 6000 --resume_from_checkpoint /workspace/output-train/lightning_logs/version_11/checkpoints/epoch\=101-step\=12648.ckpt --checkpoint-epochs 5 --precision 32 --max-phoneme-ids 400 --quality high
