# Piper_Train-Docker-RTX4080

# Run Docker container "https://hub.docker.com/r/thedeem/piper-training"
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/humanity/piper:/workspace thedeem/piper-training:1.0.0

# Go into virtual environment
source .venv/bin/activate

# Install numpy
pip install numpy==1.24.4

# Install tochmetrics
pip install torchmetrics==0.11.4

cython>=0.29.0,<1
librosa>=0.9.2,<1
piper-phonemize~=1.1.0
numpy>=1.19.0
onnxruntime>=1.11.0
pytorch-lightning~=1.9.0
onnx



# Start training
python3 -m piper_train --dataset-dir /workspace/output-train/ --accelerator 'gpu' --gpus 1 --batch-size 20 --validation-split 0.0 --num-test-examples 0 --max_epochs 6000 --resume_from_checkpoint /workspace/output-train/lightning_logs/version_11/checkpoints/epoch\=101-step\=12648.ckpt --checkpoint-epochs 5 --precision 32 --max-phoneme-ids 400 --quality high
