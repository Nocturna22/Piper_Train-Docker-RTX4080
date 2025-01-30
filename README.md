# Piper_Train-Docker-RTX4080

# Run Docker container "https://hub.docker.com/r/thedeem/piper-training"
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/humanity/piper:/workspace thedeem/piper-training:1.0.0

# Go into virtual environment
source .venv/bin/activate

# Start training
python3 -m piper_train --dataset-dir /workspace/output-train/ --accelerator 'gpu' --gpus 1 --batch-size 20 --validation-split 0.0 --num-test-examples 0 --max_epochs 6000 --resume_from_checkpoint /workspace/output-train/lightning_logs/version_11/checkpoints/epoch\=101-step\=12648.ckpt --checkpoint-epochs 5 --precision 32 --max-phoneme-ids 400 --quality high
