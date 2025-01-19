#!/bin/bash

# Submit a job on ucloud (Ubuntu VM). Make sure to select CUDA as a parameter

echo "Enter the server IP address:"
read SERVER_IP

LOCAL_PROJECT_DIR="/Users/rea/Library/Mobile Documents/com~apple~CloudDocs/MSc_IT_and_Cognition/wintersemester2024/Free_Topic/Code/gendered-llm" # Local directory path
SERVER_PROJECT_DIR="/home/ucloud/gendered-llm" # Server project directory path
RESULTS_DIR_SERVER="$SERVER_PROJECT_DIR/training_results_RU_animacy/" # path to training results on server

SSH_KEY_PATH="/Users/rea/.ssh/id_rsa"

#scp -i "$SSH_KEY_PATH" -r "$LOCAL_PROJECT_DIR" ucloud@$SERVER_IP:$SERVER_PROJECT_DIR
rsync -avz -e "ssh -i $SSH_KEY_PATH" --exclude='venv' --exclude='data/raw_UD-data/' "$LOCAL_PROJECT_DIR/" ucloud@$SERVER_IP:"$SERVER_PROJECT_DIR"

# SSH into the server and run commands
ssh "-i $SSH_KEY_PATH" ucloud@$SERVER_IP << 'EOF'

    # Install Python 3.8 (server has python version 3.12.3. We need to downgrade to python 3.8.2)
    export DEBIAN_FRONTEND=noninteractive  # Disable interactive prompts
    sudo apt update
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.8 python3.8-venv python3.8-dev

    # Navigate to the project directory
    cd /home/ucloud/gendered-llm

    # Set up virtual environment with Python 3.8
    python3.8 -m venv venv  # Create virtual environment with Python 3.8
    source venv/bin/activate  # Activate the virtual environment

    # Install required dependencies from the requirements.txt
    pip install -r requirements.txt

    # Run the script in the background using nohup (write output to both output.log and terminal)
    echo "Running the script in the background..."
    nohup python3 scripts/Layer-wise-analysis-mBERT-RU-animacy.py 2>&1 | tee output.log &
    echo "Job started successfully in the background. Check the output.log for progress."


EOF

echo "Job completed successfully!"
# Download data
echo "Downloading results to local machine..."
scp -i "$SSH_KEY_PATH" -r ucloud@$SERVER_IP:"$RESULTS_DIR_SERVER" "$LOCAL_PROJECT_DIR" # download training results