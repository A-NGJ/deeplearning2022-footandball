#!/bin/sh

# List of GPU queues
# gpua100, gpuv100, gpua10, gpua40

### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J footandball_train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Choose cpu model
###BSUB -R "select[model == XeonGold6226R]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### request RAM system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
### please uncomment the following line and put in your e-mail address,
### if you want to receive e-mail notifications on a non-default address
#BSUB -u s210500@student.dtu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_out/training%J.out
#BSUB -e job_out/training%J.err
# -- end of LSF options --

# Load environmental variables
source ./dev.env

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}

# Activate venv
module load python3/3.10.7
source ${REPO}/venv/bin/activate

if [[ $? -ne 0 ]]; then
    exit 1
fi

## run training
python3 train_detector.py --config config1.txt --run-dir ${REPO}/runs/train/${date}

if [[ $? -ne 0 ]]; then
    exit 1
fi
