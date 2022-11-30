#!/bin/sh

# List of GPU queues
# gpua100, gpuv100, gpua10, gpua40

### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J footandball_train
### -- ask for number of cores (default: 1) --
#BSUB -n 8
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
###BSUB -u s<number>@student.dtu.dk
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

# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
    mkdir ${REPO}/job_out
fi

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_1

# Activate venv
module load python3/3.10.7
source ${REPO}/venv/bin/activate

if [[ $? -ne 0 ]]; then
    exit 1
fi

## run training
python3 train_detector.py --config config_01.txt --run-dir ${date}_1


date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_2
python3 train_detector.py --config config_02.txt --run-dir ${date}_2

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_3
python3 train_detector.py --config config_03.txt --run-dir ${date}_3

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_4
python3 train_detector.py --config config_04.txt --run-dir ${date}_4

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_5
python3 train_detector.py --config config_05.txt --run-dir ${date}_5

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_6
python3 train_detector.py --config config_06.txt --run-dir ${date}_6

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_7
python3 train_detector.py --config config_07.txt --run-dir ${date}_7

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_8
python3 train_detector.py --config config_08.txt --run-dir ${date}_8

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_9
python3 train_detector.py --config config_09.txt --run-dir ${date}_9

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_10
python3 train_detector.py --config config_10.txt --run-dir ${date}_10

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_11
python3 train_detector.py --config config_11.txt --run-dir ${date}_11

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}_12
python3 train_detector.py --config config_12.txt --run-dir ${date}_12

if [[ $? -ne 0 ]]; then
    exit 1
fi
