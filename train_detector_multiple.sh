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
#BSUB -R "rusage[mem=16GB]"
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

# Activate venv
module load python3/3.10.7
source ${REPO}/venv/bin/activate

if [[ $? -ne 0 ]]; then
    exit 1
fi

## run training
for i in 11 12 21 22 23 24 25 26 27 28 29 30 
do
	printf -v pad_i "%02d" $i
	mkdir "${REPO}/runs/train/${date}_${pad_i}"
	python train_detector.py --config "${REPO}/configs/config${pad_i}.txt" --run-dir "${date}_${pad_i}"
done


if [[ $? -ne 0 ]]; then
    exit 1
fi
