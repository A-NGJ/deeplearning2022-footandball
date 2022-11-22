#!/bin/sh

date=$(date +%Y%m%d_%H%M)
mkdir runs/test/$date

### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J training
### -- ask for number of cores (default: 1) --
#BSUB -n 1
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

# data path 
source ./dev.env

# Load env
source ./venv_activate.sh

# run detection
python3 run_detector.py --path video_1sec.mp4 --weights models/model_20201019_1416_final.pth --out_video out_video.avi --device cuda

