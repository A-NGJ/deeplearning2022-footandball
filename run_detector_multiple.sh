#/bin/bash
# List of GPU queues
# gpua100, gpuv100, gpua10, gpua40

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J footandball_detect
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### request RAM system-memory
#BSUB -R "rusage[mem=8GB]"
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

soccer_net_path="${DATA_PATH}/soccer_net/tracking/test"

# Run tests on multiple models, in this case form 01 to 10
for j in 24; 
do
    printf -v pad_j "%02d" $j
    for i in {116..125}; do
        python3 run_detector.py --path "${soccer_net_path}/SNMOT-${i}/img1/000001.jpg" \
            --weights "${REPO}/runs/train/20221203_2202_${pad_j}/model.pth" -o out.mp4 \
            --device cpu --run-dir "${date}_${pad_j}_${i}" --player-threshold 0.9 \
            --metric-path "${soccer_net_path}/SNMOT-${i}"
    done
done


if [[ $? -ne 0 ]]; then
    exit 1
fi
