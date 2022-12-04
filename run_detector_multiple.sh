#/bin/bash

source ./dev.env
date=$(date +%Y%m%d_%H%M)

soccer_net_path="${DATA_PATH}/soccer_net/tracking/test"

# Run tests on multiple models, in this case form 01 to 10
for j in {1..10}; do
    printf -v pad_j "%02d" $j
    for i in {116..125}; do
        python3 run_detector.py --path "${soccer_net_path}/SNMOT-${i}/img1/000001.jpg" \
            --weights "${REPO}/models/model_20201019_1416_final.pth" -o out.mp4 \
            --device cpu --run-dir "${date}_baseline_${i}" --player-threshold 0.9 \
            --metric-path "${soccer_net_path}/SNMOT-${i}"
    done
done
