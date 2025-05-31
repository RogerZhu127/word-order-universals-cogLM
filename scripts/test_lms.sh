#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU=0
for FOLD in 0
do
    for LANG in 0000000
    do
        # LSTM LM
        sbatch $SCRIPT_DIR/lstm_script_lms_1.sh $LANG $FOLD $GPU
        
    done
done
