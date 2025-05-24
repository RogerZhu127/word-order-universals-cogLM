#!/bin/bash
    
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=watgpu408,watgpu508,watgpu608
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=y485zhu@uwaterloo.ca

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j-err.out # File to which STDERR will be written

 
# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)

LANG=$1
FOLD=$2
GPU=$3
TRAVERSAL=$4
 
# Task to run
SAVE_DIR="work/results/${LANG}/${FOLD}/${TRAVERSAL}/rnn"
fairseq-train --task language_modeling "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/fairseq" --save-dir "${SAVE_DIR}" --arch rnn_small_lm --share-decoder-input-output-embed --dropout 0.3 --optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 400 --clip-norm 0.0 --warmup-init-lr 1e-07 --tokens-per-sample 128 --sample-break-mode none --max-tokens 512 --update-freq 4 --no-epoch-checkpoints --max-epoch 10 --device-id "${GPU}" --fp16 --reset-optimizer | tee  "${SAVE_DIR}/log.txt" 
fairseq-eval-lm "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/fairseq" --path "${SAVE_DIR}/checkpoint_last.pt" --tokens-per-sample 128 --gen-subset "test" --output-word-probs 2> "${SAVE_DIR}/test.results"
python src/get_sentence_scores.py -i "${SAVE_DIR}/test.results" -O "${SAVE_DIR}/test.scores"