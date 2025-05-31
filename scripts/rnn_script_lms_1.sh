#!/bin/bash
#SBATCH --job-name=lang_batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=u22
#SBATCH --qos=m5
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

LANG=$1
FOLD=$2
GPU=$3
 
# Task to run
SAVE_DIR="work/results/${LANG}/${FOLD}/rnn"
mkdir -p "${SAVE_DIR}"

fairseq-train --task language_modeling "work/tree_per_line/${LANG}/${FOLD}/fairseq" --save-dir "${SAVE_DIR}" --arch rnn_small_lm --share-decoder-input-output-embed --dropout 0.3 --optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 400 --clip-norm 0.0 --warmup-init-lr 1e-07 --tokens-per-sample 128 --sample-break-mode none --max-tokens 512 --update-freq 4 --no-epoch-checkpoints --max-epoch 10 --device-id "${GPU}" --fp16 --reset-optimizer | tee  "${SAVE_DIR}/log.txt" 
fairseq-eval-lm "work/tree_per_line/${LANG}/${FOLD}/fairseq" --path "${SAVE_DIR}/checkpoint_last.pt" --tokens-per-sample 128 --gen-subset "test" --output-word-probs 2> "${SAVE_DIR}/test.results"
python src/get_sentence_scores.py -i "${SAVE_DIR}/test.results" -O "${SAVE_DIR}/test.scores"
