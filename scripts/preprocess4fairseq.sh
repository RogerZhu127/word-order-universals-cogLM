#!/bin/bash
for FOLD in 0 20000 40000 60000 80000
do
  for LANG in 0000000 0000001 0000010 0000011 0000100 0000101 0000110 0000111 0001000 0001001 0001010 0001011 0001100 0001101 0001110 0001111 0010000 0010001 0010010 0010011 0010100 0010101 0010110 0010111 0011000 0011001 0011010 0011011 0011100 0011101 0011110 0011111 0100000 0100001 0100010 0100011 0100100 0100101 0100110 0100111 0101000 0101001 0101010 0101011 0101100 0101101 0101110 0101111 0110000 0110001 0110010 0110011 0110100 0110101 0110110 0110111 0111000 0111001 0111010 0111011 0111100 0111101 0111110 0111111 1000000 1000001 1000010 1000011 1000100 1000101 1000110 1000111 1001000 1001001 1001010 1001011 1001100 1001101 1001110 1001111 1010000 1010001 1010010 1010011 1010100 1010101 1010110 1010111 1011000 1011001 1011010 1011011 1011100 1011101 1011110 1011111 1100000 1100001 1100010 1100011 1100100 1100101 1100110 1100111 1101000 1101001 1101010 1101011 1101100 1101101 1101110 1101111 1110000 1110001 1110010 1110011 1110100 1110101 1110110 1110111 1111000 1111001 1111010 1111011 1111100 1111101 1111110 1111111
  do
    echo "Processing ${LANG} ${FOLD}"
    SAVE_DIR="work/tree_per_line/${LANG}/${FOLD}/fairseq"
    mkdir -p ${SAVE_DIR}
    cut -f1 "work/tree_per_line/${LANG}/${FOLD}/rnng/spm.vocab" | tail -n +5 | sed "s/$/ 100/g" > "${SAVE_DIR}/fairseq.vocab"
    fairseq-preprocess --only-source --srcdict "${SAVE_DIR}/fairseq.vocab" --trainpref "work/tree_per_line/${LANG}/${FOLD}/trn.subword_tokens.txt" --validpref "work/tree_per_line/${LANG}/${FOLD}/dev.subword_tokens.txt" --testpref "work/tree_per_line/${LANG}/${FOLD}/tst.subword_tokens.txt" --destdir "${SAVE_DIR}" --workers 20

    for TRAVERSAL in "td" "lc-as"
    do
      SAVE_DIR="work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/fairseq"
      mkdir -p ${SAVE_DIR}
      cat "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/trn.actions.txt" | python src/extract_actions.py > "work/tree_per_line/${LANG}/${FOLD}/fairseq/action.vocab"
      cat "work/tree_per_line/${LANG}/${FOLD}/fairseq/fairseq.vocab" "work/tree_per_line/${LANG}/${FOLD}/fairseq/action.vocab" > "work/tree_per_line/${LANG}/${FOLD}/fairseq/fairseq-action.vocab"
      fairseq-preprocess --only-source --srcdict "work/tree_per_line/${LANG}/${FOLD}/fairseq/fairseq-action.vocab" --trainpref "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/trn.actions.rnng_subwords.txt" --validpref "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/dev.actions.rnng_subwords.txt" --testpref "work/tree_per_line/${LANG}/${FOLD}/${TRAVERSAL}/tst.actions.rnng_subwords.txt" --destdir "${SAVE_DIR}" --workers 20
    done
  done
done
