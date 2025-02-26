#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=1hours
#SBATCH --time=20-00:00:00
#SBATCH --mem=32G

# activate your env here
conda activate neuron_specialization
DATA_DIR=ec30/fairseq-data-bin-sharded/shard0
CHECKPOINT=scripts/mT-big-baseline/checkpoints/checkpoint_baseline_mtbig.pt

# eval
PAIRS=('en-de' 'en-nl' 'en-fr' 'en-es' 'en-ru' 'en-cs' 'en-hi' 'en-bn' 'en-ar' 'en-he' 'en-sv' 'en-da' 'en-it' 'en-pt' 'en-pl' 'en-bg' 'en-kn' 'en-mr' 'en-mt' 'en-ha' 'en-af' 'en-lb' 'en-ro' 'en-oc' 'en-uk' 'en-sr' 'en-sd' 'en-gu' 'en-ti' 'en-am' 'de-en' 'nl-en' 'fr-en' 'es-en' 'ru-en' 'cs-en' 'hi-en' 'bn-en' 'ar-en' 'he-en' 'sv-en' 'da-en' 'it-en' 'pt-en' 'pl-en' 'bg-en' 'kn-en' 'mr-en' 'mt-en' 'ha-en' 'af-en' 'lb-en' 'ro-en' 'oc-en' 'uk-en' 'sr-en' 'sd-en' 'gu-en' 'ti-en' 'am-en')
for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}

    SAVE_DIR=scripts/mT-big-baseline/results
    mkdir -p $SAVE_DIR
    fairseq-generate ${DATA_DIR} \
        --task translation_multi_simple_epoch \
        --langs en,de,nl,sv,da,af,lb,fr,es,it,pt,ro,oc,ru,cs,pl,bg,uk,sr,hi,bn,kn,mr,sd,gu,ar,he,ha,mt,ti,am \
        --lang-pairs $PAIR \
        --source-lang $SRC \
        --target-lang $TGT \
        --remove-bpe 'sentencepiece' \
        --arch transformer_vaswani_wmt_en_de_big \
        --path ${CHECKPOINT} \
        --skip-invalid-size-inputs-valid-test \
        --encoder-langtok src --decoder-langtok \
        --gen-subset test \
        --share-decoder-input-output-embed \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 10000 \
        --beam 5 \
        --seed 222 \
        --results-path ${SAVE_DIR}/${SRC}-${TGT} \
        --fp16

    # sacrebleu chrfpp
    OUT_DIR=${SAVE_DIR}/${SRC}-${TGT}
    grep ^H $OUT_DIR/generate-test.txt | LC_ALL=C sort -V | cut -f3- | sacremoses -l ${TGT} detokenize > $OUT_DIR/test-sys.txt
    grep ^T $OUT_DIR/generate-test.txt | LC_ALL=C sort -V | cut -f2- | sacremoses -l ${TGT} detokenize > $OUT_DIR/test-ref.txt
    sacrebleu $OUT_DIR/test-ref.txt -i $OUT_DIR/test-sys.txt -l ${SRC}-${TGT} > $OUT_DIR/test_detok_bleu.txt
    sacrebleu $OUT_DIR/test-ref.txt -i $OUT_DIR/test-sys.txt -l ${SRC}-${TGT} -m chrf --chrf-word-order 2 > $OUT_DIR/test_detok_chrfpp.txt

    
    # comet
    grep ^S ${OUT_DIR}/generate-test.txt | LC_ALL=C sort -V | awk -F'\t' '{ sub(/.*__[a-z]+__/, ""); print }' | sacremoses -l ${SRC} detokenize > ${OUT_DIR}/test-src.txt
    SOURCE_SENT=${OUT_DIR}/test-src.txt
    HYPOTHESIS=${OUT_DIR}/test-sys.txt
    REFERENCE=${OUT_DIR}/test-ref.txt
    comet-score -s ${SOURCE_SENT} -t ${HYPOTHESIS} -r ${REFERENCE} --quiet --only_system > ${OUT_DIR}/test_comet.txt
    

done   
