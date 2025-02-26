#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=2hours
#SBATCH --time=20-00:00:00
#SBATCH --mem=32G
#SBATCH --nodelist=ilps-cn117


# activate your env here
conda activate neuron_specialization


# this is the one you downloaded
CHECKPOINT=scripts/mT-big/checkpoints/mT_big_NS_checkpoint.pt
CONFIG=scripts/mT-big/configs/multilingual.yml
SAVE_DIR_HOME=scripts/mT-big/results

# evaluation on Flores en-centric pairs
PAIRS=('en-de' 'en-nl' 'en-fr' 'en-es' 'en-ru' 'en-cs' 'en-hi' 'en-bn' 'en-ar' 'en-he' 'en-sv' 'en-da' 'en-it' 'en-pt' 'en-pl' 'en-bg' 'en-kn' 'en-mr' 'en-mt' 'en-ha' 'en-af' 'en-lb' 'en-ro' 'en-oc' 'en-uk' 'en-sr' 'en-sd' 'en-gu' 'en-ti' 'en-am' 'de-en' 'nl-en' 'fr-en' 'es-en' 'ru-en' 'cs-en' 'hi-en' 'bn-en' 'ar-en' 'he-en' 'sv-en' 'da-en' 'it-en' 'pt-en' 'pl-en' 'bg-en' 'kn-en' 'mr-en' 'mt-en' 'ha-en' 'af-en' 'lb-en' 'ro-en' 'oc-en' 'uk-en' 'sr-en' 'sd-en' 'gu-en' 'ti-en' 'am-en')
for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}
    
    # tok Bleu
    SAVE_DIR=$SAVE_DIR_HOME/${SRC}-${TGT}
    mkdir -p $SAVE_DIR
    bash neuron_specialization/scripts/evaluate.sh \
        --lang-pairs $PAIR \
        --source-lang $SRC \
        --target-lang $TGT \
        --config $CONFIG \
        --checkpoint-name $CHECKPOINT \
        --remove-bpe 'sentencepiece' \
        --skip-invalid-size-inputs-valid-test \
        --gen-subset test \
        --beam 5 --seed 222 \
        --results-path $SAVE_DIR \
        --fp16 --max-tokens 30000 \

    ## detok sacrebleu chrfpp
    OUT_DIR=$SAVE_DIR
    grep ^H $SAVE_DIR/generate-test.txt | LC_ALL=C sort -V | cut -f3- | sacremoses -l ${TGT} detokenize > $SAVE_DIR/test-sys.txt
    grep ^T $SAVE_DIR/generate-test.txt | LC_ALL=C sort -V | cut -f2- | sacremoses -l ${TGT} detokenize > $SAVE_DIR/test-ref.txt
    sacrebleu $SAVE_DIR/test-ref.txt -i $SAVE_DIR/test-sys.txt -l ${SRC}-${TGT} > $SAVE_DIR/test_detok_bleu.txt
    sacrebleu $SAVE_DIR/test-ref.txt -i $SAVE_DIR/test-sys.txt -l ${SRC}-${TGT} -m chrf --chrf-word-order 2 > $SAVE_DIR/test_detok_chrfpp.txt


    ## comet
    grep ^S ${SAVE_DIR}/generate-test.txt | LC_ALL=C sort -V | awk -F'\t' '{ sub(/.*__[a-z]+__/, ""); print }' | sacremoses -l ${SRC} detokenize > ${SAVE_DIR}/test-src.txt
    SOURCE_SENT=${SAVE_DIR}/test-src.txt
    HYPOTHESIS=${SAVE_DIR}/test-sys.txt
    REFERENCE=${SAVE_DIR}/test-ref.txt
    comet-score -s ${SOURCE_SENT} -t ${HYPOTHESIS} -r ${REFERENCE} --quiet --only_system > ${OUT_DIR}/test_comet.txt
    

done