#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=10
#SBATCH --job-name=lsn
#SBATCH --time=20-00:00:00
#SBATCH --mem=118G


# activate your env here
conda activate neuron_specialization


# 1. Extract neurons using validation set
DATA_DIR=ec30/fairseq-data-bin-sharded/shard0
BASELINE_CHECKPOINT_DIR=scripts/mT-big-baseline/checkpoints/checkpoint_baseline_mtbig.pt
SAVE_NEURON_DIR=scripts/mT-big-NS/acts

PAIRS=('en-de' 'en-nl' 'en-sv' 'en-da' 'en-af' 'en-lb' 'en-fr' 'en-es' 'en-it' 'en-pt' 'en-ro' 
       'en-oc' 'en-ru' 'en-cs' 'en-pl' 'en-bg' 'en-uk' 'en-sr' 'en-hi' 'en-bn' 
       'en-kn' 'en-mr' 'en-sd' 'en-gu' 'en-ar' 'en-he' 'en-ha' 'en-mt' 'en-ti' 'en-am'
       'de-en' 'nl-en' 'sv-en' 'da-en' 'af-en' 'lb-en' 'fr-en' 'es-en' 'it-en' 'pt-en' 'ro-en' 
       'oc-en' 'ru-en' 'cs-en' 'pl-en' 'bg-en' 'uk-en' 'sr-en' 'hi-en' 'bn-en' 
       'kn-en' 'mr-en' 'sd-en' 'gu-en' 'ar-en' 'he-en' 'ha-en' 'mt-en' 'ti-en' 'am-en')

for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}

    mkdir -p $SAVE_NEURON_DIR/$PAIR
    python neuron_specialization/toolbox/get_ns_masks.py \
        ${DATA_DIR} \
        --task translation_multi_simple_epoch \
        --langs en,de,nl,sv,da,af,lb,fr,es,it,pt,ro,oc,ru,cs,pl,bg,uk,sr,hi,bn,kn,mr,sd,gu,ar,he,ha,mt,ti,am \
        --lang-pairs $PAIR \
        --source-lang $SRC \
        --target-lang $TGT \
        --remove-bpe 'sentencepiece' \
        --path ${CHECKPOINT_DIR} \
        --sampling-method temperature \
        --skip-invalid-size-inputs-valid-test \
        --encoder-langtok src \
        --decoder-langtok \
        --gen-subset valid \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 30000 \
        --seed 222 \
        --save_neuron_path $NEURON_DIR

done


# 2. get masks
MASK_DIR=scripts/mT-big-NS/masks
mkdir -p $MASK_DIR
python neuron_specialization/toolbox/get_ns_masks.py \
        --neuron-dir $SAVE_NEURON_DIR \
        --mask-format-path lsn_code/mask_format.pt \
        --save-mask-dir $MASK_DIR \
        --save-fig-dir scripts/mT-big-NS \
        --fc1-neuron-dim 4096 \
        --fc1-weight-dim 1024 \
        --en2x-enc-k 95 --en2x-dec-k 95 \
        --x2en-enc-k 95 --x2en-dec-k 95

# 3. Neuron Specialization Training
bash neuron_specialization/scripts/train.sh --config scripts/mT-big/configs/multilingual.yml