#!/bin/bash

# A demo recipe for Singing Voice Synthesis (SVS) task in UniAudio
# SVS: text + instruction --> singing audio
. ./path.sh

# Install necessary packages
pip3 install fairseq==0.12.2 einops==0.6.0 sentencepiece encodec

# Define general configurations
stage=1
stop_stage=100
ngpu=1  # number of GPUs to use for training

train_set="train"
valid_set="val"
test_sets="singing_test"

# Training config
seed=999
debug=false
batch_scale=8000
learning_rate=0.005
port=12345
train_opts=
inference_opts=
tag="svs_experiment"
inference_tag="svs_inference"
resume=""
data_tag=""
TASK='singing_voice'

# Link necessary directories if they don't exist
if [ ! -d "utils" ]; then
  ln -s ../tools/kaldi/utils ./
fi
if [ ! -d "data_scripts" ]; then
  ln -s ../tools/data_scripts ./
fi

. utils/parse_options.sh

if [ ! -z $resume ]; then
    train_opts="--resume $resume"
    inference_opts="--resume $resume"
fi

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
else
    export HOST_GPU_NUM=8
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
fi

### Stage 1-2: Data Preparation and Splitting ###

# Prepare dataset
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Preparing SVS dataset"
    # Add commands to prepare `wav.scp`, `text.scp`, and `instruction.scp` if needed.
fi

# Split data for GPUs
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Splitting data for $ngpu GPUs"
    for part in $test_sets $valid_set $train_set; do
      mkdir -p data/${part}/${ngpu}splits
      # Shuffle and split data for parallel processing
      cat data/${part}/wav.scp | shuf > data/${part}/wav.scp.shuf
      split_scp=
      for n in `seq 1 $ngpu`; do
          split_scp="$split_scp data/${part}/${ngpu}splits/wav.${n}.scp"
      done
      utils/split_scp.pl data/${part}/wav.scp.shuf $split_scp
    done
fi

### Stage 3: Process Audio, Text, and Instruction Sequences ###

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preparing audio, text, and instruction sequences for SVS"
    for part in $valid_set $train_set; do
        echo "Preparing $part ..."

        # Process text sequences (lyrics or prompts)
        utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/text_codec_dump.JOB.log \
            python3 data_scripts/offline_tokenization.py \
              --input-file data/${part}/${ngpu}splits/text.JOB.scp \
              --output-file data/${part}/${ngpu}splits/text_codec.JOB.pt \
              --tokenizer text --rank JOB || exit 1;

        # Process instruction sequences
        utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/instruction_codec_dump.JOB.log \
            python3 data_scripts/offline_tokenization.py \
              --input-file data/${part}/${ngpu}splits/instruction.JOB.scp \
              --output-file data/${part}/${ngpu}splits/instruction_codec.JOB.pt \
              --tokenizer text_t5 --rank JOB || exit 1;

        # Process audio sequences (target singing audio)
        utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/audio_codec_dump.JOB.log \
            python3 data_scripts/offline_tokenization.py \
              --input-file data/${part}/${ngpu}splits/wav.JOB.scp \
              --output-file data/${part}/${ngpu}splits/audio_codec.JOB.pt \
              --tokenizer audio --rank JOB || exit 1;
    done
fi

### Stage 4: Create JSON Files for SVS Task ###

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Creating data JSON files for SVS"
    for part in $valid_set $train_set; do
        for n in `seq 0 $[$ngpu-1]`; do
            python3 data_scripts/create_data_json.py \
             --task singing_voice \
             --out-json   $PWD/data/${part}/${ngpu}splits/data_svs.${n}.json \
             --text_seq  $PWD/data/${part}/${ngpu}splits/text_codec.$[$n+1].pt \
             --instruction_seq  $PWD/data/${part}/${ngpu}splits/instruction_codec.$[$n+1].pt \
             --audio_seq  $PWD/data/${part}/${ngpu}splits/audio_codec.$[$n+1].pt \
             &
        done
        wait
    done
fi

### Stage 5: Training ###

train_data_jsons="data/${train_set}/${ngpu}splits/data_svs.ALL.json"
valid_data_jsons="data/${valid_set}/${ngpu}splits/data_svs.ALL.json"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    mkdir -p exp 
    if [ -z $tag ]; then
        echo "Please provide a tag for this experiment" && exit 1;
    fi
    echo "Starting training..."
    NCCL_DEBUG=TRACE torchrun \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ../../train.py \
        --exp_dir exp \
        --seed $seed \
        --cudnn_deterministic \
        --train_data_jsons $train_data_jsons \
        --valid_data_jsons $valid_data_jsons \
        --batch_scale $batch_scale \
        --learning_rate $learning_rate \
        --non-acoustic-repeat 3 \
        --audio-tokenizer "soundstream" \
        --text-tokenizer "text_t5" \
        --n_layer 24 \
        --n_head 16 \
        --n_embd 1536 \
        $train_opts
fi

### Stage 6: Inference ###

inference_dir=exp/${tag}/inference_${inference_tag}
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Starting inference for SVS..."
    mkdir -p ${inference_dir}
    for part in $test_sets; do
        mkdir -p ${inference_dir}/${part}
        echo "Inference on set: ${part}"
        data_json="data/${part}/data_svs.json"  # Update with the path to your SVS JSON file

        utils/run.pl --max-jobs-run 8 JOB=0:$[${ngpu}-1] \
          ${inference_dir}/${part}/inference.JOB.log \
          python3 ../../infer.py \
            --exp_dir exp/${tag} \
            --rank JOB \
            --inference_mode 'sampling' \
            --n_samples 1 \
            --seed 888 \
            --rank JOB \
            --data_json $data_json \
            --generate_target audio \
            --fixed_length False \
            --maxlen_ratio 7 \
            --minlen_ratio 0.5 \
            --output_dir ${inference_dir}/${part}/JOB \
            $inference_opts
    done
fi