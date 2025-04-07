#!/bin/bash

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_path=$2
cache_dir='./cache'
tasks=none
num_fewshot=none
limit=none
eval_ppl=true
eval_ppl_seqlen=2048
use_cuda_graph=true
seed=0
# Quantization
a_per_tensor=false
a_per_token=false
bits_a=16
sym_a=false
groupsize_a=-1
w_per_channel=true
bits_w=4
sym_w=false
groupsize_w=-1
q_per_tensor=true
q_per_token=false
bits_q=16
sym_q=true
groupsize_q=-1
k_per_tensor=true
k_per_token=false
bits_k=16
sym_k=true
groupsize_k=-1
v_per_tensor=true
v_per_token=false
bits_v=16
sym_v=true
groupsize_v=-1
s_per_tensor=true
s_per_token=false
bits_s=16
sym_s=false
groupsize_s=-1
# SmoothQuant
smoothquant=false
smoothquant_alpha=0.5
smoothquant_dataset=pile
smoothquant_nsamples=512
smoothquant_seqlen=512
# GPTQ
gptq=true
gptq_dataset=c4
gptq_nsamples=128
gptq_seqlen=2048
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=true
gptq_static_groups=false
# Chatbot Simulation
chat=false
# Log
logfile='logs/out.txt'
# Analysis Tools
analyze_stats=false
stats_csv_path='cache/llama3.1-8b-instruct-w8a8sq.csv'
get_layerwise_distance=false

CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
    --model_path $model_path \
    --cache_dir $cache_dir \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --limit $limit \
    --eval_ppl $eval_ppl \
    --eval_ppl_seqlen $eval_ppl_seqlen \
    --use_cuda_graph $use_cuda_graph \
    --seed $seed \
    --a_per_tensor $a_per_tensor \
    --a_per_token $a_per_token \
    --bits_a $bits_a \
    --sym_a $sym_a \
    --groupsize_a $groupsize_a \
    --w_per_channel $w_per_channel \
    --bits_w $bits_w \
    --sym_w $sym_w \
    --groupsize_w $groupsize_w \
    --q_per_tensor $q_per_tensor \
    --q_per_token $q_per_token \
    --bits_q $bits_q \
    --sym_q $sym_q \
    --groupsize_q $groupsize_q \
    --k_per_tensor $k_per_tensor \
    --k_per_token $k_per_token \
    --bits_k $bits_k \
    --sym_k $sym_k \
    --groupsize_k $groupsize_k \
    --v_per_tensor $v_per_tensor \
    --v_per_token $v_per_token \
    --bits_v $bits_v \
    --sym_v $sym_v \
    --groupsize_v $groupsize_v \
    --s_per_tensor $s_per_tensor \
    --s_per_token $s_per_token \
    --bits_s $bits_s \
    --sym_s $sym_s \
    --groupsize_s $groupsize_s \
    --smoothquant $smoothquant \
    --smoothquant_alpha $smoothquant_alpha \
    --smoothquant_dataset $smoothquant_dataset \
    --smoothquant_nsamples $smoothquant_nsamples \
    --smoothquant_seqlen $smoothquant_seqlen \
    --gptq $gptq \
    --gptq_dataset $gptq_dataset \
    --gptq_nsamples $gptq_nsamples \
    --gptq_seqlen $gptq_seqlen \
    --gptq_true_sequential $gptq_true_sequential \
    --gptq_percdamp $gptq_percdamp \
    --gptq_act_order $gptq_act_order \
    --gptq_static_groups $gptq_static_groups \
    --chat $chat \
    --logfile $logfile \
    --analyze_stats $analyze_stats \
    --stats_csv_path $stats_csv_path \
    --get_layerwise_distance $get_layerwise_distance
