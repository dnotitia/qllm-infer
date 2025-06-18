#!/bin/bash

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_path=$2

#export CUDA_VISIBLE_DEVICES=$DEVICES
#export HF_HOME=~/.cache/huggingface
#export HF_DATASETS_TRUST_REMOTE_CODE=1

cache_dir='./cache'
#tasks=piqa,winogrande,arc_easy,arc_challenge,hellaswag
#tasks=mmlu
tasks=none
num_fewshot=none
limit=none
eval_ppl=true
eval_ppl_seqlen=2048
use_cuda_graph=true
seed=0
#---------------------------------------------------------------------#
# Quantization
a_per_tensor=true
a_per_token=false
bits_a=16
sym_a=true
groupsize_a=-1
a_qrazor=false
a_qrazor_bits=4
a_qrazor_group=32
#---------------------------------------------------------------------#
w_per_channel=true
bits_w=4
sym_w=false
groupsize_w=-1
#groupsize_w=128
w_qrazor=false
w_qrazor_bits=4
w_qrazor_group=32
#---------------------------------------------------------------------#
q_per_tensor=true
q_per_token=false
bits_q=16
sym_q=true
groupsize_q=-1
q_qrazor=false
q_qrazor_bits=4
q_qrazor_group=128
#---------------------------------------------------------------------#
k_pre_RoPE_quant=false
k_per_tensor=true
k_per_token=false  #true = per-channel quant, false = per-cheannel quant
bits_k=16
sym_k=true
groupsize_k=-1
k_qrazor=false
k_qrazor_bits=4
k_qrazor_group=16
#---------------------------------------------------------------------#
v_per_tensor=true
v_per_token=false
bits_v=16
sym_v=true
groupsize_v=-1
v_qrazor=false
v_qrazor_bits=4
v_qrazor_group=16
#---------------------------------------------------------------------#
s_per_tensor=true
s_per_token=false
bits_s=16
sym_s=false
groupsize_s=-1
s_qrazor=false
s_qrazor_bits=4
s_qrazor_group=32
# SmoothQuant
smoothquant=false
smoothquant_alpha=0.85
smoothquant_dataset=pile
#smoothquant_dataset=wikitext2
smoothquant_nsamples=512
smoothquant_seqlen=1024
# GPTQ
gptq=true
#gptq_dataset=c4
gptq_dataset=wikitext2
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
stats_csv_path='cache/llama3-8b_W8A8.csv'
get_layerwise_distance=false

#--w_per_channel  false == per-tensor quantization

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
    --a_qrazor_bits $a_qrazor_bits \
    --a_qrazor_group $a_qrazor_group \
    --a_qrazor $a_qrazor \
    --w_per_channel $w_per_channel \
    --bits_w $bits_w \
    --sym_w $sym_w \
    --groupsize_w $groupsize_w \
    --w_qrazor_bits $w_qrazor_bits \
    --w_qrazor_group $w_qrazor_group \
    --w_qrazor $w_qrazor \
    --q_per_tensor $q_per_tensor \
    --q_per_token $q_per_token \
    --bits_q $bits_q \
    --sym_q $sym_q \
    --groupsize_q $groupsize_q \
    --q_qrazor_bits $q_qrazor_bits \
    --q_qrazor_group $q_qrazor_group \
    --q_qrazor $q_qrazor \
    --k_pre_RoPE_quant $k_pre_RoPE_quant \
    --k_per_tensor $k_per_tensor \
    --k_per_token $k_per_token \
    --bits_k $bits_k \
    --sym_k $sym_k \
    --groupsize_k $groupsize_k \
    --k_qrazor_bits $k_qrazor_bits \
    --k_qrazor_group $k_qrazor_group \
    --k_qrazor $k_qrazor \
    --v_per_tensor $v_per_tensor \
    --v_per_token $v_per_token \
    --bits_v $bits_v \
    --sym_v $sym_v \
    --groupsize_v $groupsize_v \
    --v_qrazor_bits $v_qrazor_bits \
    --v_qrazor_group $v_qrazor_group \
    --v_qrazor $v_qrazor \
    --s_per_tensor $s_per_tensor \
    --s_per_token $s_per_token \
    --bits_s $bits_s \
    --sym_s $sym_s \
    --groupsize_s $groupsize_s \
    --s_qrazor_bits $s_qrazor_bits \
    --s_qrazor_group $s_qrazor_group \
    --s_qrazor $s_qrazor \
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
    --get_layerwise_distance $get_layerwise_distance \
