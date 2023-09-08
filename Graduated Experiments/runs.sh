#!/bin/bash
DEVICE=$1
#declare -a Methods=("badam" "tfcl" "bgd" "naive" "ewconline" "mas" "synaptic_intelligence" "vcl")
declare -a Methods=("tfcl" "vcl")
declare -a Exps=("CIsplitmnist" "CIsplitfmnist" "pmnist")

for method in "${Methods[@]}"; do
    for exp in "${Exps[@]}"; do
        CUDA_VISIBLE_DEVICES=$DEVICE python main.py -a $method -x $exp 
        sleep 10
    done
done