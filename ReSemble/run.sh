#!/bin/bash
WARM=20
TOTAL=22

DATA="../data"
RESULT="./results"
app1="654.roms-s0"
echo $app1
path_cache=$DATA/${app1}.txt.xz

path_bo=$DATA/${app1}.bo_file.txt 
path_spp=$DATA/${app1}.spp_file.txt 
path_isb=$DATA/${app1}.sisb_file.txt 
path_domino=$DATA/${app1}.domino_file.txt 
model_save_path=$RESULT/${app1}.pth
path_to_prefetch_file=$RESULT/${app1}.rl.csv

python ./ensemble_rl.py $WARM $TOTAL $path_cache "$path_bo;$path_spp" "$path_isb;$path_domino" $model_save_path $path_to_prefetch_file &&\
echo "done for app "$app1



