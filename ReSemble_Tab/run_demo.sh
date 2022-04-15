#!/bin/bash
WARM=4
TOTAL=8

DATA="../sample_data"
RESULT="../results"
app1="654.roms-s0"
Python_file="./ensemble_tab.py"
version="tab-4"
echo $app1

mkdir $RESULT

app_list=(654.roms-s0)

for app1 in ${app_list[*]}; do
    path_cache=$DATA/${app1}.txt.xz
    path_bo=$DATA/${app1}.bo_file.txt 
    path_spp=$DATA/${app1}.spp_file.txt 
    path_isb=$DATA/${app1}.sisb_file.txt 
    path_domino=$DATA/${app1}.domino_file.txt 
    model_save_path=$RESULT/${app1}.$version.pkl
    path_to_prefetch_file=$RESULT/${app1}.$version.pref.txt
    path_to_stats=$RESULT/${app1}.$version.rewards.csv

	python $Python_file $WARM $TOTAL $path_cache $path_bo $path_spp $path_isb $path_domino $model_save_path $path_to_prefetch_file $path_to_stats &&\
    
    echo "Done for app "$app1
done
