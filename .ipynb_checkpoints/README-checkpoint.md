# ReSemble: Reinforced Ensemble Framework for Data Prefetching
# Update for SC
This repo contains code accompanying the manuscript, "ReSemble: Reinforced Ensemble Framework for Data Prefetching"

## Dependencies
* python: 3.x
* Pytorch: 0.4+
* All dependencies see `environment.yml`

## Sample data
The prefetching suggestions need to be generated using ChampSim seperately. Here we provide a generated sample data for running the demo, which includes:
* Cache missing trace for application `654.roms` in SPEC 2017.
* Prefetching suggestions under four different prefetchers: BO, SPP, Domino, and SISB.

## Run Demos
We implemented MLP-based ReSemble model and a tabular variant. Scripts for demos are provided for simple functional tests.
### MLP-based ReSemble
1. `cd ./ReSemble_MLP`
2. `./run_demo.sh`
3. Model and output prefetching file are generated at dir `results/`

### Tabular Varient
1. `cd ./ReSemble_Tab`
2. `./run_demo.sh`
3. Model and output prefetching file are generated at dir `results/`

## Customed Confuguration
1. Hyperperameters are set in `./ReSemble_MLP/config.py` and `./ReSemble_Tab/config.py`
2. Input prefetchers number and type can be customed.
  * write your own prefetcher in ChampSim
  * generate the prefetching traces of multiple prefetchers using ChampSim
  * generate ReSemble prefetching results using the command as below:
  
  ```python ./ensemble_rl.py warm_up_instructions total_sim_instructions path_cache path_spatial_prefetcher1 path_spatial_prefetcher2 path_temporal_prefetcher1 path_temporal_prefetcher2 model_save_path prefetch_file_output_path rewards_output_path```
 
 ## Dataset and Simulator
 We use the dataset and simulator provided for ML Prefetching Competition, which is based on a modified ChampSim:
 * https://github.com/Quangmire/ChampSim
 
 ## DOI
 
 [![DOI](https://zenodo.org/badge/394989763.svg)](https://zenodo.org/badge/latestdoi/394989763)
