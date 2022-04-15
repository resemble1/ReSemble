# ReSemble: Reinforced Ensemble Framework for Data Prefetching
# Update for SC
This repo contains code accompaning the manuscript, "ReSemble: Reinforced Ensemble Framework for Data Prefetching"
## Dependencies
* python: 3.x
* Pytorch: 0.4+

## Run Demo
### MLP-based ReSemble
1. `cd ./ReSemble_MLP`
2. `./run_demo.sh`
3. Model and output prefetching file are generated at dir `./results`

### Tabular Varient
1. `cd ./ReSemble_Tab`
2. `./run_demo.sh`
3. Model and output prefetching file are generated at dir `./results`

## Customed Confuguration
1. Hyperperameters are set in `./ReSemble/config.py`
2. Input prefetchers number and type can be customed.
  * write your own prefetcher in ChampSim
  * generate the prefetching traces of multiple prefetchers using ChampSim
  * the prefetcher is classified as spatial and temporal, each is formed as a list of paths in the command as below, paths divided by ";":
  ```python ./ensemble_rl.py warm_up_instructions total_sim_instructions path_cache "path_spatial_prefetcher1;path_spatial_prefetcher2;..." "path_temporal_prefetcher1;path_temporal_prefetcher2;..." $model_save_path $prefetch_file_output_path```
 
 ## Dataset and Simulator
 We use the dataset and simulator provided for ML Prefetching Competition, which is based on a modified ChampSim:
 * https://github.com/Quangmire/ChampSim

[![DOI](https://zenodo.org/badge/394989763.svg)](https://zenodo.org/badge/latestdoi/394989763)
