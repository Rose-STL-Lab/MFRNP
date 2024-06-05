# Multi-Fidelity Residual Neural Processes for Scalable Surrogate Modeling
Code for paper "Multi-Fidelity Residual Neural Processes for Scalable Surrogate Modeling", to be presented at ICML2024.

## Environment Setup
Create conda environment and install the packages:
```
conda env create -f environment.yml
```

## Download Data
link for dataset

## Running Experiments
Run Full and OOD tasks (Fluid, Heat2,3,5 and Poisson2,3,5):
```
./run_task.sh
```
Run ablation study with MFRNP-H:
```
./run_task_ablation.sh
```
Results are saved at "result" directory.

## Running Single Task
```
python train.py --data_path <path_to_dataset> --save_dir <name_of_directory_to_be_saved_in_result_folder> --config <path_to_config_file> --levels <#_of_total_fidelity_levels> --device <cuda_or_cpu>
```
### Example for Running Poisson Task with 2 Fidelities on GPU
```
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config pde_config.yaml --levels 2 --device cuda
```

## Cite Us

```
@inproceedings{niu2024multi,
  author       = {Niu, Ruijia and Wu, Dongxia and Kim, Kai and Ma, Yi-An and Watson-Parris, Duncan and Yu, Rose},
  title        = {Multi-Fidelity Residual Neural Processes for Scalable Surrogate Modeling},
  booktitle    = {International Conference on Machine Learning, {ICML} 2024},
  series       = {Proceedings of Machine Learning Research},
  year         = {2024}
}
```
