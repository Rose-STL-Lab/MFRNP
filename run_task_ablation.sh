# MFRNP-H
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config pde_config_ablation.yaml --levels 3 --device cuda
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config pde_config_ablation.yaml --levels 5 --device cuda

python train.py --data_path "data/OOD/poisson/l3" --save_dir poisson_OOD --config pde_config_ablation.yaml --levels 3 --device cuda
python train.py --data_path "data/OOD/poisson/l5" --save_dir poisson_OOD --config pde_config_ablation.yaml --levels 5 --device cuda

python train.py --data_path "data/full_dataset/heat" --save_dir heat --config pde_config_ablation.yaml --levels 3 --device cuda
python train.py --data_path "data/full_dataset/heat" --save_dir heat --config pde_config_ablation.yaml --levels 5 --device cuda

python train.py --data_path "data/OOD/heat/l3" --save_dir heat_OOD --config pde_config_ablation.yaml --levels 3 --device cuda
python train.py --data_path "data/OOD/heat/l5" --save_dir heat_OOD --config pde_config_ablation.yaml --levels 5 --device cuda