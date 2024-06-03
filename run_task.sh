# Full Experiments
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config pde_config.yaml --levels 2 --device cuda
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config pde_config.yaml --levels 3 --device cuda
python train.py --data_path "data/full_dataset/poisson" --save_dir poisson --config Poisson5_config.yaml --levels 5 --device cuda

python train.py --data_path "data/full_dataset/heat" --save_dir heat --config pde_config.yaml --levels 2 --device cuda
python train.py --data_path "data/full_dataset/heat" --save_dir heat --config pde_config.yaml --levels 3 --device cuda
python train.py --data_path "data/full_dataset/heat" --save_dir heat --config pde_config.yaml --levels 5 --device cuda

python train.py --data_path "data/full_dataset/fluid" --save_dir fluid --config Fluid_config.yaml --levels 2 --device cuda

# OOD Experiments
python train.py --data_path "data/OOD/poisson/l2" --save_dir poisson_OOD --config pde_config.yaml --levels 2 --device cuda
python train.py --data_path "data/OOD/poisson/l3" --save_dir poisson_OOD --config pde_config.yaml --levels 3 --device cuda
python train.py --data_path "data/OOD/poisson/l5" --save_dir poisson_OOD --config pde_config.yaml --levels 5 --device cuda

python train.py --data_path "data/OOD/heat/l2" --save_dir heat_OOD --config pde_config.yaml --levels 2 --device cuda
python train.py --data_path "data/OOD/heat/l3" --save_dir heat_OOD --config pde_config.yaml --levels 3 --device cuda
python train.py --data_path "data/OOD/heat/l5" --save_dir heat_OOD --config pde_config.yaml --levels 5 --device cuda