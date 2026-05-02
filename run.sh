# #!/bin/bash

# python main_stage1.py --root_dir data/phase_data_simulation --batch_size 16 --lr 1e-4 --save_path checkpoints/phase_data_simulation_50 --num_epochs 400 --phase_size 50 --pat_size 50 --weight_decay 0.01 #  --resume_path checkpoints/inr/stage1/latest.pth 

# python main_stage1.py --root_dir data/phase_data_simulation --batch_size 64 --lr 4e-4 --save_path checkpoints/phase_150_simulation --num_epochs 100 --phase_size 50 --pat_size 150 --weight_decay 0.01  # --resume_path checkpoints/phase_150_iter1/stage1/latest.pth 

# python main_stage2.py --root_dir data/data_6000us_iter1 --batch_size 64 --lr 2e-4 --save_path checkpoints/experiment_150_6000us_stage1 --phase_size 50 --pat_size 150 --num_epochs 100 --weight_decay 0.01 #  --resume_path checkpoints/inr/stage2/latest.pth --num_epochs 600



# python main_stage2.py --root_dir data/data_4000 --batch_size 64 --lr 2e-4 --save_path checkpoints/ex.periment_150_4000us --phase_size 50 --pat_size 150 --num_epochs 100 --weight_decay 0.01 #  --resume_path checkpoints/inr/stage2/latest.pth --num_epochs 600

# wu/AIOptics/data/phase_data_simulation_2pi_twophases
# simulation 2026-03-10
python main_stage1.py --root_dir data/4f_random_perlin_20k --batch_size 32 --lr 4e-4 --save_path checkpoints/4f_random_perlin_20k_d4144_2  --weight_decay 0.01 --num_epochs 1000 --input_size 50 --pat_size 100 --exp_name 4f_random_perlin_20k_d4144 # --resume_path checkpoints/4f_comb_phases_20k_l1_mean_losses/stage1/latest.pth

# 2026-3-10
# python main_stage2_simulation.py --root_dir data/4f_twophases_2pi --batch_size 32 --lr 4e-4 --save_path checkpoints/4f_twophases_2pi_200 --num_epochs 400 --weight_decay 0.01 --phase_size 50 --pat_size 200  #  --resume_path checkpoints/inr/stage2/latest.pth --num_epochs 600

# python main_stage1.py --root_dir data/amp_data --batch_size 64 --lr 4e-4 --save_path checkpoints/amp_data_150 --num_epochs 400 --phase_size 50 --pat_size 150 --weight_decay 0.01  # --resume_path checkpoints/phase_150_iter1/stage1/latest.pth