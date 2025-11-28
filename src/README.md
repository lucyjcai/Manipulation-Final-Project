run this command to train:

python imitate_episodes.py --ckpt_dir /home/kelly_lucy/Desktop/manipulation_project/Manipulation-Final-Project/src/checkpoints --policy_class ACT --task_name sim_open_drawer --batch_size 8 --seed 0 --num_epochs 200 --lr 1e-5 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200

run this command to evaluate:

python imitate_episodes.py --ckpt_dir /home/kelly_lucy/Desktop/manipulation_project/Manipulation-Final-Project/src/checkpoints --policy_class ACT --task_name sim_open_drawer --batch_size 8 --seed 0 --num_epochs 200 --lr 1e-5 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --eval --onscreen_render --temporal_agg