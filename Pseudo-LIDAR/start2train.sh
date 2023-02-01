OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 150 --max_depth 50 \
                --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_sup_transdssl --scales 0 --learning_rate 1e-4 \
                --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model transdssl --thermal --train_mode sup --vggloss --debug\
