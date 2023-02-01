# tmp/Kaist_GBNet_RGB_AdamW/models/weights_11
# tmp/Kaist_GBNet_thermal/models/weights_8
# tmp/Kaist_GBNet_thermal_distill_v2/models/weights_11
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python evaluate_depth_v2.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
                --batch_size 1 --frame_ids 0 --use_stereo --model_name Kaist_GBNet_thermal --models_to_load encoder depth \
                --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model transdssl --thermal \
                --load_weights_folder tmp/Kaist_sup_pwsa_v5/models/weights_30 #--debug #--num_workers 0
