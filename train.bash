CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=25906 train.py \
--config ./configs/retrieval_icfg.yaml \
--output_dir output/ICFG \
--max_epoch 30 \
--batch_size_train 24 \
--batch_size_test 64 \
--init_lr 1e-5  \
--k_test 32 \
--epoch_eval 1 \
#--load_head \

