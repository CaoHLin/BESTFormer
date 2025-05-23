NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train.py 
