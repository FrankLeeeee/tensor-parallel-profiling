
python -m torch.distributed.launch --nproc_per_node $0 --master_addr localhost --master_port 29500  range_test.py --size $1 --mode $2 --start $3 --end $4 --interval $5