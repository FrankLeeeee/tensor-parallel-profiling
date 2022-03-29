from argparse import Action
from email.policy import default
import colossalai
from colossalai.nn.layer.utils import CheckpointModule
from numpy import require
import torch
import time
from functools import partial


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('-t', '--tp-size', type=int, required=True)
    parser.add_argument('-d', '--depth', type=int)
    parser.add_argument('-c', '--checkpoint', action='store_true')
    parser.add_argument('-m', '--mode', type=str, required=True)
    parser.add_argument('-s', '--start', type=int, default=32)
    parser.add_argument('-e', '--end', type=int, default=2**16)
    parser.add_argument('-g', '--growth', type=int, default=2)
    parser.add_argument('-l', '--layer', type=str, choices=['linear', 'layernorm'], required=True)
    parser.add_argument('-b', '--by', type=str, choices=['bs', 'dim'], required=True)
    args = parser.parse_args()
    return args

def launch(args):
    config = dict(
        parallel=dict(
            tensor=dict(size=args.tp_size, mode=args.mode)
        )
    )

    if args.mode == '2.5d':
        config['parallel']['tensor']['depth'] = args.tp_size // 4

    colossalai.launch_from_torch(config=config)

class LinearModel(CheckpointModule):

    def __init__(self, dim, mlp_ratio, checkpoint=False):
        super().__init__(checkpoint)
        self.dense_1 = colossalai.nn.Linear(dim, dim*mlp_ratio)
        self.dense_2 = colossalai.nn.Linear(dim*mlp_ratio, dim)

    def forward(self, x):
        return self.dense_2(self.dense_1(x))

class LayerNormModel(CheckpointModule):
    def __init__(self, dim, checkpoint=False):
        super().__init__(checkpoint)
        self.norm_1 = colossalai.nn.LayerNorm(dim)
        self.norm_2 = colossalai.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm_2(self.norm_1(x))


def build_model(model_type, dim, checkpoint, mlp_ratio=None):
    if model_type == 'linear':
        return LinearModel(dim, mlp_ratio, checkpoint)
    elif model_type == 'layernorm':
        return LayerNormModel(dim, checkpoint=checkpoint)

def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()

def get_memory_states():
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_cached = torch.cuda.max_memory_cached() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
    return max_allocated, max_cached

def profile_model(model, warmup_steps, profile_steps, data_func):
    def _run_step(data):
        out = model(data)
        out.mean().backward()

    data_list = [data_func().cuda() for _ in range(warmup_steps)]
    for data in data_list:
        _run_step(data)

    data_list = [data_func().cuda() for _ in range(profile_steps)]
    start = get_time_stamp()
    for data in data_list:
        _run_step(data)
    end = get_time_stamp()
    return (end - start) / profile_steps
        
def get_batch_data(dim, batch_size, seq_length, mode):
    if mode in ['2d', '2.5d']:
        batch_size = batch_size // 2
        dim = dim // 2
    elif mode == '3d':
        batch_size = batch_size // 4
        dim = dim // 2

    data = torch.rand(batch_size, seq_length, dim).cuda()
    return data

def main():
    args = parse_args()
    launch(args)
    logger = colossalai.logging.get_dist_logger()
    
    if torch.distributed.get_rank() == 0:
        world_size = torch.distributed.get_world_size()
        is_with_checkpoint = 'with_checkpoint' if args.checkpoint else 'without_checkpoint'
        logger.log_to_file(f'./{args.layer}_{is_with_checkpoint}_logs', suffix=f'tp{args.tp_size}_ws{world_size}_mode{args.mode}')
    logger.info(f'config:\n{args.__dict__}\n', ranks=[0])

    if args.by == 'dim':
        start_dim = args.start
        end_dim = args.end
        growth_factor = args.growth

        while start_dim < end_dim:
            model = build_model(model_type=args.layer, dim=start_dim, mlp_ratio=4, checkpoint=args.checkpoint)
            model = model.cuda()
            data_func = partial(get_batch_data, dim=start_dim, batch_size=32, seq_length=512, mode=args.mode)
            avg_step_time = profile_model(model=model, warmup_steps=10, profile_steps=50, data_func=data_func)
            max_allocated, max_cached = get_memory_states()
            logger.info(f'dimension: {start_dim}, average step time: {avg_step_time}, max allocated: {max_allocated}, max cached: {max_cached}', ranks=[0])
            torch.cuda.empty_cache()
            start_dim *= growth_factor
    elif args.by == 'bs':
        dim = 1024
        start_bs = args.start
        end_bs = args.end
        growth_factor = args.growth

        while start_bs < end_bs:
            model = build_model(model_type=args.layer, dim=dim, mlp_ratio=4, checkpoint=args.checkpoint)
            model = model.cuda()
            data_func = partial(get_batch_data, dim=dim, batch_size=start_bs, seq_length=512, mode=args.mode)
            avg_step_time = profile_model(model=model, warmup_steps=10, profile_steps=50, data_func=data_func)
            max_allocated, max_cached = get_memory_states()
            logger.info(f'batch size: {start_bs}, average step time: {avg_step_time}, max allocated: {max_allocated}, max cached: {max_cached}', ranks=[0])
            torch.cuda.empty_cache()
            start_bs *= growth_factor

if __name__ == '__main__':
    main()
    

    






