import colossalai
import torch
from functools import partial
from utils import build_model, get_memory_states, get_time_stamp
from colossalai.utils import MultiTimer
from colossalai.core import global_context as gpc


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('-s', '--start', type=int, default=32)
    parser.add_argument('-e', '--end', type=int, default=2**14)
    parser.add_argument('-g', '--growth', type=int, default=2)
    parser.add_argument('-l', '--layer', type=str, choices=['linear', 'layernorm'], required=True)
    parser.add_argument('-b', '--by', type=str, choices=['bs', 'dim'], required=True)
    args = parser.parse_args()
    return args


def profile_model(model, warmup_steps, profile_steps, data_func, timer, prof):
    
    def _run_step(data):
        timer.start('forward')
        out = model(data)
        timer.stop('forward', keep_in_history=True)
        timer.start('backward')
        out.mean().backward()
        timer.stop('backward', keep_in_history=True)
    
    data_list = [data_func() for _ in range(warmup_steps)]
    for data in data_list:
        _run_step(data)
    timer.reset('forward')
    timer.reset('backward')

    data_list = [data_func() for _ in range(profile_steps)]
    start = get_time_stamp()
    prof.start()
    for data in data_list:
        _run_step(data)
        prof.step()
    prof.stop()
    end = get_time_stamp()
    avg_step_time = (end - start) / profile_steps

    max_allocated, max_cached = get_memory_states()
    fwd_time = timer.get_timer('forward').get_history_mean()
    bwd_time = timer.get_timer('backward').get_history_mean()
    return avg_step_time, fwd_time, bwd_time, max_allocated, max_cached
        
def get_batch_data(dim, batch_size, seq_length, mode):
    if mode in ['2d', '2.5d']:
        batch_size = batch_size // 2
        dim = dim // 2
    elif mode == '3d':
        batch_size = batch_size // 4
        dim = dim // 2

    data = torch.rand(batch_size, seq_length, dim).cuda()
    return data

def configure_log_file(logger, args):
    if torch.distributed.get_rank() == 0:
        world_size = torch.distributed.get_world_size()
        logger.log_to_file(f'./logs/{args.layer}/by_{args.by}', suffix=f'ws{world_size}_mode{gpc.config.parallel.tensor.mode}', mode='w')


def main():
    args = parse_args()
    colossalai.launch_from_torch(args.config)
    logger = colossalai.logging.get_dist_logger()
    timer = MultiTimer()

    configure_log_file(logger, args)
    logger.info(f'config:\n{gpc.config}\n', ranks=[0])
    world_size = torch.distributed.get_world_size()
    
    if args.by == 'dim':
        start_dim = args.start
        end_dim = args.end
        growth_factor = args.growth

        while start_dim < end_dim:
            model = build_model(model_type=args.layer, dim=start_dim, mlp_ratio=4, checkpoint=True)
            model = model.cuda()
            data_func = partial(get_batch_data, dim=start_dim, batch_size=32, seq_length=512, mode=gpc.config.parallel.tensor.mode)
            prof_log_path = f'./profiling/ws{world_size}_tp{gpc.config.parallel.tensor.mode}_dim{start_dim}'
            prof = torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_log_path))
            avg_step_time, fwd_time, bwd_time, max_alloc, max_cached = profile_model(model=model, warmup_steps=5, profile_steps=5, data_func=data_func, timer=timer, prof=prof)
            logger.info(f'dimension: {start_dim}, average step time: {avg_step_time}, forward: {fwd_time}, backward: {bwd_time} max allocated: {max_alloc}, max cached: {max_cached}', ranks=[0])
            torch.cuda.empty_cache()
            start_dim *= growth_factor
    elif args.by == 'bs':
        dim = 1024
        start_bs = args.start
        end_bs = args.end
        growth_factor = args.growth

        while start_bs <= end_bs:
            model = build_model(model_type=args.layer, dim=dim, mlp_ratio=4, checkpoint=True)
            model = model.cuda()
            data_func = partial(get_batch_data, dim=dim, batch_size=start_bs, seq_length=512, mode=gpc.config.parallel.tensor.mode)
            prof_log_path = f'./profiling/ws{world_size}_tp{gpc.config.parallel.tensor.mode}_bs{start_bs}'
            prof = torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_log_path))
            avg_step_time, fwd_time, bwd_time, max_alloc, max_cached = profile_model(model=model, warmup_steps=5, profile_steps=5, data_func=data_func, timer=timer, prof=prof)
            logger.info(f'batch size: {start_bs}, average step time: {avg_step_time}, forward: {fwd_time}, backward: {bwd_time} max allocated: {max_alloc}, max cached: {max_cached}', ranks=[0])
            torch.cuda.empty_cache()
            start_bs *= growth_factor

if __name__ == '__main__':
    main()
    

    






