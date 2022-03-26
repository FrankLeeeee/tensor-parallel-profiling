import colossalai
import torch

def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--interval', type=int, required=True)
    args = parser.parse_args()
    return args

def launch(args):
    config = dict(
        parallel=dict(
            tensor=args.size,
            mode=args.mode
        )
    )

    if args.mode == '2.5d':
        config['parallel']['depth'] = args.size // 4

    colossalai.launch_from_torch(config=config)

class Net(torch.nn.Module):

    def __init__(self, dim, mlp_ratio):
        self.dense_1 = colossalai.nn.Linear(dim, dim*mlp_ratio)
        self.dense_2 = colossalai.nn.Linear(dim*mlp_ratio, dim)

    def forward(self, x):
        return self.dense_2(self.dense_1(x))


def build_model(dim, mlp_ratio):
    return Net(dim, mlp_ratio)


def profile_model(dim, mlp_ratio, warmup_steps, profile_steps, batch_size=32, seq_length=512):
    model = build_model(dim, mlp_ratio)

    def _run_step():
        data = torch.rand(batch_size, seq_length, dim).cuda()
        out = model(data)
        out.mean().backward()

    for _ in range(warmup_steps):
        _run_step()


    for _ in range(profile_steps):
        _run_step()
        

def main():
    args = parse_args()

    step = (args.end - args.start) // args.interval
    dim_range = range(args.start, args.end, step)

    for dim in dim_range:
        profile_model(dim, mlp_ratio=4, warmup_steps=10, profile_steps=50)


    
if __name__ == '__main__':
    main()
    

    






