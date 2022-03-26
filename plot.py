import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(file_path):
    # log_format: 
    # dimension: 32, average step time: 0.003071599006652832, max allocated: 0.009787559509277344, max cached: 0.021484375
    dim_list = []
    avg_step_time_list = []
    max_allocated_list = []
    max_cached_list = []
    with open(file_path) as f:
        for line in f:
            if 'average step time' in line:
                parts = line.split(',')
                max_cached = float(parts[-1].split(':')[1])
                max_allocated = float(parts[-2].split(':')[1])
                avg_step_time = float(parts[-3].split(':')[1])
                dim = float(parts[-4].split(':')[-1])
                dim_list.append(dim)
                max_allocated_list.append(max_allocated)
                max_cached_list.append(max_cached)
                avg_step_time_list.append(avg_step_time)
    # return dim_list, avg_step_time_list, max_allocated_list, max_cached_list
    return dim_list[:8], avg_step_time_list[:8], max_allocated_list[:8], max_cached_list[:8]


def plot(title, x_list, y_list, label_list, x_label, y_label, output_path):
    plt.cla()
    for x, y, label in zip(x_list, y_list, label_list):
        plt.plot(x, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    return parser.parse_args()

def get_info_from_file_name(file_name):
    parts = file_name.stem.split('_')
    mode = parts[-1].lstrip('mode').upper()
    world_size = int(parts[-2].lstrip('ws'))
    tp_size = int(parts[-3].lstrip('tp'))
    return world_size, tp_size, mode

def main():
    args = parse_args()
    log_file_path_list = Path(args.log_path).glob('*.log')
    output_path = Path(args.output_path)

    dim_all = []
    avg_time_all = []
    max_alloc_all = []
    max_cached_all = []
    label_all = []

    for file_path in log_file_path_list:
        world_size, tp_size, mode = get_info_from_file_name(file_path)
        dim_list, avg_step_time_list, max_allocated_list, max_cached_list = parse_log_file(file_path)
        dim_all.append(dim_list)
        avg_time_all.append(avg_step_time_list)
        max_alloc_all.append(max_allocated_list)
        max_cached_all.append(max_cached_list)
        label_all.append(f'{mode} (WS={world_size}, TP={tp_size})')

    plot(title='Average Step Time', x_list=dim_all, y_list=avg_time_all, label_list=label_all, x_label='dimension', y_label='time/s', output_path=output_path.joinpath('avg_step_time.jpg'))
    plot(title='Max Allocated Memory', x_list=dim_all, y_list=max_alloc_all, label_list=label_all, x_label='dimension', y_label='Memory/GB', output_path=output_path.joinpath('max_alloc.jpg'))
    plot(title='Max Cached Memory', x_list=dim_all, y_list=max_cached_all, label_list=label_all, x_label='dimension', y_label='Memory/GB', output_path=output_path.joinpath('max_cached.jpg'))


if __name__ == '__main__':
    main()