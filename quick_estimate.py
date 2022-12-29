import numpy as np
import argparse
from cost_utils import end_to_end_time
import random


gpu_device_map = {
    "A100-80G": {"memory_bandwidth": 1555*1073741824, "flops": 312*10**12, "memory_limit": 80*1073741824},
    "A100-40G": {"memory_bandwidth": 1555*1073741824, "flops": 312*10**12, "memory_limit": 40*1073741824},
    "A40": {"memory_bandwidth": 696*1073741824, "flops": 150*10**12, "memory_limit": 48*1073741824},
}

# delay: s, bandwidth: GB/s
network_setting_map = {
    "NVLink": {"delay": 0.0, "bandwidth": 600*1073741824},
    "PCIe": {"delay": 0.0, "bandwidth": 64*1073741824},
    "Data Center": {"delay": 0.0001, "bandwidth": 10/8*1073741824},
    "Cross Region": {"delay": 0.05, "bandwidth": 2/8*1073741824}
}


# 8 A100-80G in a single box
def get_data_center_setting():
    device_set = [gpu_device_map['A100-80G'] for _ in range(8)]
    delay_matrix = np.full((8, 8), network_setting_map['PCIe']['delay'])
    bandwidth_matrix = np.full((8, 8), network_setting_map['PCIe']['bandwidth'])
    stage_partitions = [96]
    stage_device_sets = [[i for i in range(8)]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_scheduled_test1_setting():
    device_set = [gpu_device_map['A100-40G'] for _ in range(16)]
    delay_matrix = np.full((16, 16), network_setting_map['Cross Region']['delay'])
    bandwidth_matrix = np.full((16, 16), network_setting_map['Cross Region']['bandwidth'])
    for i in range(4):
        delay_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['delay']
        bandwidth_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['bandwidth']
    stage_partitions = [24, 24, 24, 24]
    stage_device_sets = [[i for i in range(4)], [i for i in range(4, 8)],
                         [i for i in range(8, 12)], [i for i in range(12, 16)]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_random_test1_setting():
    device_set = [gpu_device_map['A100-40G'] for _ in range(16)]
    delay_matrix = np.full((16, 16), network_setting_map['Cross Region']['delay'])
    bandwidth_matrix = np.full((16, 16), network_setting_map['Cross Region']['bandwidth'])
    for i in range(4):
        delay_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['delay']
        bandwidth_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['bandwidth']
    stage_partitions = [24, 24, 24, 24]
    shuffled_array = [i for i in range(16)]
    random.Random(4).shuffle(shuffled_array)
    stage_device_sets = [shuffled_array[0:4], shuffled_array[4:8],
                         shuffled_array[8: 12], shuffled_array[12: 16]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_petals_test1_setting():
    device_set = [gpu_device_map['A100-40G'] for _ in range(16)]
    delay_matrix = np.full((16, 16), network_setting_map['Cross Region']['delay'])
    bandwidth_matrix = np.full((16, 16), network_setting_map['Cross Region']['bandwidth'])
    for i in range(4):
        delay_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['delay']
        bandwidth_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['bandwidth']
    stage_partitions = [6 for _ in range(16)]
    stage_device_sets = [[i] for i in range(16)]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--setting', type=str, default='random test1', metavar='S',
                        help='Which setting to estimate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='task batch-size')
    parser.add_argument('--input-seq-length', type=int, default=128, metavar='N',
                        help='input sequence length.')
    parser.add_argument('--output-seq-length', type=int, default=32, metavar='N',
                        help='output sequence length.')
    args = parser.parse_args()
    if args.setting == "data center":
        runtime_setting = get_data_center_setting()
    elif args.setting == "scheduled test1":
        runtime_setting = get_scheduled_test1_setting()
    elif args.setting == "petals test1":
        runtime_setting = get_petals_test1_setting()
    elif args.setting == "random test1":
        runtime_setting = get_random_test1_setting()
    else:
        assert False

    batch_size = args.batch_size
    seq_in = args.input_seq_length
    seq_out = args.output_seq_length
    print(runtime_setting)
    device_info = runtime_setting['device_set']
    delay_matrix = runtime_setting['delay_matrix']
    bandwidth_matrix = runtime_setting['bandwidth_matrix']
    stage_partitions = runtime_setting['stage_partitions']
    stage_device_sets = runtime_setting['stage_device_sets']
    print(f"Setting: <{args.setting}> batch size: {batch_size}, input seq: {seq_in}, output seq: {seq_out}.")
    end_to_end_time(batch_size, seq_in, seq_out, stage_device_sets, stage_partitions, device_info,
                    delay_matrix, bandwidth_matrix)


if __name__ == '__main__':
    main()
