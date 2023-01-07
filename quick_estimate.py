import numpy as np
import argparse
from cost_utils import end_to_end_time
import random


gpu_device_map = {
    "A100-80G": {"memory_bandwidth": 1555*1073741824, "flops": 312*10**12, "memory_limit": 80*1073741824},
    "A100-40G": {"memory_bandwidth": 1555*1073741824, "flops": 312*10**12, "memory_limit": 40*1073741824},
    "A40": {"memory_bandwidth": 696*1073741824, "flops": 150*10**12, "memory_limit": 48*1073741824},
    "A4000": {"memory_bandwidth": 448*1073741824, "flops": 153*10**12, "memory_limit": 16*1073741824},
}

# delay: s, bandwidth: GB/s
network_setting_map = {
    "NVLink": {"delay": 0.0, "bandwidth": 600*1073741824},
    "PCIe": {"delay": 0.0, "bandwidth": 64*1073741824},
    "Data Center": {"delay": 0.0001, "bandwidth": 10/8*1073741824},
    "Cross Region": {"delay": 0.05, "bandwidth": 2/8*1073741824},
    "Cross Region Slow": {"delay": 0.05, "bandwidth": 1/8*1073741824},
    "Miner box": {"delay": 0.0, "bandwidth": 4/8*1073741824},
    "Petals paper-1-1": {"delay": 0.1, "bandwidth": 0.1/8*1073741824},
    "FluidStack same region": {"delay": 0.0005, "bandwidth": 9.6/8*1073741824},
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
    delay_matrix = np.full((16, 16), network_setting_map['PCIe']['delay'])
    bandwidth_matrix = np.full((16, 16), network_setting_map['PCIe']['bandwidth'])
    for i in range(4):
        delay_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['delay']
        bandwidth_matrix[i*4:(i+1)*4, i*4:(i+1)*4] = network_setting_map['PCIe']['bandwidth']
    stage_partitions = [6 for _ in range(16)]
    stage_device_sets = [[i] for i in range(16)]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_petals_paper_setting_1_1():
    device_set = [gpu_device_map['A100-80G'] for _ in range(3)]
    delay_matrix = np.full((3, 3), network_setting_map['Petals paper-1-1']['delay'])
    bandwidth_matrix = np.full((3, 3), network_setting_map['Petals paper-1-1']['bandwidth'])
    stage_partitions = [32 for _ in range(3)]
    stage_device_sets = [[i] for i in range(3)]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets, "btype": 1}


def get_miner_machine_setting1():
    device_set = [gpu_device_map['A4000'] for _ in range(48)]
    delay_matrix = np.full((48, 48), network_setting_map['Cross Region']['delay'])
    bandwidth_matrix = np.full((48, 48), network_setting_map['Cross Region']['bandwidth'])
    for i in range(4):
        delay_matrix[i * 12:(i + 1) * 12, i * 12:(i + 1) * 12] = network_setting_map['Miner box']['delay']
        bandwidth_matrix[i * 12:(i + 1) * 12, i * 12:(i + 1) * 12] = network_setting_map['Miner box']['bandwidth']
    stage_partitions = [24, 24, 24, 24]
    stage_device_sets = [[i for i in range(12)], [i for i in range(12, 24)],
                         [i for i in range(24, 36)], [i for i in range(36, 48)]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_fully_decentralized_miner_machine_setting1():
    device_set = [gpu_device_map['A4000'] for _ in range(48)]
    delay_matrix = np.full((48, 48), network_setting_map['Cross Region Slow']['delay'])
    bandwidth_matrix = np.full((48, 48), network_setting_map['Cross Region Slow']['bandwidth'])

    stage_partitions = [24, 24, 24, 24]
    stage_device_sets = [[i for i in range(12)], [i for i in range(12, 24)],
                         [i for i in range(24, 36)], [i for i in range(36, 48)]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets}


def get_fluidstack_2A40_opt13b_tp2pp1_setting():
    device_set = [gpu_device_map['A4000'] for _ in range(2)]
    delay_matrix = np.full((2, 2), network_setting_map['FluidStack same region']['delay'])
    bandwidth_matrix = np.full((2, 2), network_setting_map['FluidStack same region']['bandwidth'])
    stage_partitions = [40]
    stage_device_sets = [[i for i in range(2)]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets, "h_dim": 5120}


def get_fluidstack_2A40_opt13b_tp1pp2_setting():
    device_set = [gpu_device_map['A4000'] for _ in range(2)]
    delay_matrix = np.full((2, 2), network_setting_map['FluidStack same region']['delay'])
    bandwidth_matrix = np.full((2, 2), network_setting_map['FluidStack same region']['bandwidth'])
    stage_partitions = [20, 20]
    stage_device_sets = [[0], [1]]
    return {"device_set": device_set, "delay_matrix": delay_matrix, "bandwidth_matrix": bandwidth_matrix,
            "stage_partitions": stage_partitions, "stage_device_sets": stage_device_sets, "h_dim": 5120}


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--setting', type=str, default='fluidstack 2A40 opt13b tp1pp2', metavar='S',
                        help='Which setting to estimate')
    parser.add_argument('--benchmark-group', action='store_true',
                        help='whether or not to measure a group of settings.')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
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
    elif args.setting == "miner machine1":
        runtime_setting = get_miner_machine_setting1()
    elif args.setting == "fully decentralized miner machine1":
        runtime_setting = get_fully_decentralized_miner_machine_setting1()
    elif args.setting == "petals paper setting1-1":
        runtime_setting = get_petals_paper_setting_1_1()
    elif args.setting == "fluidstack 2A40 opt13b tp2pp1":
        runtime_setting = get_fluidstack_2A40_opt13b_tp2pp1_setting()
    elif args.setting == "fluidstack 2A40 opt13b tp1pp2":
        runtime_setting = get_fluidstack_2A40_opt13b_tp1pp2_setting()
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
    if "btype" in runtime_setting:
        btype = runtime_setting['btype']
    else:
        btype = 2  # default is fp16
    if "h_dim" in runtime_setting:
        h_dim = runtime_setting['h_dim']
    else:
        h_dim = 12288

    if args.benchmark_group:
        print(f"Setting: <{args.setting}>")
        batch_sizes = [1, 16]
        seq_in_set = [512]
        seq_out_set = [1, 32, 64, 128]
        for batch_size in batch_sizes:
            for seq_in in seq_in_set:
                for seq_out in seq_out_set:
                    print("---------------------------------")
                    print(f"batch size: {batch_size}, input seq: {seq_in}, output seq: {seq_out}.")
                    end_to_end_time(batch_size, seq_in, seq_out, stage_device_sets, stage_partitions, device_info,
                                    delay_matrix, bandwidth_matrix, h_dim=h_dim, b_type=btype)
    else:
        print(f"Setting: <{args.setting}> batch size: {batch_size}, input seq: {seq_in}, output seq: {seq_out}.")
        end_to_end_time(batch_size, seq_in, seq_out, stage_device_sets, stage_partitions, device_info,
                        delay_matrix, bandwidth_matrix, h_dim=h_dim, b_type=btype)


if __name__ == '__main__':
    main()
