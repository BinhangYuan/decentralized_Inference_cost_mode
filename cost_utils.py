
def compute_prompt_time_stage(seq_in, batch_size, m_d, c_d, num_layers, tp_degree=1, h_dim=12288, b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / m_d
    layer_compute_time = 24 * batch_size * seq_in * h_dim * h_dim / tp_degree / c_d
    return (layer_scan_time + layer_compute_time) * num_layers


def compute_token_step_time_stage(batch_size, m_d, c_d, num_layers, tp_degree=1, h_dim=12288, b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / m_d
    layer_compute_time = 24 * batch_size * h_dim * h_dim / tp_degree / c_d
    return (layer_scan_time + layer_compute_time) * num_layers


def communicate_prompt_time_stage(seq_in, batch_size, num_layers, device_sets,
                                  delay_matrix, bandwidth_matrix,
                                  h_dim=12288, b_type=2) -> float:
    tp_degree = len(device_sets)
    step_time = 0
    for i in device_sets:
        current_step = 0
        for j in device_sets:
            if i != j:
                current_step += (delay_matrix[i,j]+batch_size*seq_in*h_dim*b_type/tp_degree/bandwidth_matrix[i,j])
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers
    return result


def communicate_token_step_time_stage(batch_size, num_layers, device_set,
                                      delay_matrix, bandwidth_matrix,
                                      h_dim=12288, b_type=2) -> float:
    tp_degree = len(device_set)
    step_time = 0
    for i in device_set:
        current_step = 0
        for j in device_set:
            if i != j:
                current_step += (delay_matrix[i,j] + batch_size*h_dim*b_type/tp_degree/bandwidth_matrix[i,j])
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers
    return result


def communication_pipeline_prompt_time_cross_stage(seq_in, batch_size, device_set1, device_set2,
                                                   delay_matrix, bandwidth_matrix,
                                                   h_dim=12288, b_type=2) -> float:
    send_scatter_time = None
    tp_degree2 = len(device_set2)
    chunk_size = batch_size * seq_in * h_dim * b_type / tp_degree2
    for i in device_set1:
        for j in device_set2:
            send_time = delay_matrix[i, j] + batch_size * h_dim * b_type / bandwidth_matrix[i,j]
            scatter_time = 0
            for k in device_set2:
                if j != k:
                    scatter_time += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
            if send_scatter_time is None:
                send_scatter_time = send_time + scatter_time
            else:
                send_scatter_time = max(send_scatter_time, send_time + scatter_time)
    all_gather_time = 0
    for i in device_set2:
        current_step = 0
        for j in device_set2:
            if i != j:
                current_step += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
        all_gather_time = max(all_gather_time, current_step)
    result = send_scatter_time + all_gather_time
    return result


def communication_pipeline_token_step_time_cross_stage(batch_size, device_set1, device_set2,
                                                       delay_matrix, bandwidth_matrix,
                                                       h_dim=12288, b_type=2) -> float:
    send_scatter_time = None
    tp_degree2 = len(device_set2)
    chunk_size = batch_size * h_dim * b_type / tp_degree2
    for i in device_set1:
        for j in device_set2:
            send_time = delay_matrix[i, j] + batch_size * h_dim * b_type / bandwidth_matrix[i, j]
            scatter_time = 0
            for k in device_set2:
                if j != k:
                    scatter_time += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
            if send_scatter_time is None:
                send_scatter_time = send_time + scatter_time
            else:
                send_scatter_time = max(send_scatter_time, send_time + scatter_time)
    all_gather_time = 0
    for i in device_set2:
        current_step = 0
        for j in device_set2:
            if i != j:
                current_step += (delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
        all_gather_time = max(all_gather_time, current_step)
    result = send_scatter_time + all_gather_time
    return result


def communication_pipeline_token_step_time_cross_stage_last(batch_size, device_set1, device_set2,
                                                            delay_matrix, bandwidth_matrix) -> float:
    send_time = None
    chunk_size = batch_size * 4
    for i in device_set1:
        for j in device_set2:
            if send_time is None:
                send_time = delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j]
            else:
                send_time = min(send_time,  delay_matrix[i, j] + chunk_size / bandwidth_matrix[i, j])
    return send_time


def end_to_end_time(batch_size, seq_in, seq_out, stage_device_sets, stage_partitions, device_info,
                    delay_matrix, bandwidth_matrix,
                    h_dim=12288, b_type=2) -> float:
    prompt_compute_time = 0
    prompt_tp_comm_time = 0
    prompt_pp_comm_time = 0
    stage_num = len(stage_partitions)
    for i in range(stage_num):
        device_set = stage_device_sets[i]
        m_d = device_info[device_set[0]]['memory_bandwidth']
        c_d = device_info[device_set[0]]['flops']
        num_layers = stage_partitions[i]
        compute_time = compute_prompt_time_stage(seq_in, batch_size, m_d, c_d, num_layers, len(device_set),
                                                 h_dim, b_type)
        print(f"Prompt phase stage-<{i}> compute time {compute_time}")
        prompt_compute_time += compute_time
        comm_time = communicate_prompt_time_stage(seq_in, batch_size, num_layers, device_set, delay_matrix,
                                                           bandwidth_matrix, h_dim, b_type)
        print(f"Prompt phase stage-<{i}> comm time {comm_time}")
        prompt_tp_comm_time += comm_time
    for i in range(stage_num-1):
        comm_time = communication_pipeline_prompt_time_cross_stage(seq_in, batch_size, stage_device_sets[i],
                                                                   stage_device_sets[i+1], delay_matrix,
                                                                   bandwidth_matrix, h_dim, b_type)
        print(f"Prompt phase pipeline stage-<{i}, {i+1}> comm time {comm_time}")
        prompt_pp_comm_time + comm_time
    print(f"Prompt phase, compute: {prompt_compute_time}, tp comm time: {prompt_tp_comm_time}, "
          f"pp comm time: {prompt_pp_comm_time}")

    token_step_compute_time = 0
    token_step_tp_comm_time = 0
    token_step_pp_comm_time = 0
    stage_num = len(stage_partitions)
    for i in range(stage_num):
        device_set = stage_device_sets[i]
        m_d = device_info[device_set[0]]['memory_bandwidth']
        c_d = device_info[device_set[0]]['flops']
        num_layers = stage_partitions[i]
        compute_time = compute_token_step_time_stage(batch_size, m_d, c_d, num_layers, len(device_set),
                                                     h_dim, b_type)
        print(f"Token step phase stage-<{i}> compute time {compute_time}")
        token_step_compute_time += compute_time
        comm_time = communicate_token_step_time_stage(batch_size, num_layers, device_set, delay_matrix,
                                                      bandwidth_matrix, h_dim, b_type)
        print(f"Token step phase stage-<{i}> comm time {comm_time}")
        token_step_tp_comm_time += comm_time
    for i in range(stage_num):
        if i < stage_num - 1:
            comm_time = communication_pipeline_token_step_time_cross_stage(batch_size, stage_device_sets[i],
                                                                           stage_device_sets[i + 1], delay_matrix,
                                                                           bandwidth_matrix, h_dim, b_type)
            print(f"Token_step phase pipeline stage-<{i}, {i + 1}> comm time {comm_time}")
        else:
            if stage_num != 1:
                comm_time = communication_pipeline_token_step_time_cross_stage_last(batch_size, stage_device_sets[i],
                                                                                    stage_device_sets[0], delay_matrix,
                                                                                    bandwidth_matrix)
                print(f"Token_step phase pipeline stage-<{i}, {0}> comm time {comm_time}")
        token_step_pp_comm_time + comm_time
    print(f"Token step, compute: {token_step_compute_time}, tp comm time: {token_step_tp_comm_time}, "
          f"pp comm time: {token_step_pp_comm_time}")
    total_time = prompt_compute_time + prompt_tp_comm_time + prompt_pp_comm_time
    total_time += (token_step_compute_time + token_step_tp_comm_time + token_step_pp_comm_time) * seq_out
    print(f"Total time: {total_time}")
    print(f"Throughput: {batch_size*seq_out/total_time}")
    return total_time

