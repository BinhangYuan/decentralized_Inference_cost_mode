# Decentralized Inference Cost Model


## Estimated time and throughput

#### Data center 1
  - Setting: [8XA100-80G PCIe]

| batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|------------|-----------|------------|-------------|----------------------|
| 1          | 128       | 32         | 0.901       | 35.51                |
| 16         | 128       | 32         | 1.523       | 335.97               |


#### Spot cross region: 
  - Setting: [4XA100-40G PCIe, 4XA100-40G PCIe, 4XA100-40G PCIe, 4XA100-40G PCIe]
  - Scheduled:
  
| batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|------------|-----------|------------|-------------|----------------------|
| 1          | 128       | 32         | 1.780       | 17.97                |
| 16         | 128       | 32         | 2.696       | 189.89               |

  - Pure Pipeline (scheduled):

| batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|------------|-----------|------------|-------------|----------------------|
| 1          | 128       | 32         | 7.064       | 4.536                |
| 16         | 128       | 32         | 9.730       | 52.61                |

 - 4X4 Random Assign:

| batch-size | input seq | output seq | runtime (s)   | throughput (token/s) |
|------------|-----------|------------|---------------|----------------------|
| 1          | 128       | 32         | 1906.78       | 0.016                |
| 16         | 128       | 32         | 1970.73       | 0.259                |


#### Miner Machine

- Scheduled: pp=4, dp=12 (PCIe 4.0: 4GB/s)

|  batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|-------------|-----------|------------|-------------|----------------------|
| 1           | 128       | 32         | 2.34        | 13.67                |
| 16          | 128       | 32         | 7.63        | 67.10                |


- Scheduled: pp=4, dp=12 (PCIe 2.0: 0.5GB/s)

|  batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|-------------|-----------|------------|-------------|----------------------|
| 1           | 128       | 32         | 4.59        | 6.96                 |
| 16          | 128       | 32         | 43.72       | 11.71                |


#### Miner Machine 1GPU per Machine

- Delay 50 ms, Bandwidth 1 Gb/s
- Scheduled: pp=4, dp=12

|  batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|-------------|-----------|------------|-------------|----------------------|
| 1           | 128       | 32         | 6981.93     | 0.0045               |
| 16          | 128       | 32         | 7137.54     | 0.07                 |


#### Petals Paper Simulated Results

- 3 A100-80G, 1 Gb/s, 5ms

| batch-size | input seq | output seq | runtime (s) | throughput (token/s) |
|------------|-----------|------------|-------------|----------------------|
| 1          | 128       | 32         | 3.616       | 8.848                |
| 64         | 128       | 32         | 14.85       | 137.85               |


## Some Comparisonm between Estimation and Real-time

#### OPT-13B FluidStack
- 2 X A4000
- Inter-machine delay (ping results): 0.5 ms, bandwidth (iperf3 results): 9.65 Gbps 
- E: estimated, B: benchmarked 

- TP=2, PP=1:

| batch-size | input seq | output seq | E/B runtime(s) | E/B throughput (token/s) |
|------------|-----------|------------|----------------|--------------------------|
| 1          | 512       | 1          | /  1.67        | /                        |
| 1          | 512       | 32         | /  6.51        | /                        |
| 1          | 512       | 64         | /  11.74       | /                        |
| 1          | 512       | 128        | /  25.97       | /                        |
| 16         | 512       | 1          | /  1.72        | /                        |
| 16         | 512       | 32         | /  13.39       | /                        |
| 16         | 512       | 64         | /  24.01       | /                        |
| 16         | 512       | 128        | /  46.55       | /                        |

- TP=1, PP=2:

| batch-size | input seq | output seq | E/B runtime(s) | E/B throughput (token/s) |
|------------|-----------|------------|----------------|--------------------------|
| 1          | 512       | 1          | /   0.16       | /                        |
| 1          | 512       | 32         | /   2.14       | /                        |
| 1          | 512       | 64         | /   4.13       | /                        |
| 1          | 512       | 128        | /   8.13       | /                        |
| 16         | 512       | 1          | /   0.26       | /                        |
| 16         | 512       | 32         | /   3.27       | /                        |
| 16         | 512       | 64         | /   6.41       | /                        |
| 16         | 512       | 128        | /   12.79      | /                        |

#### OPT-30B FluidStack
- 4 X A4000
- Inter-machine delay (ping results): 0.5 ms, bandwidth (iperf3 results): 9.65 Gbps 
- E: estimated, B: benchmarked 

- TP=4, PP=1:

| batch-size | input seq | output seq | E/B runtime(s) | E/B throughput (token/s) |
|------------|-----------|------------|----------------|--------------------------|
| 1          | 512       | 1          | / 2.06         | /                        |
| 1          | 512       | 32         | / 4.84         | /                        |
| 1          | 512       | 64         | / 7.93         | /                        |
| 1          | 512       | 128        | / 13.83        | /                        |
| 16         | 512       | 1          | / 2.34         | /                        |
| 16         | 512       | 32         | / 11.06        | /                        |
| 16         | 512       | 64         | / 18.77        | /                        |
| 16         | 512       | 128        | / 35.16        | /                        |


- TP=2, PP=2:

| batch-size | input seq | output seq | E/B runtime(s) | E/B throughput (token/s) |
|------------|-----------|------------|----------------|--------------------------|
| 1          | 512       | 1          | /   1.16       | /                        |
| 1          | 512       | 32         | /   3.84       | /                        |
| 1          | 512       | 64         | /   7.26       | /                        |
| 1          | 512       | 128        | /   12.66      | /                        |
| 16         | 512       | 1          | /   1.26       | /                        |
| 16         | 512       | 32         | /   6.85       | /                        |
| 16         | 512       | 64         | /   13.24      | /                        |
| 16         | 512       | 128        | /   25.78      | /                        |


- TP=1, PP=4:

| batch-size | input seq | output seq | E/B runtime(s) | E/B throughput (token/s) |
|------------|-----------|------------|----------------|--------------------------|
| 1          | 512       | 1          | /              | /                        |
| 1          | 512       | 32         | /              | /                        |
| 1          | 512       | 64         | /              | /                        |
| 1          | 512       | 128        | /              | /                        |
| 16         | 512       | 1          | /              | /                        |
| 16         | 512       | 32         | /              | /                        |
| 16         | 512       | 64         | /              | /                        |
| 16         | 512       | 128        | /              | /                        |

