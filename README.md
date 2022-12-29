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