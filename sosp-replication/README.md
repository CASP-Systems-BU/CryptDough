# CryptDough Artifact Evaluation Guide for SOSP 2026
Welcome to the artifact evaluation guide for CryptDough. This document provides instructions to reproduce the results published in the SOSP 2026 paper, CryptDough. For information on how to use the framework and build applications, please refer to the main [README](../README.md) file.

We are applying for all three badges: {[Available](#available-badge), [Functional](#functional-badge), [Reproduced](#reproduced-badge)}.

## Available Badge

We make the artifact available to reviewers in our [GitHub repository](https://github.com/CASP-Systems-BU/CryptDough), which includes a README file highlighting the key dependencies, a getting-started guide, and the main features. Additionally, we provide extensive [documentation](https://casp-systems-bu.github.io/CryptDough/) on the framework and its API.

We plan to attach an open-source license to the artifact and upload it to Zenodo after approval and before the artifact decision deadline.

## Functional Badge

We provide a modular and extensible MPC framework that supports machine learning, relational queries, and time-series analytics. Our main claim is that our architecture, abstractions, and enabling vectorized algorithms yield a compact artifact whose performance is close to or better than state-of-the-art systems in each field.

We demonstrate this through:
1. `core/protocols`: The MPC protocol primitives interface and 4 implemented protocols (ABY, ABY3, FantasticFour, and SPDZ2k).
2. `core/communication`: The communication component interface and two implementations based on (i) TCP and (ii) MPI.
3. `core/random`: Preprocessing interfaces and implementations.
4. `backend/common`: The runtime engine, providing transparent multi-threaded execution.
5. `core/containers`: Data abstractions supporting the machine learning, relational query, and time-series analytics functionalities.
6. `core/operators`: Implementations of secure operators built on top of the MPC primitives.


We provide a few examples to showcase our supported analytics:
1. `bench/tva/cloud.cpp`: A time-series analytics application that uses a gap session window operator.
2. `bench/queries/other/distinct_patients.cpp`: A relational query application that uses filtering, sorting, and aggregation operators.
3. `bench/models/vgg16.cpp`: A machine learning model that uses convolution, ReLU, average-pooling, and fully connected layer operators.
4. `bench/multi-workload/patients_spo.cpp`: A multi-workload computation that combines time-series analytics, relational queries, and machine learning inference.


## Reproduced Badge

The experimental section in the paper supports two claims: (i) we can implement a multi-workload query and easily switch between the different supported protocols, and (ii) our approach is not at a disadvantage against specialized MPC systems and, in fact, our performance can sometimes exceed that of prior state-of-the-art systems.

We compare against 5 prior state-of-the-art systems.
1. [`TVA`](https://www.usenix.org/conference/usenixsecurity23/presentation/faisal) in [commit](https://github.com/CASP-Systems-BU/tva/tree/e777cbadd28ce5da7d98c657aca64275bac164ac): A specialized MPC system for time series analytics.
2. [`ORQ`](https://dl.acm.org/doi/10.1145/3731569.3764833) in [commit](https://github.com/CASP-Systems-BU/orq/tree/2d7946a95f6d1d49e020789b70a6cfbdc1198a46): A specialized MPC system for relational queries.
3. [`Pigeon`](https://petsymposium.org/popets/2025/popets-2025-0090.pdf) in [commit](https://github.com/chart21/hpmpc/tree/3d714566858739b430267c366dc9313ece0e0394): A specialized MPC system in machine learning that utilizes both CPU and GPU for local computation.
4. [`Piranha`](https://www.usenix.org/conference/usenixsecurity22/presentation/watson) in [commit](https://github.com/ucbrise/piranha/tree/dfbcb59d4e24ab69eb3606b49a102e602fdbee87): A specialized MPC system in machine learning utilizing GPU for local computation.
5. [`MP-SPDZ`](https://eprint.iacr.org/2020/521): A general-purpose MPC framework.


All our experiments run on Ubuntu Linux 24.04.4 LTS and use 4 different settings, {BM-LAN, BM-WAN, AWS-LAN, AWS-WAN}, according to the following tags.
1. **BM**: Our own hosted bare-metal machines (AMD EPYC 9655P 96-core machines).
2. **AWS**: AWS EC2 instances.
3. **LAN**: Up to 25 Gbps bandwidth and 0.17-0.25 ms round-trip time (RTT).
4. **WAN**: Up to 6 Gbps bandwidth and 20 ms RTT; this is the same as LAN but with the flag `-s wan` specified when using the `run_experiment.py` script.


The experimental section has 7 experiments, comparing against 5 different baselines.
1. **[Fig 5 Multi-workload Query](#fig-5-multi-workload-query)**: Supports claim #1 and runs in BM-LAN and BM-WAN
2. **[Fig 6 Comparison with ORQ](#fig-6-comparison-with-orq)**: Supports claim #2 and runs in BM-LAN and BM-WAN
3. **[Fig 7 Comparison with TVA](#fig-7-comparison-with-tva)**: Supports claim #2 and runs in BM-LAN and BM-WAN
4. **[Fig 8 Scalability](#fig-8-scalability)**: Supports claim #2 and runs in BM-LAN
5. **[Table 2 Comparison with Pigeon (3PC)](#table-2-comparison-with-pigeon-3pc)**: Supports claim #2 and runs in BM-LAN and BM-WAN
6. **[Table 3 Comparison with Piranha (2PC)](#table-3-comparison-with-piranha-2pc)**: Supports claim #2 and runs in BM-LAN and BM-WAN for CryptDough but in AWS-LAN and AWS-WAN for piranha since it requires GPU.
7. **[Table 4 Comparison with MP-SPDZ (SPDZ2k)](#table-4-comparison-with-mp-spdz-spdz2k)**: Supports claim #2 and runs in BM-LAN and BM-WAN.


### Setup

First, we [set up CryptDough](#cryptdough-installation); then we use the CryptDough scripts to [install the state-of-the-art systems](#baseline-installation) we compare against.

Because the artifact evaluation requires multiple nodes, some with GPU support, we offer access to ready-to-use clusters. If you choose to use them, you can skip this section and go directly to the [experiments section](#experiments) below.

#### CryptDough installation
Please refer to the main [README](../README.md) file to learn more about the system architecture and requirements. For these experiments, we are interested in the cluster setup, which we install as follows:
1. Ensure that you have 4 nodes connected together. Call them `node0`, `node1`, `node2`, and `node3`, and ensure that `node0` has SSH access to the other machines. Note that SSH access is used only for benchmarking purposes and is not required in a real production deployment.
2. Clone this repository and enter the directory:
```bash
$ git clone https://github.com/CASP-Systems-BU/CryptDough
$ cd CryptDough
```
3. Install CryptDough on all machines, where, for instance, `<ip-0>` is the IP address of `node0`. `_update_hostfile.sh` sets an alias so that `<ip-0>` maps to `nodei`; hence, do not replace `node0` in the second command.

```bash
$ ./scripts/setup/_update_hostfile.sh -x node -i <ip-0>,<ip-1>,<ip-2>,<ip-3>
$ ./scripts/orchestration/deploy.sh ~/CryptDough node0 node1 node2 node3
```

#### Baseline installation
After [installing CryptDough](#cryptdough-installation), you can run the following script to install all baseline systems.

```bash
$ ./sosp-replication/setup/setup_baselines.sh node0 node1 node2 node3
```

### Experiments
All of the following experiments are long-running. To avoid improper termination, please use [screen](https://linuxize.com/post/how-to-use-linux-screen/) to run them. All commands are run from `node0`.

After running all experiments, you can proceed to the [plotting section](#plotting).

#### Fig 5: Multi-workload Query 
(Human time: 2 minutes, runtime: 3 hours)

This experiment supports claim #1 and runs in BM-LAN and BM-WAN. In this experiment, we run our multi-workload query using the 4 protocols {2PC, 3PC, 4PC, SPDZ2k} in both LAN and WAN setups.

From `node0`, run the following command and the results will be logged in `./sosp-replication/data/logs/fig5`.
```bash
$ ./sosp-replication/experiments/figure-5/fig5.sh
```

#### Fig 6 Comparison with ORQ
(Human time: 2 minutes, runtime: 5 hours)

This experiment supports claim #2 and runs in BM-LAN and BM-WAN. In this experiment, we compare against ORQ performance using 8 queries for both the 2PC and 3PC protocols.

For CryptDough, first run the following command and the results will be logged in `./sosp-replication/data/logs/fig6-cdough`.
```bash
$ ./sosp-replication/experiments/figure-6/fig6-cdough.sh
```

For ORQ, run the following command and the results will be logged in `./sosp-replication/data/logs/fig6-orq`.
```bash
$ ./sosp-replication/experiments/figure-6/fig6-orq.sh
```

#### Fig 7 Comparison with TVA
(Human time: 2 minutes, runtime: 5 hours)

This experiment supports claim #2 and runs in BM-LAN and BM-WAN. In this experiment, we compare against TVA performance using 3 queries for both 3PC and 4PC protocols.

For CryptDough, first run the following command and the results will be logged in `./sosp-replication/data/logs/fig7-cdough`.
```bash
$ ./sosp-replication/experiments/figure-7/fig7-cdough.sh
```

For TVA, run the following command and the results will be logged in `./sosp-replication/data/logs/fig7-tva`.
```bash
$ ./sosp-replication/experiments/figure-7/fig7-tva.sh
```

#### Fig 8 Scalability
(Human time: 2 minutes, runtime: 12 hours)

This experiment supports claim #2 and runs in BM-LAN. In this experiment we show scalability for [Comparison - RCA/PPA - Conv2d - Sorting] using the 3PC protocol.

Run the following command and the results will be logged in `./sosp-replication/data/logs/fig8-cdough`.
```bash
$ ./sosp-replication/experiments/figure-8/fig8-cdough.sh
```

#### Table 2 comparison with Pigeon (3PC)
(Human time: 2 minutes, runtime: 1 hour)

This experiment supports claim #2 and runs in BM-LAN and BM-WAN. In this experiment, we compare against Pigeon ML inference for the 3PC protocol. Both systems use model parallelism, where multiple processes perform inference at the same time. We use the slowest process latency in both systems to determine the end-to-end latency.

For CryptDough, first run the following command and the results will be logged in `./sosp-replication/data/logs/table2-cdough`.
```bash
$ ./sosp-replication/experiments/table-2/table2-cdough.sh
```

For Pigeon, run the following command and the results will be logged in `./sosp-replication/data/logs/table2-pigeon`.
```bash
$ ./sosp-replication/experiments/table-2/table2-pigeon.sh
```

#### Table 3 comparison with Piranha (2PC)
(Human time: 2 minutes, runtime: 1 hour)

This experiment supports claim #2 and runs in BM-LAN and BM-WAN for CryptDough, but in AWS-LAN and AWS-WAN for Piranha since it requires a GPU. In this experiment, we compare against Piranha using the 2PC protocol.

For CryptDough, first run the following command and the results will be logged in `./sosp-replication/data/logs/table3-cdough`.
```bash
$ ./sosp-replication/experiments/table-3/table3-cdough.sh
```

For Piranha, run the following command and the results will be logged in `./sosp-replication/data/logs/table3-piranha`.
```bash
$ ./sosp-replication/experiments/table-3/table3-piranha.sh
```

#### Table 4 comparison with MP-SPDZ (SPDZ2k)
(Human time: 2 minutes, runtime: 1 hour)

This experiment supports claim #2 and runs in BM-LAN and BM-WAN. In this experiment, we compare CryptDough against MP-SPDZ.

For CryptDough, first run the following command and the results will be logged in `./sosp-replication/data/logs/table4-cdough`.
```bash
$ ./sosp-replication/experiments/table-4/table4-cdough.sh
```

For MP-SPDZ, run the following command and the results will be logged in `./sosp-replication/data/logs/table4-mpspdz`.
```bash
$ ./sosp-replication/experiments/table-4/table4-mpspdz.sh
```

### Plotting
After running the [experiments above](#experiments), the logs are collected and stored under `./sosp-replication/data/logs/`.

To extract this data into CSV tables, run the following command:
```bash
$ sosp-replication/plotting/extract_data.sh
```

After extracting the data from the logs, run the plotting script; the paper plots will be produced under `sosp-replication/data/plots`.
```bash
$ ./sosp-replication/plotting/plot_benchmarks.py
```
