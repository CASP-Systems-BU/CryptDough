# CryptDough: A unified analytics engine for secure Multi-Party Computation

CryptDough, aka CDough, is a Multi-Party Computation (MPC) framework built for secure mixed workload applications including Machine Learning inference, Relational queries, and timeseries analytics. Users can implement their target computation using declarative protocol-agnostic API. The framework is built for ease of use for non-MPC experts while enabling extensibility for cryptographers interested in implementing more MPC protocols or operators.

You can find the [complete documentation here](https://casp-systems-bu.github.io/CryptDough/).

## Table of Contents

- [Dependencies](#dependencies)
- [Building CryptDough](#building-cryptdough)
  - [Single-Node](#single-node)
  - [Cluster](#cluster)
- [Running CryptDough](#running-cryptdough)
  - [Compiling cdough Programs](#compiling-cdough-programs)
  - [`run_experiment.py`](#run_experimentpy)
  - [Running CryptDough Programs Locally](#running-cryptdough-programs-locally)
  - [Running CryptDough Programs on Multiple Servers](#running-cryptdough-programs-on-multiple-servers)
  - [Running CryptDough Programs in Simulated WAN](#running-cryptdough-programs-in-simulated-wan)
  - [Running the Test Suite](#running-the-test-suite)
  - [Running Queries](#running-queries)
- [Writing New CryptDough Programs](#writing-new-cdough-programs)
- [Creating a Cluster](#creating-a-cluster)

This repository is organized as follows:
- `bench/`: MPC applications used for benchmarking primitives, operators, and different literature computations (Queries and models).
- `doc/`: Framework documentation
- `examples/`: Example applications
- `include/`: Framework implementation including (i) core functionality of CryptDough (`core/`), including the implementation of the MPC primitives for the different protocols (`core/protocols/`), containers with data abstractions for the supported workloads (`core/containers/`), MPC operators (`core/operators/`), and party communication interfaces and implementation (`core/communication/`), as well as the framework backend (`backend/`).
- `scripts/`: Various scripts for testing and benchmarking CryptDough
- `tests/`: the test suite

## Dependencies
The project is written in C++ 20 and requires installation of (i) apt-available packages found in [`_setup_required.sh`](./scripts/setup/_setup_required.sh) and (ii) open source projects on github such as {[`libOTe`](./scripts/setup/_clone_libote.sh), [`Blaze`](./scripts/setup/_setup_blaze.sh), [`SecureJoin`](./scripts/setup/_setup_securejoin.sh)}.

### Ubuntu
You can install all dependencies on Ubuntu by running the following command:
```
mkdir build; cd build && ../scripts/setup/setup.sh
```

### Other operating systems
For non-Ubuntu systems (Mac, other \*nix), you will need to install these packages yourself:
```
git cmake pkg-config build-essential manpages-dev gfortran wget libsqlite3-0 libsqlite3-dev libsodium23 libsodium-dev libopenmpi3 libopenmpi-dev openmpi-bin openmpi-common python3 python3-pip libtool autoconf automake
```

Note that: OpenMPI used for one of the communicator interface implementation. The other one called "nocopy" does not use it.

In addition to the following open source libraries:
- [libOTe](https://github.com/osu-crypto/libOTe), used for secure Beaver triples (2PC).
- [secure-join](https://github.com/Visa-Research/secure-join), used for a two-party oblivious pseudorandom function (OPRF) with secret-shared output.
- [Blaze](https://bitbucket.org/blaze-lib/blaze.git), used for Machine Learning operators, can be turned on (similarly off) using command `cmake -DUSE_BLAZE=ON ..`

> Note: the above dependencies may impose additional restrictions on the environment. For example, they may not support all Linux distributions that cdough's online phase supports.

> [!IMPORTANT]
> You should still run `setup.sh` even if you install the dependencies manually, because other libraries will be compiled and built manually. You can ignore errors from, e.g. `apt` not being installed.

## Building CryptDough

CryptDough is a multi-party computation framework and requires installation on different nodes, where each node is a computing party. However, for testing purposes, we enable running computation from either a single-node or a cluster.

Please, review the following subsections for information for each mode. All instructions assume \*nix systems.

### Single-Node

When running an MPC computation in single-node mode, each computing party will be a different process in this node. This mode requires the typical dependencies setup (see [above](#dependencies)). Simply clone this repository then run the setup script as follows:

```bash
$ git clone https://github.com/CASP-Systems-BU/CryptDough
$ mkdir -p ./CryptDough/build
$ cd ./CryptDough/build
$ ../scripts/setup/setup.sh
```

If you do not have `apt` (other Linux, macOS, etc.) you will need to install the
dependencies manually (see [above](#dependencies)).

### Cluster

If you want to use CryptDough where each computing party runs as a process on a separate node, you need to establish connection among the computing nodes then choose an arbitrary node to be `node0`. This `node0` should have ssh-access to the other nodes. We will use this node in this setup to compile the MPC program, distribute it to other computing parties, and coordinate the computation. We provide instructions on how you can do so using different cloud service providers [below](#creating-a-cluster).

> [!WARNING]
> Having SSH access into all other nodes, of course, compromises the non-collusion assumption of multi-party computation. We require SSH access here for ease of development and testing. However, in a real system, cdough nodes would need to be configured out-of-band. We do not currently support this setup.
>
> Connections between parties do _not_ use TLS for MPC. SSH is only used for setup.

Make sure that the repo is cloned on `node0` then:

- First, we run `scripts/_update_hostfile.sh` to write to `/etc/hosts/` so that we can refer to the servers as `node0`, `node1`, etc.
- Second, we run the deployment script `./scripts/orchestration/deploy.sh` with the location of the repository and the names of the servers. The following example assumes a four-party setup, although we can achieve a setup with fewer parties by simply not entering the corresponding IP address(es) and name(s). Replace `<ip-X>` with the IP address of `nodeX`.

```bash
$ git clone https://github.com/CASP-Systems-BU/CryptDough
$ cd CryptDough
$ ./scripts/setup/_update_hostfile.sh -x node -i <ip-0>,<ip-1>,<ip-2>,<ip-3>
$ ./scripts/orchestration/deploy.sh ~/CryptDough node0 node1 node2 node3
```

`deploy.sh` runs `setup.sh`, so we do not need to run `setup.sh` explicitly. If you are on a machine without `apt`, you will need to install dependencies manually **on all machines**, as specified [above](#dependencies).

## Running CryptDough

This section describes the process to compile and run programs. To run programs, you must be in the `build` directory. You can follow the standard `cmake` patterns for compiling cdough programs, and we provide commands for running programs in a local or distributed fashion. Compilation is the same in either case.

### Compiling cdough Programs

Assuming all dependencies are installed correctly, the following example compiles a single program, [TPCH Q1](https://github.com/CASP-Systems-BU/CryptDough/blob/main/bench/queries/tpch/q1.cpp), in the replicated three-party protocol (the default).

```bash
$ mkdir build
$ cd build
$ cmake .. -DPROTOCOL=3
$ make q1
```

Various options can be specified to `cmake`.
- `-DPROTOCOL=N` to change the protocol. We currently support
   - `-DPROTOCOL=1` a single-party plaintext test protocol
   - `-DPROTOCOL=2` [ABY](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/aby-framework-efficient-mixed-protocol-secure-two-party-computation/) two party dishonest majority protocol with Beaver Triples
   - `-DPROTOCOL=3` [Araki et al.](https://eprint.iacr.org/2016/768) three party replicated honest majority protocol (the default)
   - `-DPROTOCOL=4` [Fantastic Four](https://eprint.iacr.org/2020/1330) honest-majority malicious 4PC protocol
   - `-DPROTOCOL=5` [SPDZ2k](https://eprint.iacr.org/2018/482.pdf) dishonest-majority malicious protocol with arbitrary number of parties. We currently support the online phase only.
- `-DNO_X86_SSE=1` to disable x86 hardware optimizations (you will get warnings otherwise if built on ARM platforms, like newer Macs)
- `-DPROFILE=1` enable profiling (compile with `-pg`)
- `-DEXTRA=XXX` pass the additional argument `XXX` to `make`
- `-DCOMM=XXX` enable the given communicator. Options are `"MPI" "NOCOPY"`. If you do not specify anything, CMake will use `NOCOPY`.
- `-DTRIPLES=XXX` specify the kind of Beaver triples to use for 2PC (`ZERO` (all zeros, for profiling the online phase), `DUMMY` (insecurely generated, fast), or `REAL` (secure)).

We provide some `cmake` shortcuts to make compiling multiple executables easier.

```bash
# compile everything
$ make -j
# compile tests
$ make -j tests-only
# compile all tpch queries
$ make -j tpch-queries
# compile queries for the secretflow comparison
$ make -j secretflow-queries
# compile all other queries
$ make -j other-queries
```

See `debug/cdough_debug.h` and `CMakeLists.txt` for more information on compile options. Not all compile options are made available via CMake and instead must be manually configured within `cdough_debug.h`.

### `run_experiment.py`

We provide an execution harness script, `run_experiment.py`, which automates the compilation and execution process. **This is the recommended method of running cdough programs.**

To see a comprehensive set of options for the `run_experiment` script, simply run it with `-h` to display a help message:

```bash
$ ../scripts/run_experiment.py -h
```

Key options include:
- `-p` Protocol (1-5); default: 3
- `-npc` Number of parties; default: protocol number or 2 for SPDZ2k
- `-s` Setting (same/lan/wan); default: same
- `-c` Communicator (mpi/nocopy); default: mpi for same, nocopy otherwise
- `-n` Number of communicator threads (NoCopyComm only); default: -1
- `-r` Number of rows (comma-separated INT or a^b expressions); default: 2^20
- `-d` Use powers of 10 for -r flag (allows ranges like 6-8 to expand to [10^6, 10^7, 10^8])
- `-f` Scale factor for TPC-H queries (overrides -r if set)
- `-t` Number of threads as powers of 2 (MIN or MIN-MAX)
- `-T` Number of threads (arbitrary); default: 0
- `-b` Batch size; default: -12
- `-e` Number of repetitions; default: 1
- `-o` Optimization level (0-3); default: 2
- `-y` Type of Beaver triples for 2PC (zero/dummy/real); default: zero
- `-m` Additional cmake arguments (can be repeated)
- `-a` Additional experiment arguments (can be repeated)
- `-x` Prefix for remote nodes; default: node

### Running CryptDough Programs Locally

For testing and development, you may want to run cdough programs locally. In this setting, separate processes act as each node of the multi-party computation.

A minimal test:

```bash
$ cd build
$ ../scripts/run_experiment.py test_primitives
# ... everything should pass ...
```

This command will run the program `test_primitives.cpp` with all default options:
- Replicated 3PC
- `same` environment
- MPI communicator
- 1 thread
- etc.

> [!TIP]
> We recommend always using `mpi` for local (`-s same`) tests. The no-copy communicator performs poorly over the loopback interface.

We can try a different program:
```bash
$ ../scripts/run_experiment.py micro_primitives    
# ...
Vector 1048576 x 32b
[=SW]            Start
[ SW]      Reserve MUL 1.625e-06 sec
[ SW]      Reserve AND 4.25e-06 sec
[ SW]              AND 0.01723  sec
[ SW]             MULT 0.01167  sec
[ SW]               EQ 0.04605  sec
[ SW]               GR 0.07226  sec
[ SW]              RCA 0.1298   sec
[ SW]             RCA< 0.05846  sec
[ SW]      Dot Product 0.008903 sec
```

More rows ($2^{24}\approx 16\mathrm{M}$):
```bash
$ ../scripts/run_experiment.py -r 2^24 micro_primitives
# ...
Vector 16777216 x 32b
[=SW]            Start
[ SW]      Reserve MUL 9.59e-07 sec
[ SW]      Reserve AND 3.541e-06 sec
[ SW]              AND 0.1686   sec
[ SW]             MULT 0.1992   sec
[ SW]               EQ 1.14     sec
[ SW]               GR 1.499    sec
[ SW]              RCA 1.913    sec
[ SW]             RCA< 0.8353   sec
[ SW]      Dot Product 0.1913   sec
```

A different protocol (Malicious-secure Fantastic 4PC):

```bash
$ ../scripts/run_experiment.py -r 2^24 -p 4 micro_primitives
# ...
Vector 16777216 x 32b
[=SW]            Start
[ SW]      Reserve MUL 1.167e-06 sec
[ SW]      Reserve AND 3.958e-06 sec
[ SW]              AND 1.126    sec
[ SW]             MULT 1.109    sec
[ SW]               EQ 6.42     sec
[ SW]               GR 8.01     sec
[ SW]              RCA 4.401    sec
[ SW]             RCA< 2.709    sec
[ SW]      Dot Product 1.449    sec
```

Or more threads:

```bash
$ ../scripts/run_experiment.py -r 2^24 -p 4 -T 4 micro_primitives
# ...
Vector 16777216 x 32b
[=SW]            Start
[ SW]      Reserve MUL 1.421e-05 sec
[ SW]      Reserve AND 3.875e-06 sec
[ SW]              AND 0.7819   sec
[ SW]             MULT 0.5872   sec
[ SW]               EQ 3.194    sec
[ SW]               GR 4.201    sec
[ SW]              RCA 2.858    sec
[ SW]             RCA< 2.194    sec
[ SW]      Dot Product 0.8624   sec
```

You should not expect much of a speedup with more threads when running locally: this is the number of worker threads _per party_. There is also a main thread per party. In the above example, therefore, we have 4 parties, each with $4+1=5$ threads, but our test machine only has 8 cores.

### Running CryptDough Programs on Multiple Servers

CryptDough programs can be run over LAN just by changing the setting (`-s`) argument to `run_experiment.py`:

```
$ ../scripts/run_experiment.py -s lan -c nocopy -n 4 -T 8 micro_sorting
```

This runs the `micro_sorting` experiment, with the 3PC protocol (the default), on nodes `node0`, `node1`, and `node2`. We use the `nocopy` communicator with `4` communication threads and `8` worker threads. `run_experiment.py` takes care of configuring the other nodes, and copies the compiled binary from `node0` to all other nodes in the cluster. 

> [!WARNING]
> cdough programs will crash in mysterious ways if different versions of a binary are present on different hosts.

> [!TIP]
> We recommend using `-c nocopy` for LAN tests. We have found `-n 4` (4 communication threads) sufficient for the LAN environment.
>
> If your machines are named something else, you can specify a new prefix with `-x [node]`. However, our scripts assume a consistent numbering:
>
> ```bash
> $ ../scripts/run_experiment.py -s lan -p 4 -x lab-server- test_primitives
> ```
>
> This will run `test_primitives`, with the 4PC protocol, on nodes `lab-server-0`,`lab-server-1`, `lab-server-2`, and `lab-server-3`.

To check the scaling behavior of cdough, we can use the variable-thread (`-t`) argument. This example will run `micro_sorting` in LAN, with the 3PC protocol, using `1, 2, 4, 8, 16, 32` worker threads.

```bash
$ ../scripts/run_experiment.py -s lan -c nocopy -n 4 -t 0-5 micro_sorting
```

### Running CryptDough Programs in Simulated WAN

CryptDough demonstrates practical performance over the internet (WAN). However, it is tricky to deploy a geodistributed WAN cluster. We provide a way to apply artifical delay and bandwidth throttling, uses the Linux `tc qdisc`. If your cluster machines are physically in LAN setup, you can use this very easy to use WAN simulator as follows:


```bash
# Assuming pwd = ./build
# Turn on simWAN for node0 (implied), node1, node2, and node3
$ ../scripts/comm/cluster-wan-sim.sh on node{1,2,3}
# Run your experiment
# [Soon, run_experiment will handle running cluster-wan-sim for you]
$ ../scripts/run_experiment.py -s wan ...
# Disable simWAN
$ ../scripts/comm/cluster-wan-sim.sh off node{1,2,3}
```

This simple script makes some assumptions about the network topology (e.g., that all nodes are routable over the same interface), so modifications may be required for more complex deployments.

`cluster-wan-sim` also runs `ping` just as a sanity check:

```bash
# from node0
$ ping -q -c 3 node1
# ...
rtt min/avg/max/mdev = 0.177/0.235/0.270/0.041 ms
$ ./scripts/cluster-wan-sim.sh on node{1,2,3}
# ...
Enabling WAN on bond0@ 6Gbit, 10ms
# ...
# Latency has increased:
rtt min/avg/max/mdev = 20.321/20.394/20.446/0.053 ms
# ...
$ ./scripts/cluster-wan-sim.sh off node{1,2,3}
# ...
# Back to normal:
rtt min/avg/max/mdev = 0.150/0.200/0.258/0.044 ms
```

You can use `iperf` to check the bandwidth limiter.

### Running the Test Suite

Tests can be compiled and run like any other program. Tests can be run in LAN, but this just slows things down without exercising much functionality, so we recommend running tests locally. We provide a script to automate running the entire test suite.

The syntax is:

```
../scripts/testing/run_multithread_test.sh [protocol=3] [threads=1]
```

For example:

```bash
# Run the test suite under 3PC with 1 thread...
$ ../scripts/testing/run_multithread_test.sh
# 4PC and 2 threads...
$ ../scripts/testing/run_multithread_test.sh 4 2
# 2PC and 1 thread
$ ../scripts/testing/run_multithread_test.sh 2
```

### Running Queries

Queries (`/bench/queries`) can be run in the same way as other programs, but we also provide a test harness to run an entire suite of queries.

```bash
$ ../scripts/query-experiments.sh 
Usage: ../scripts/query-experiments.sh <tpch|other|secretflow> <sf> <protocol> <threads> <enviro=lan> <query_range=1..22>
```

The first argument specifies the suite of queries:
- `tpch`: run all tpch queries. Optionally, specify a range of queries as the final argument in Bash syntax:
  - Comma separated values: `1,2,9,14` will run TPC-H queries 1, 2, 9, and 14.
  - Ranges: `8..20` will run TPC-H queries 8 through 20, inclusive.
  - The two syntaxes cannot currently be mixed.
- `other`: run the other 9 queries from the cdough paper
- `secretflow`: run the SecretFlow queries

The next argument, `sf` refers to Scale Factor. For TPC-H, this is a well-defined term in the specification. For other queries, we define SF1 to be approximately the same size, on average; that is, about 5M rows.

`enviro` is as in `run_experiment.py`: it may be `lan`, `wan`, or `same` (local).

## Writing New cdough Programs

See the [examples](https://github.com/CASP-Systems-BU/CryptDough/tree/main/examples) for some simple programs that showcase the main features of the framework. **Example 0** contains a minimal C++ file you can use to get started with your own programs.

`CMakeLists.txt` only looks in certain directories for C++ sources. New programs should be placed into the `examples/`, `bench/`, or `tests/` directories. Alternatively, modify `CMakeLists.txt` to look for your sources.

## Creating a Cluster

We provide instructions for launching a cluster both on AWS and on CloudLab. For AWS, we provide a cluster launching script, and for CloudLab, we provide simple step-by-step instructions. Of course, other types of clusters will work, but we do not provide an automated process for creating them.

### AWS

To deploy an AWS cluster, you will need an AWS access key and secret key. This is often already stored inside `~/.aws/credentials`, in which case no action is needed. If it is not (or if you do not wish to create such a file), enter your access key and secret key into `scripts/orchestration/aws/secrets.sh` and run the following command before proceeding.

```bash
$ source scripts/orchestration/aws/secrets.sh
```

To create the cluster, run the following command.

```bash
$ ./scripts/orchestration/aws/deploy-aws-cluster.sh <cluster-name> <number-of-nodes>
```

You will now have access to a cluster with your specified number of nodes. To setup the repository, follow the instructions in the [section](#building-cryptdough) about building cdough.

### CloudLab

From the main page in CloudLab, execute the following operations.

```
click the 'Experiments' dropdown in the upper left and select 'Start Experiment'
click 'Next'
enter your desired number of nodes (2, 3, or 4)
select either Ubuntu 22 or Ubuntu 24 as your operating system
select any Intel-based server type
click 'Next'
give the cluster an arbitrary name
use the default node names (node0, node1, ...)
click 'Next'
click 'Finish'
```

When the cluster has been successfully created, you can SSH into `node0` and follow the instructions in the [section](#building-cryptdough) about building cdough. Note that on CloudLab, you **do not need** to run the `_update_hostfile.sh` script, as CloudLab has already configured the servers to recognize each other as `node0`, `node1`, and so on.

