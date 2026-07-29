#!/bin/bash
#
# table3-piranha.sh — Run the Piranha (ucbrise/piranha) 2PC ML-inference
# experiments for Table 3, orchestrated from node0.
#
# Table 3 compares CryptDough against Piranha under the 2PC protocol. This
# script runs ONLY the Piranha side (the CryptDough side is table3-cdough.sh).
# It mirrors the six table3-cdough.sh experiments as secure inference (forward
# pass only), one forward pass per experiment:
#   * AlexNet + CIFAR-10   (batch 192)  <-> cdough "alexnet -r 192"
#   * VGG16   + CIFAR-10   (batch 192)  <-> cdough "vgg16 -r 192"
#   * VGG16   + ImageNet   (batch 8)    <-> cdough "vgg16-imagenet -r 8"
# each in LAN and WAN, with:
#   * party 0 -> node0 (listener; prints the timing/communication stats)
#   * party 1 -> node1 (connector)
# Logs are written to data/logs/table-3/piranha with the same names as the
# cdough logs ({lan,wan}-2pc-<model>.log holds party 0's output; a matching
# …-party1.log is kept for debugging).
#
# Architecture matching (what makes the experiments identical):
#   * bench/models/alexnet.cpp == Piranha files/models/alexnet-cifar10.json
#   * bench/models/vgg16.cpp   == Piranha files/models/vgg16-cifar10.json
#   * bench/models/vgg16-imagenet.cpp has no upstream Piranha counterpart, so
#     piranha-files/vgg16-imagenet.json (authored from that benchmark's layer
#     definitions) is copied into Piranha's files/models/ on both nodes.
# The datasets themselves are intentionally NOT installed: Piranha only warns
# when files/<dataset>/... is missing and feeds zero-filled batches, and the
# secure forward pass is input-value-independent, so timings are unaffected.
# Expect (and ignore) "Error opening ..." lines in the logs.
#
# WAN emulation: the WAN half applies wan-sim.py on|off on both nodes
# (tc netem, 6 Gbit rate + 10 ms per-link delay = 20 ms RTT), the same
# emulation run_experiment.py -s wan applies for the cdough side.
# cluster-wan-sim.sh is not used directly because it identifies node0 by
# `hostname --short`, which the other node cannot resolve on this cluster; the
# node0/node1 aliases (set by _update_hostfile.sh) are used instead. This
# needs passwordless sudo for tc on both nodes; an EXIT trap always disables
# the emulation again.
#
# Prerequisites (already handled by the artifact setup):
#   * Piranha is installed at sosp-replication/baselines/piranha on node0 and
#     node1 (see sosp-replication/setup/setup_piranha.sh), built with -DTWOPC
#     and -DFLOAT_PRECISION=26 against the side-by-side CUDA 11.8 toolkit.
#   * node0 has SSH access to node0 and node1, and the repository is deployed
#     to the same absolute path on both nodes.
#
# Tunables (env vars):
#   RUNS   Number of repetitions per experiment (default 1).
#
# Usage (from node0):
#   ./sosp-replication/experiments/table-3/table3-piranha.sh
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Piranha install directory and the Table 3 log directory. The repository is
# deployed to the same absolute path on every node, so piranha_dir is valid on
# each node reached over SSH. sosp-replication (script_dir/../..) always
# exists, so we can resolve the log dir's parent before the log dir itself is
# created.
piranha_dir="$(cd "$script_dir/../../baselines/piranha" && pwd)"
log_dir="$(cd "$script_dir/../.." && pwd)/data/logs/table-3/piranha"
mkdir -p "$log_dir"

# WAN emulation script shared with the CryptDough side (run_experiment.py -s wan).
# The repository is deployed to the same absolute path on both nodes, so the
# same path is valid on node1 over SSH.
wan_sim_py="$(cd "$script_dir/../../../scripts/profiling/comm" && pwd)/wan-sim.py"

# The piranha binary is linked against the side-by-side CUDA 11.8 runtime
# installed by setup_piranha.sh, which is not on the default loader path, so
# every invocation needs LD_LIBRARY_PATH.
CUDA_HOME="/usr/local/cuda-11.8"
PIRANHA_RUN_ENV="LD_LIBRARY_PATH=${CUDA_HOME}/lib64 CUDA_VISIBLE_DEVICES=0"

RUNS=${RUNS:-1}

# 2PC party-to-node mapping: party N runs on node N.
NODES=(node0 node1)

# Resolve a node alias/host to a dotted-decimal IPv4 address. Piranha's config
# lists party_ips as plain addresses handed to its socket layer, so aliases
# must be resolved before they go into the config.
resolve_ip() {
    local host="${1#*@}" # strip any "user@" prefix
    if [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf '%s' "$host" # already a dotted-decimal IPv4 address
        return 0
    fi
    local ip
    ip="$(python3 -c 'import socket, sys; print(socket.gethostbyname(sys.argv[1]))' "$host" 2>/dev/null)" || true
    if [[ -z "$ip" ]]; then
        echo "!! Could not resolve host '$host' to an IPv4 address." >&2
        exit 1
    fi
    printf '%s' "$ip"
}

IP0="$(resolve_ip "${NODES[0]}")"
IP1="$(resolve_ip "${NODES[1]}")"
echo "==> Wire IPs: party0=${IP0} (${NODES[0]}), party1=${IP1} (${NODES[1]})"

# Apply the WAN emulation state on both directions of the node0<->node1 link:
# locally toward node1 and on node1 toward node0. Each side auto-detects its
# interface from the (resolvable) peer alias.
set_wan() { # $1 = on|off
    local state="$1"
    python3 "$wan_sim_py" "$state" -H "${NODES[1]}" &&
        ssh "${NODES[1]}" "python3 '${wan_sim_py}' ${state} -H ${NODES[0]}"
}

# Always disable WAN emulation on exit so an aborted WAN half can never
# corrupt later LAN measurements.
wan_enabled=0
disable_wan() {
    if [[ "$wan_enabled" -eq 1 ]]; then
        echo "==> Disabling WAN emulation (cleanup)..."
        set_wan off
        wan_enabled=0
    fi
}
trap disable_wan EXIT

# Distribute the VGG16-ImageNet model file (absent from upstream Piranha) into
# Piranha's files/models/ on both nodes.
model_src="${script_dir}/piranha-files/vgg16-imagenet.json"
echo "==> Distributing vgg16-imagenet.json to Piranha's files/models/ on: ${NODES[*]}"
for host in "${NODES[@]}"; do
    if ! ssh "$host" "cat > '${piranha_dir}/files/models/vgg16-imagenet.json'" < "$model_src"; then
        echo "!! Failed to copy vgg16-imagenet.json to ${host}." >&2
        exit 1
    fi
done

# run_piranha <env-label> <model-tag> <model-json> <batch>:
# run one Table 3 experiment RUNS times. For each run it
#   1. generates the experiment config on node0 from the localhost sample
#      (2 parties, resolved wire IPs, the model file, the requested batch size,
#      and inference-only benchmarking flags: 1 epoch x 1 iteration, no test
#      pass, per-forward-pass runtime/communication stats),
#   2. copies the config to node1,
#   3. kills stale piranha processes on both nodes so their ports are free,
#   4. starts party 0 on node0 in the background (listener), waits until it is
#      actually listening (otherwise party 1 can exhaust its connection-retry
#      budget while party 0 is still initializing CUDA), then runs party 1 on
#      node1 in the foreground.
# Party 0's output (the timing stats) is appended to <env>-2pc-<model>.log and
# party 1's to <env>-2pc-<model>-party1.log.
run_piranha() {
    local env_label="$1" model_tag="$2" model_json="$3" batch="$4"
    local cfg="files/samples/table3_${model_tag}_config.json"
    local logf="${log_dir}/${env_label}-2pc-${model_tag}.log"
    local logf1="${log_dir}/${env_label}-2pc-${model_tag}-party1.log"

    echo "==> Generating ${cfg} on ${NODES[0]} (network=${model_json}, batch=${batch})..."
    if ! ssh "${NODES[0]}" "cd '${piranha_dir}' && python3 - '${IP0}' '${IP1}' '${model_json}' '${batch}' '${cfg}' '${env_label}-2pc-${model_tag}' <<'PYEOF'
import getpass, json, sys
ip0, ip1, model_json, batch, cfg_path, run_name = sys.argv[1:7]
user = getpass.getuser()
cfg = json.load(open('files/samples/localhost_config.json'))
cfg['num_parties'] = 2
cfg['party_ips']   = [ip0, ip1]
cfg['party_users'] = [user, user]
cfg['run_name']    = 'table3_' + run_name
cfg['network']     = model_json
# One secure inference forward pass over a single batch of the requested size.
cfg['custom_batch_size']       = True
cfg['custom_batch_size_count'] = int(batch)
cfg['custom_epochs']           = True
cfg['custom_epoch_count']      = 1
cfg['custom_iterations']       = True
cfg['custom_iteration_count']  = 1
cfg['inference_only']          = True
cfg['no_test']                 = True
cfg['last_test']               = False
cfg['lr_schedule']             = [3]
# Print per-forward-pass runtime and communication stats; skip accuracy, which
# is meaningless without datasets (zero-filled batches).
cfg['eval_inference_stats'] = True
cfg['eval_epoch_stats']     = True
cfg['eval_accuracy']        = False
json.dump(cfg, open(cfg_path, 'w'), indent=2)
print('wrote ' + cfg_path)
PYEOF"; then
        echo "!! Failed to generate ${cfg} on ${NODES[0]}." >&2
        exit 1
    fi

    ssh "${NODES[0]}" "cat '${piranha_dir}/${cfg}'" | \
        ssh "${NODES[1]}" "cat > '${piranha_dir}/${cfg}'"

    for run in $(seq 1 "$RUNS"); do
        echo "=== Piranha ${model_tag} | 2PC ${env_label} | batch ${batch} | run ${run}/${RUNS} ==="

        # Clear stale party processes from a prior run so their ports are free.
        for host in "${NODES[@]}"; do
            ssh "$host" "pkill -f 'piranha -p' 2>/dev/null; true"
        done

        echo "    Starting party 0 on ${NODES[0]} (listener; background)..."
        ssh "${NODES[0]}" "cd '${piranha_dir}' && env ${PIRANHA_RUN_ENV} ./piranha -p 0 -c '${cfg}' < /dev/null" \
            >> "$logf" 2>&1 &
        local p0=$!

        local listening=0
        for _ in $(seq 1 30); do
            if ssh "${NODES[0]}" "ss -tln 2>/dev/null | grep -q ':3200'"; then
                listening=1
                break
            fi
            sleep 2
        done
        if [[ "$listening" -ne 1 ]]; then
            echo "!! Party 0 never started listening for ${model_tag} (${env_label}); see ${logf}." >&2
            kill "$p0" 2>/dev/null
            exit 1
        fi

        echo "    Starting party 1 on ${NODES[1]} (foreground)..."
        if ! ssh "${NODES[1]}" "cd '${piranha_dir}' && env ${PIRANHA_RUN_ENV} ./piranha -p 1 -c '${cfg}' < /dev/null" \
            >> "$logf1" 2>&1; then
            echo "!! Party 1 FAILED for ${model_tag} (${env_label}); see ${logf1}." >&2
            kill "$p0" 2>/dev/null
            exit 1
        fi
        if ! wait "$p0"; then
            echo "!! Party 0 FAILED for ${model_tag} (${env_label}); see ${logf}." >&2
            exit 1
        fi
        sleep 5
    done
}

echo "==> Note: dataset files are intentionally absent; 'Error opening ...' log"
echo "    lines are expected and harmless (timing is input-value-independent)."

# LAN experiments with 2PC
run_piranha lan alexnet        files/models/alexnet-cifar10.json 192
run_piranha lan vgg16          files/models/vgg16-cifar10.json   192
run_piranha lan vgg16-imagenet files/models/vgg16-imagenet.json  8

# WAN experiments with 2PC
echo "==> Enabling WAN emulation (6 Gbit, 20 ms RTT) between ${NODES[0]} and ${NODES[1]}..."
wan_enabled=1
if ! set_wan on; then
    echo "!! Failed to enable WAN emulation; aborting before the WAN half." >&2
    exit 1
fi

run_piranha wan alexnet        files/models/alexnet-cifar10.json 192
run_piranha wan vgg16          files/models/vgg16-cifar10.json   192
run_piranha wan vgg16-imagenet files/models/vgg16-imagenet.json  8

disable_wan

echo
echo "==> Done. Piranha Table 3 logs are in ${log_dir}:"
ls -l "$log_dir"
