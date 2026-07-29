#!/bin/bash

script_dir=$(dirname "$0")
baseline_dir="$script_dir/../../baselines/orq/build"
run_script="../scripts/run_experiment.sh"

cd "$baseline_dir"

log_dir="../../../data/logs/fig6/orq"
mkdir -p "$log_dir"

# Grace period, in seconds, between SIGINT and SIGKILL on timeout. The per-run
# timeout itself is passed as the first argument to run_with_timeout on each
# experiment line (mirroring the explicit `--timeout` on the fig6-cdough.sh
# lines). Override GRACE_SEC by exporting it before invoking this script.
GRACE_SEC="${GRACE_SEC:-15}"

# run_with_timeout <timeout_sec> <command...>
#
# Runs the given command with a wall-clock limit of <timeout_sec> seconds. The
# command is launched in its own process group so that, on timeout, the whole
# local process tree (run_experiment.sh plus its mpirun/startmpc children) can
# be signaled. On timeout we send SIGINT first -- run_experiment.sh installs
# `trap "exit 1" SIGINT`, so it exits cleanly -- and escalate to SIGKILL if the
# group is still alive after GRACE_SEC. Returns the command's exit status, or
# 124 if it timed out. Redirections belong on the caller so stdout still flows
# to the per-experiment log file; status messages go to stderr.
run_with_timeout() {
    local timeout_sec=$1
    shift

    setsid "$@" &
    local cmd_pid=$!
    # setsid makes the child a process-group leader, so its PGID equals its PID.
    local pgid=$cmd_pid

    local elapsed=0
    while kill -0 "$cmd_pid" 2>/dev/null; do
        if (( elapsed >= timeout_sec )); then
            echo "[timeout] '$*' exceeded ${timeout_sec}s; sending SIGINT" >&2
            kill -INT -"$pgid" 2>/dev/null

            local waited=0
            while kill -0 "$cmd_pid" 2>/dev/null && (( waited < GRACE_SEC )); do
                sleep 1
                (( waited++ ))
            done

            if kill -0 "$cmd_pid" 2>/dev/null; then
                echo "[timeout] still alive after ${GRACE_SEC}s; sending SIGKILL" >&2
                kill -KILL -"$pgid" 2>/dev/null
            fi

            wait "$cmd_pid" 2>/dev/null
            return 124
        fi
        sleep 1
        (( elapsed++ ))
    done

    wait "$cmd_pid"
    return $?
}

# LAN Experiments with 2PC
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  aspirin >> "$log_dir/lan-2pc-aspirin.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  rcdiff >> "$log_dir/lan-2pc-rcdiff.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  pwd-reuse >> "$log_dir/lan-2pc-pwd-reuse.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  credit_score >> "$log_dir/lan-2pc-credit_score.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  comorbidity >> "$log_dir/lan-2pc-comorbidity.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  secrecy_q2 >> "$log_dir/lan-2pc-secrecy_q2.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  market-share >> "$log_dir/lan-2pc-market-share.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  custom_agg >> "$log_dir/lan-2pc-custom_agg.log"
run_with_timeout 1000 $run_script -p 2 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12 -m "-DTRIPLES=ZERO"  distinct_patients >> "$log_dir/lan-2pc-distinct_patients.log"


# LAN Experiments with 3PC
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  aspirin >> "$log_dir/lan-3pc-aspirin.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  rcdiff >> "$log_dir/lan-3pc-rcdiff.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  pwd-reuse >> "$log_dir/lan-3pc-pwd-reuse.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  credit_score >> "$log_dir/lan-3pc-credit_score.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  comorbidity >> "$log_dir/lan-3pc-comorbidity.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  secrecy_q2 >> "$log_dir/lan-3pc-secrecy_q2.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  market-share >> "$log_dir/lan-3pc-market-share.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  custom_agg >> "$log_dir/lan-3pc-custom_agg.log"
run_with_timeout 1000 $run_script -p 3 -f 1 -c nocopy -n 4 -s lan -T 16 -b -12  distinct_patients >> "$log_dir/lan-3pc-distinct_patients.log"

# WAN Experiments with 2PC
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  aspirin >> "$log_dir/wan-2pc-aspirin.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  rcdiff >> "$log_dir/wan-2pc-rcdiff.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  pwd-reuse >> "$log_dir/wan-2pc-pwd-reuse.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  credit_score >> "$log_dir/wan-2pc-credit_score.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  comorbidity >> "$log_dir/wan-2pc-comorbidity.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  secrecy_q2 >> "$log_dir/wan-2pc-secrecy_q2.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  market-share >> "$log_dir/wan-2pc-market-share.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  custom_agg >> "$log_dir/wan-2pc-custom_agg.log"
run_with_timeout 2000 $run_script -p 2 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1 -m "-DTRIPLES=ZERO"  distinct_patients >> "$log_dir/wan-2pc-distinct_patients.log"


# WAN Experiments with 3PC
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  aspirin >> "$log_dir/wan-3pc-aspirin.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  rcdiff >> "$log_dir/wan-3pc-rcdiff.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  pwd-reuse >> "$log_dir/wan-3pc-pwd-reuse.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  credit_score >> "$log_dir/wan-3pc-credit_score.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  comorbidity >> "$log_dir/wan-3pc-comorbidity.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  secrecy_q2 >> "$log_dir/wan-3pc-secrecy_q2.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  market-share >> "$log_dir/wan-3pc-market-share.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  custom_agg >> "$log_dir/wan-3pc-custom_agg.log"
run_with_timeout 2000 $run_script -p 3 -f 1 -c nocopy -n 4 -s wan -T 16 -b -1  distinct_patients >> "$log_dir/wan-3pc-distinct_patients.log"