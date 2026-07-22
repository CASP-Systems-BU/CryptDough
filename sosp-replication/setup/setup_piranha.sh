#!/usr/bin/env bash
#
# install_piranha.sh
#
# Provisions two AWS g6.8xlarge instances (each with one NVIDIA L4 GPU),
# installs Piranha (https://github.com/ucbrise/piranha) on both, and runs a
# distributed 2-party (SecureML / TWOPC) test across the two instances.
#
#   ./install_piranha.sh [PREFIX] [AWS_REGION]
#
# Example:
#   ./install_piranha.sh piranha us-east-1
#
# The script will:
#   - Create a VPC (10.0.0.0/16), subnet, internet gateway, routing, and a
#     security group (SSH + full intra-VPC traffic for the MPC parties)
#   - Launch two g6.8xlarge instances from an AWS Deep Learning Base GPU AMI
#     (NVIDIA driver + CUDA toolkit pre-installed)
#   - Install Piranha's dependencies, build CUTLASS and Piranha on both nodes
#   - Download the MNIST dataset on both nodes
#   - Run a distributed 2-party SecureML test across the two instances
#   - Write per-node logs to ~/install_piranha.log on each instance
#
# Tear down afterwards with:  ./teardown_piranha.sh [PREFIX] [AWS_REGION]
#
# ============================================================================
set -euo pipefail

PREFIX="${1:-piranha}"
AWS_REGION="${2:-us-east-1}"
AWS_INSTANCE_TYPE="g6.8xlarge"
ROOT_VOLUME_GB=200
SSH_USER="ubuntu"
VPC_CIDR="10.0.0.0/16"
SUBNET_CIDR="10.0.1.0/24"

PIRANHA_REPO="https://github.com/ucbrise/piranha.git"
PIRANHA_FLOAT_PRECISION=26
PIRANHA_PROTOCOL_FLAG="-DTWOPC"

# State file to record resource IDs for cleanup.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/${PREFIX}-state.env"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

err() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

save_state() {
    local key="$1" value="$2"
    echo "${key}=${value}" >> "$STATE_FILE"
}

log "=== Checking prerequisites ==="

for tool in aws jq ssh scp; do
    if ! command -v "$tool" &>/dev/null; then
        err "'${tool}' is not installed. Please install it before running this script."
        exit 1
    fi
done
log "  CLI tools (aws, jq, ssh, scp) are available"

log "Checking AWS authentication..."
if ! aws sts get-caller-identity --region "$AWS_REGION" &>/dev/null; then
    err "AWS credentials not found or invalid."
    err "  Option 1: Run 'aws configure' with valid IAM credentials"
    err "  Option 2: Run 'aws sso login' if using AWS SSO"
    err "  Option 3: Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export AWS_DEFAULT_REGION="$AWS_REGION"
log "  AWS authenticated - Account: ${AWS_ACCOUNT_ID}, Region: ${AWS_REGION}"

# Initialize the state file
echo "# Resource state file for ${PREFIX}" > "$STATE_FILE"
echo "# Generated at $(date)" >> "$STATE_FILE"
echo "PREFIX=${PREFIX}" >> "$STATE_FILE"
echo "AWS_REGION=${AWS_REGION}" >> "$STATE_FILE"
log "  State file: ${STATE_FILE}"

echo ""
log "Configuration:"
log "  Prefix:        ${PREFIX}"
log "  Region:        ${AWS_REGION}"
log "  Instance type: ${AWS_INSTANCE_TYPE} (1x NVIDIA L4 GPU each, 2 instances)"
log "  Piranha build: PIRANHA_FLAGS=\"-DFLOAT_PRECISION=${PIRANHA_FLOAT_PRECISION} ${PIRANHA_PROTOCOL_FLAG}\""
echo ""

log "=========================================="
log "Phase 1: Creating AWS network infrastructure"
log "=========================================="

log "Creating VPC (${VPC_CIDR})..."
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block "$VPC_CIDR" \
    --query 'Vpc.VpcId' --output text)
aws ec2 create-tags --resources "$VPC_ID" \
    --tags Key=Name,Value="${PREFIX}-vpc"
aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-hostnames '{"Value": true}'
save_state "VPC_ID" "$VPC_ID"
log "  Created VPC: ${VPC_ID}"

log "Creating Internet Gateway..."
IGW_ID=$(aws ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID"
aws ec2 create-tags --resources "$IGW_ID" \
    --tags Key=Name,Value="${PREFIX}-igw"
save_state "IGW_ID" "$IGW_ID"
log "  Created and attached Internet Gateway: ${IGW_ID}"

log "Creating route table..."
RTB_ID=$(aws ec2 create-route-table \
    --vpc-id "$VPC_ID" \
    --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route \
    --route-table-id "$RTB_ID" \
    --destination-cidr-block "0.0.0.0/0" \
    --gateway-id "$IGW_ID" > /dev/null
aws ec2 create-tags --resources "$RTB_ID" \
    --tags Key=Name,Value="${PREFIX}-rtb"
save_state "RTB_ID" "$RTB_ID"
log "  Created route table: ${RTB_ID} (0.0.0.0/0 -> IGW)"

log "Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name "${PREFIX}-sg" \
    --description "Security group for Piranha GPU MPC experiments" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text)
# SSH from anywhere (management)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "0.0.0.0/0" > /dev/null
# ICMP from anywhere (connectivity checks)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol icmp --port -1 --cidr "0.0.0.0/0" > /dev/null
# All traffic within the VPC so the MPC parties can talk on any port
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol -1 --cidr "$VPC_CIDR" > /dev/null
aws ec2 create-tags --resources "$SG_ID" \
    --tags Key=Name,Value="${PREFIX}-sg"
save_state "SG_ID" "$SG_ID"
log "  Created security group: ${SG_ID}"

log "Creating SSH key pair..."
KEY_NAME="${PREFIX}-key"
KEY_FILE="${SCRIPT_DIR}/${KEY_NAME}.pem"
aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
chmod 400 "$KEY_FILE"
save_state "KEY_NAME" "$KEY_NAME"
save_state "KEY_FILE" "$KEY_FILE"
log "  Created key pair: ${KEY_NAME} (saved to ${KEY_FILE})"

echo ""

log "=========================================="
log "Phase 2: Launching g6.8xlarge instances"
log "=========================================="

# Look up the latest AWS Deep Learning Base GPU AMI (NVIDIA driver + CUDA
# toolkit pre-installed). Piranha's README recommends a Deep Learning Base AMI.
log "Looking up latest Deep Learning Base GPU AMI (Ubuntu 22.04)..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)
if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    err "Could not find a Deep Learning Base GPU AMI in ${AWS_REGION}."
    err "Set AMI_ID manually or try another region (e.g. us-east-1, us-west-2)."
    exit 1
fi
save_state "AMI_ID" "$AMI_ID"
log "  Using AMI: ${AMI_ID}"

# Launch both parties. Tag node0 (party 0) and node1 (party 1).
launch_instance() {
    local name="$1"
    aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$AWS_INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --subnet-id "$SUBNET_ID" \
        --security-group-ids "$SG_ID" \
        --associate-public-ip-address \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PREFIX}-${name}}]" \
        --query 'Instances[0].InstanceId' --output text
}

AZS=($(aws ec2 describe-availability-zones --region "$AWS_REGION" \
    --query 'AvailabilityZones[].ZoneName' --output text))
[ ${#AZS[@]} -gt 0 ] || AZS=("${AWS_REGION}a" "${AWS_REGION}b" "${AWS_REGION}c")
log "Candidate AZs: ${AZS[*]}"

launched=false
for AZ in "${AZS[@]}"; do
    log "Trying availability zone ${AZ}..."
    SUBNET_ID=$(aws ec2 create-subnet \
        --vpc-id "$VPC_ID" --cidr-block "$SUBNET_CIDR" \
        --availability-zone "$AZ" \
        --query 'Subnet.SubnetId' --output text 2>/dev/null) \
        || { log "  Could not create subnet in ${AZ}; skipping."; continue; }
    aws ec2 create-tags --resources "$SUBNET_ID" \
        --tags Key=Name,Value="${PREFIX}-subnet" >/dev/null 2>&1 || true
    aws ec2 associate-route-table --route-table-id "$RTB_ID" --subnet-id "$SUBNET_ID" > /dev/null

    set +e
    NODE0_ID=$(launch_instance "node0"); r0=$?
    NODE1_ID=$(launch_instance "node1"); r1=$?
    set -e

    if [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] \
       && [ -n "$NODE0_ID" ] && [ "$NODE0_ID" != None ] \
       && [ -n "$NODE1_ID" ] && [ "$NODE1_ID" != None ]; then
        save_state "SUBNET_ID" "$SUBNET_ID"
        save_state "NODE0_ID" "$NODE0_ID"
        save_state "NODE1_ID" "$NODE1_ID"
        log "  Launched node0: ${NODE0_ID}"
        log "  Launched node1: ${NODE1_ID}"
        log "  Using AZ ${AZ}, subnet ${SUBNET_ID}"
        launched=true
        break
    fi

    log "  ${AZ} lacks g6.8xlarge capacity; cleaning up and trying the next AZ..."
    for _id in "$NODE0_ID" "$NODE1_ID"; do
        case "$_id" in i-*) aws ec2 terminate-instances --instance-ids "$_id" >/dev/null 2>&1 || true ;; esac
    done
    for _id in "$NODE0_ID" "$NODE1_ID"; do
        case "$_id" in i-*) aws ec2 wait instance-terminated --instance-ids "$_id" 2>/dev/null || true ;; esac
    done
    aws ec2 delete-subnet --subnet-id "$SUBNET_ID" >/dev/null 2>&1 || true
    NODE0_ID=""; NODE1_ID=""
done

if [ "$launched" != true ]; then
    err "Could not launch two ${AWS_INSTANCE_TYPE} instances in any AZ of ${AWS_REGION} (insufficient capacity)."
    err "Try again later or in a different region."
    exit 1
fi

log "Waiting for instances to reach 'running'..."
aws ec2 wait instance-running --instance-ids "$NODE0_ID" "$NODE1_ID"
log "  Instances are running"

log "Waiting for instance status checks to pass (this can take a few minutes)..."
aws ec2 wait instance-status-ok --instance-ids "$NODE0_ID" "$NODE1_ID"
log "  Status checks passed"

# Gather IPs
get_ip() {
    local id="$1" field="$2"
    aws ec2 describe-instances --instance-ids "$id" \
        --query "Reservations[0].Instances[0].${field}" --output text
}
NODE0_PUBLIC_IP=$(get_ip "$NODE0_ID" PublicIpAddress)
NODE0_PRIVATE_IP=$(get_ip "$NODE0_ID" PrivateIpAddress)
NODE1_PUBLIC_IP=$(get_ip "$NODE1_ID" PublicIpAddress)
NODE1_PRIVATE_IP=$(get_ip "$NODE1_ID" PrivateIpAddress)
save_state "NODE0_PUBLIC_IP" "$NODE0_PUBLIC_IP"
save_state "NODE0_PRIVATE_IP" "$NODE0_PRIVATE_IP"
save_state "NODE1_PUBLIC_IP" "$NODE1_PUBLIC_IP"
save_state "NODE1_PRIVATE_IP" "$NODE1_PRIVATE_IP"
log "  node0 (party 0) - Public: ${NODE0_PUBLIC_IP}, Private: ${NODE0_PRIVATE_IP}"
log "  node1 (party 1) - Public: ${NODE1_PUBLIC_IP}, Private: ${NODE1_PRIVATE_IP}"

echo ""

log "=========================================="
log "Phase 3: Waiting for SSH on both instances"
log "=========================================="

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -i "$KEY_FILE")

# node0() / node1() run a command on the respective instance.
node0() { ssh "${SSH_OPTS[@]}" "${SSH_USER}@${NODE0_PUBLIC_IP}" "$@"; }
node1() { ssh "${SSH_OPTS[@]}" "${SSH_USER}@${NODE1_PUBLIC_IP}" "$@"; }

wait_for_ssh() {
    local label="$1" ip="$2"
    log "  Waiting for SSH on ${label} (${ip})..."
    for _ in $(seq 1 40); do
        if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "true" 2>/dev/null; then
            log "  SSH ready on ${label}"
            return 0
        fi
        sleep 10
    done
    err "SSH never became ready on ${label} (${ip})"
    exit 1
}
wait_for_ssh "node0" "$NODE0_PUBLIC_IP"
wait_for_ssh "node1" "$NODE1_PUBLIC_IP"

echo ""

log "=========================================="
log "Phase 4: Installing & building Piranha on both nodes"
log "=========================================="

INSTALLER_LOCAL="$(mktemp "${TMPDIR:-/tmp}/piranha_installer.XXXXXX")"
cat > "$INSTALLER_LOCAL" <<INSTEOF
#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/cuda/bin:\$PATH

echo "===== Piranha install started at \$(date) ====="

# --- Detect GPU compute capability (e.g. L4 -> 8.9 -> arch 89) ---
GPU_ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
echo "Detected GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (arch sm_\${GPU_ARCH})"
NVCC_PATH=\$(command -v nvcc)
echo "Using nvcc: \${NVCC_PATH} (\$(\${NVCC_PATH} --version | grep release || true))"

# --- System dependencies (README: libgtest-dev, libssl-dev) ---
echo "----- Installing apt dependencies -----"
sudo apt-get update -yq
sudo apt-get install -yq git build-essential cmake libgtest-dev libssl-dev python3-pip

# --- Clone Piranha ---
echo "----- Cloning Piranha -----"
cd ~
rm -rf ~/piranha
git clone ${PIRANHA_REPO} ~/piranha
cd ~/piranha
git submodule update --init --recursive

# --- Build CUTLASS for the detected architecture ---
echo "----- Building CUTLASS (arch sm_\${GPU_ARCH}) — this can take a while -----"
cd ~/piranha/ext/cutlass
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=\${GPU_ARCH} \
    -DCUTLASS_ENABLE_PROFILER=OFF \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCMAKE_CUDA_COMPILER_WORKS=1 \
    -DCMAKE_CUDA_COMPILER=\${NVCC_PATH}
make -j\$(nproc)

echo "----- Building gtest -----"
if [ -d /usr/src/googletest ]; then GTEST_SRC=/usr/src/googletest; else GTEST_SRC=/usr/src/gtest; fi
sudo cmake -S "\$GTEST_SRC" -B "\$GTEST_SRC/build"
sudo cmake --build "\$GTEST_SRC/build" -j\$(nproc)
sudo find "\$GTEST_SRC/build" -name 'libgtest*.a' -exec cp -v {} /usr/lib/ \;

# --- Required directories ---
echo "----- Creating output/data directories -----"
cd ~/piranha
mkdir -p output files/MNIST files/CIFAR10

# --- Download MNIST (needed by the SecureML test) ---
echo "----- Downloading MNIST dataset -----"
python3 -m pip install --quiet --user torch torchvision
cd ~/piranha/scripts
python3 download_mnist.py

# --- Install Piranha's required CUDA toolchain (CUDA 11.8 + gcc-10) ---
# Piranha is CUDA-11-era code: its Makefile pins CUDA 11.5 and its Thrust usage
# does not compile against the CCCL Thrust shipped in CUDA 12.4+/13. The Deep
# Learning AMI now ships only CUDA 12.8-13.2, so install CUDA 11.8 (the oldest
# CUDA that still supports the L4's sm_89) side-by-side. gcc-10 is also needed:
# CUDA 11.8's nvcc miscompiles GCC 11's libstdc++ <ratio>, so it is used as the
# nvcc host compiler. (Piranha links CUTLASS headers only — no libcutlass.a — so
# the CUTLASS build above is not actually required by Piranha itself.)
echo "----- Installing CUDA 11.8 toolkit + gcc-10 -----"
if [ ! -x /usr/local/cuda-11.8/bin/nvcc ]; then
    wget -q -O /tmp/cuda_11.8.run \
        https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh /tmp/cuda_11.8.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.8 --override
fi
sudo apt-get install -yq gcc-10 g++-10

# --- Build Piranha (2-party protocol) with the CUDA 11.8 + gcc-10 toolchain ---
echo "----- Building Piranha -----"
cd ~/piranha
export PATH=/usr/local/cuda-11.8/bin:\$PATH
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-10'
make -j\$(nproc) CUDA_VERSION=11.8 PIRANHA_FLAGS="-DFLOAT_PRECISION=${PIRANHA_FLOAT_PRECISION} ${PIRANHA_PROTOCOL_FLAG}"

test -x ~/piranha/piranha
echo "===== PIRANHA_INSTALL_DONE at \$(date) ====="
INSTEOF

# Push the installer to both nodes and start it in the background, logging to
# ~/install_piranha.log. We poll the log for the completion sentinel so a long
# build doesn't depend on a single held-open SSH connection.
log "Copying installer to both nodes..."
scp "${SSH_OPTS[@]}" "$INSTALLER_LOCAL" "${SSH_USER}@${NODE0_PUBLIC_IP}:/tmp/piranha_installer.sh"
scp "${SSH_OPTS[@]}" "$INSTALLER_LOCAL" "${SSH_USER}@${NODE1_PUBLIC_IP}:/tmp/piranha_installer.sh"

log "Starting build on node0 and node1 (logging to ~/install_piranha.log)..."
node0 "chmod +x /tmp/piranha_installer.sh; nohup bash /tmp/piranha_installer.sh > ~/install_piranha.log 2>&1 & echo started"
node1 "chmod +x /tmp/piranha_installer.sh; nohup bash /tmp/piranha_installer.sh > ~/install_piranha.log 2>&1 & echo started"

# Poll each node's log until the success or failure sentinel appears.
wait_for_install() {
    local label="$1"
    local runner="$2"
    local timeout_secs=1800   # 30 min
    local elapsed=0 interval=30
    log "  Waiting for build to finish on ${label} (timeout: ${timeout_secs}s)..."
    while true; do
        if $runner "grep -q PIRANHA_INSTALL_DONE ~/install_piranha.log" 2>/dev/null; then
            log "  ${label}: build complete"
            return 0
        fi
        if ! $runner "pgrep -f piranha_installer.sh >/dev/null" 2>/dev/null; then
            if ! $runner "grep -q PIRANHA_INSTALL_DONE ~/install_piranha.log" 2>/dev/null; then
                err "${label}: installer exited without success. Last log lines:"
                $runner "tail -n 30 ~/install_piranha.log" 2>/dev/null || true
                return 1
            fi
        fi
        if [ "$elapsed" -ge "$timeout_secs" ]; then
            err "${label}: timed out waiting for build. Last log lines:"
            $runner "tail -n 30 ~/install_piranha.log" 2>/dev/null || true
            return 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
}

wait_for_install "node0" node0
wait_for_install "node1" node1

echo ""

log "=========================================="
log "Phase 5: Running distributed 2-party Piranha test"
log "=========================================="

log "Generating distributed config from the localhost sample..."
node0 "cd ~/piranha && python3 - '${NODE0_PRIVATE_IP}' '${NODE1_PRIVATE_IP}' <<'PYEOF'
import json, sys
cfg = json.load(open('files/samples/localhost_config.json'))
cfg['num_parties'] = 2
cfg['party_ips']   = [sys.argv[1], sys.argv[2]]
cfg['party_users'] = ['${SSH_USER}', '${SSH_USER}']
cfg['run_name']    = 'piranha_dist_test'
# Minimal fast smoke test: 1 epoch, 1 training iteration.
cfg['lr_schedule']            = [3]
cfg['custom_epochs']          = True
cfg['custom_epoch_count']     = 1
cfg['custom_iterations']      = True
cfg['custom_iteration_count'] = 1
cfg['no_test']                = True
cfg['last_test']              = False
json.dump(cfg, open('files/samples/dist_config.json', 'w'), indent=2)
print('wrote files/samples/dist_config.json')
PYEOF"

# Copy the generated config to node1 so both parties use identical settings.
log "Distributing config to node1..."
node0 "cat ~/piranha/files/samples/dist_config.json" | \
    node1 "cat > ~/piranha/files/samples/dist_config.json"

# The piranha binary is linked against the side-by-side CUDA 11.8 runtime, which
# is not on the default loader path, so every invocation needs LD_LIBRARY_PATH.
PIRANHA_RUN_ENV="LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 CUDA_VISIBLE_DEVICES=0"

# Start party 0 (the listener) first, in the background, and wait until it is
# actually listening before starting party 1 (the connector). Otherwise party 1
# can exhaust its connection-retry budget while party 0 is still initializing
# CUDA, and the two parties never connect.
log "Starting party 0 on node0 (listener; detached)..."
node0 "echo '===== Piranha 2PC run (party 0) started at '\$(date)' =====' >> ~/install_piranha.log"
ssh "${SSH_OPTS[@]}" -f -n "${SSH_USER}@${NODE0_PUBLIC_IP}" \
    "cd ~/piranha && env ${PIRANHA_RUN_ENV} ./piranha -p 0 -c files/samples/dist_config.json < /dev/null >> ~/install_piranha.log 2>&1"

log "Waiting for party 0 to begin listening..."
for _ in $(seq 1 30); do
    if node0 "ss -tln 2>/dev/null | grep -q ':3200'"; then
        log "  party 0 is listening"
        break
    fi
    sleep 2
done

log "Starting party 1 on node1 (connector; foreground, streaming output)..."
set +e
node1 "cd ~/piranha && echo '===== Piranha 2PC run (party 1) started at '\$(date)' =====' | tee -a ~/install_piranha.log && \
    env ${PIRANHA_RUN_ENV} ./piranha -p 1 -c files/samples/dist_config.json 2>&1 | tee -a ~/install_piranha.log"
RUN_STATUS=$?
set -e

echo ""
if [ "$RUN_STATUS" -eq 0 ]; then
    log "  Distributed 2-party Piranha test completed successfully."
else
    log "  NOTE: party 1 exited with status ${RUN_STATUS}. Check the node logs:"
    log "    node0: ssh -i ${KEY_FILE} ${SSH_USER}@${NODE0_PUBLIC_IP} 'tail -n 40 ~/install_piranha.log'"
    log "    node1: ssh -i ${KEY_FILE} ${SSH_USER}@${NODE1_PUBLIC_IP} 'tail -n 40 ~/install_piranha.log'"
fi

rm -f "$INSTALLER_LOCAL"
echo ""

log "=========================================="
log "Phase 6: Summary"
log "=========================================="

cat <<SUMMARY

============================================================
  PIRANHA TWO-INSTANCE SETUP COMPLETE
============================================================

--- Instances ---
  node0 (party 0): ${NODE0_PUBLIC_IP} (private: ${NODE0_PRIVATE_IP})
    ssh -i ${KEY_FILE} ${SSH_USER}@${NODE0_PUBLIC_IP}

  node1 (party 1): ${NODE1_PUBLIC_IP} (private: ${NODE1_PRIVATE_IP})
    ssh -i ${KEY_FILE} ${SSH_USER}@${NODE1_PUBLIC_IP}

--- Logs ---
  Per-node install/build/run log: ~/install_piranha.log on each instance.
    ssh -i ${KEY_FILE} ${SSH_USER}@${NODE0_PUBLIC_IP} 'tail -f ~/install_piranha.log'

--- Re-running the distributed test manually ---
  On node1:  cd ~/piranha && CUDA_VISIBLE_DEVICES=0 ./piranha -p 1 -c files/samples/dist_config.json
  On node0:  cd ~/piranha && CUDA_VISIBLE_DEVICES=0 ./piranha -p 0 -c files/samples/dist_config.json

--- Teardown ---
  State file: ${STATE_FILE}
  ./teardown_piranha.sh ${PREFIX} ${AWS_REGION}
  (or terminate instances ${NODE0_ID}, ${NODE1_ID} and delete VPC ${VPC_ID})

============================================================
SUMMARY

log "Done. State saved to: ${STATE_FILE}"
