#!/usr/bin/env bash
# Sets up a GCP CVM and runs TPCH experiments.

set -e

# Check if signed in to Google Cloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
  echo "Not signed in to Google Cloud. Authenticating..."
  gcloud auth login

  if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    exit
  fi

  echo "Sign in OK!"
fi

# Adjust instance name + number of cores here.
INSTANCE_NAME="cdough-cvm-$$"
NUM_CORES=8

# The queries from Figure 12: fast, median, slow
QUERIES="q11 q22 q12 q8 q21"
SCALE_FACTOR=1

# Default: regular (if arg is empty)
# Other options: 
# - SEV
# - SEV_SNP
CONF_COMP_TYPE=$1

echo -n "Confidential Computing Type: "
if [ -n "$CONF_COMP_TYPE" ]; then
  echo "$CONF_COMP_TYPE"
else
  echo "None"
fi

# n2d / Milan / central1 are required for snp.
# See the documentation:
#  - https://docs.cloud.google.com/compute/docs/general-purpose-machines#n2d_machine_types
#  - https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/supported-configurations#amd-sev-snp
MACHINE_TYPE=n2d-standard-$NUM_CORES
CPU_PLATFORM="AMD Milan"
ZONE="us-central1-a"
IMAGE="ubuntu-minimal-2404-lts-amd64"

# Comment out the last line to disable SNP
gcloud compute instances create $INSTANCE_NAME \
    --machine-type=$MACHINE_TYPE \
    --min-cpu-platform="AMD Milan" \
    --zone=$ZONE \
    --image-family=$IMAGE \
    --maintenance-policy=TERMINATE \
    --image-project=ubuntu-os-cloud \
    $([ -n "$CONF_COMP_TYPE" ] && echo "--confidential-compute-type=$CONF_COMP_TYPE")


echo "[ Instance created. Waiting to spin up. ]"

while ! (echo : | gcloud compute ssh $INSTANCE_NAME --zone=$ZONE > /dev/null); do
    echo "... try again"
    sleep 5
done

echo "[ Setting up CryptDough! This may take a while! ]"

OUTFILE=cvm-output.$$.txt

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -A <<HERE | tee $OUTFILE
    set -e
    sudo apt update
    sudo apt -y install git cmake pkg-config build-essential libsodium23 \
        libsodium-dev libntl-dev libgmp-dev libsqlite3-dev libopenmpi3 \
        libopenmpi-dev openmpi-bin openmpi-common

    mkdir -p ~/.ssh
    ssh-keyscan github.com >> ~/.ssh/known_hosts
    
    # '|| true' prevents failure if the repo already exists
    # relies on SSH forwarding (-A) from the host
    git clone git@github.com:CASP-Systems-BU/secrecy-private.git || true
    cd secrecy-private/build

    # Needed for CLI
    ../scripts/_setup_cryptotools.sh

    # TODO: enable confidential mode with a flag
    cmake .. -DPROTOCOL=1 -DSINGLE=1

    make -j cvm-queries
    for query in $QUERIES; do
      echo "[CryptDough] Running \$query:"
      ./\$query -r $SCALE_FACTOR -t 4
    done

    echo "This is \$(hostname)"

    # double check SNP working!
    echo "[ SNP status ]"
    if ! (sudo dmesg | grep -i "sev\|snp"); then
        echo "...not enabled"
    fi

    # Check load averages: should be close to # of threads
    uptime
HERE

echo "[CryptDough] CVM Experiment completed. Results summary:"

grep -E "Overall|^\[CryptDough\] Running" $OUTFILE

echo "      See $OUTFILE for full details. Deleting instance."

gcloud compute instances delete --quiet --zone=$ZONE $INSTANCE_NAME