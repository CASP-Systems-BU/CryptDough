# Running CryptDough across three organizations

This is the runbook for a real multi-party computation where the parties are separate
organizations that do not trust one another. It assumes no party will give another SSH
access, run a binary another party compiled, or open its network more than necessary.

For single-organization benchmarking on a cluster you control, use
[README.md](README.md) instead — it is simpler and faster.

- [What each party needs](#what-each-party-needs)
- [Runbook](#runbook)
- [Ports](#ports)
- [Private data](#private-data)
- [Troubleshooting](#troubleshooting)

## What each party needs

| | |
|---|---|
| A Linux machine with Docker | `curl -fsSL https://get.docker.com \| sh` |
| The agreed run manifest | one file, identical for everyone |
| Its own TLS keypair | generated locally; the private key never leaves |
| The other parties' certificates | public; exchanged out of band |
| Its own private data | mounted read-only; never leaves the machine |
| Some open ports | **rank 0 needs none** — see [Ports](#ports) |

Nobody needs SSH access to anybody. Each organization runs one command locally.

## Runbook

### 0. Agree the manifest (all parties, once)

Copy [`manifest.example.yaml`](manifest.example.yaml), fill in the parties, hosts, git
commit and base port, and distribute it. This is the single source of truth. It matters
because the port each connection uses is

```
base_port + num_parties * threads * i + threads * j
```

so if the parties disagree about `threads`, the port arithmetic diverges and nothing
connects — with no error that explains why. The same holds for every compile-time
option: a mismatch hangs, or silently computes the wrong answer.

Assign ranks deliberately: **rank 0 needs no inbound ports at all**, so give it to the
most firewalled organization. Rank 2 is the most exposed.

### 1. Generate your identity (each party, once)

```bash
./docker/gen-party-certs.sh --party <YOUR_RANK> --out /secure/tls
```

Send `party<RANK>.crt` to the other parties. **Read the printed SHA-256 fingerprint to
them over a channel an attacker cannot also control** — a phone call, in person, or a
signed message. Then collect their certificates into the same directory, so it holds
your key plus everyone's certificates:

```
/secure/tls/party0.crt  party1.crt  party2.crt  party<YOUR_RANK>.key
```

### 2. Build your own image (each party)

Every organization builds from the same commit. No party runs a binary another party
compiled.

```bash
git checkout <commit from the manifest>
./docker/build-party.sh --manifest run.yaml --src . --tag cryptdough:tls
```

Compare the printed configuration fingerprint with the other parties. If they differ,
stop: you would be running different protocols against each other.

Images are deliberately **not** bit-reproducible — `CMakeLists.txt` builds with
`-march=native`, so each party's binary is tuned to its own CPU. Verification is at the
source-and-configuration level, which is what the fingerprint covers.

### 3. Pre-flight check (each party)

```bash
./docker/check-manifest.sh --manifest run.yaml --src . --certs /secure/tls
```

This compares your CMake cache against the manifest, confirms the peer certificates are
present, and prints the exact inbound port ranges you must open. Do not launch until it
passes.

### 4. Compile the program (each party)

```bash
docker run --rm -v cdough-run:/opt/cryptdough/build cryptdough:tls bash -c \
  'cmake .. -DPROTOCOL=3 -DCOMM=NOCOPY -DCOMM_THREADS=4 -DTLS=ON -Wno-dev && make -j micro_primitives'
```

Use the flags from the manifest.

### 5. Launch — **in descending rank order**

Party *i* connects to every *j > i* and listens for every *j < i*. Listeners block in
`accept()` indefinitely; only the connecting side can time out. So the highest rank
starts first and rank 0 starts last. Higher ranks may start minutes early at no cost.

```bash
# Party 2 first:
./docker/run-party-external.sh --rank 2 \
    --hosts mpc-a.example.edu,mpc-b.example.org,mpc-c.example.com \
    --base-port 20000 --tls-dir /secure/tls --data-dir /srv/private \
    --image cryptdough:tls --build-volume cdough-run \
    -- ./micro_primitives -t 8 -b -12 -r 1048576 -s lan

# then party 1, then party 0, with the same --hosts and --base-port.
```

`--hosts` is in **rank order** and must be byte-identical everywhere. Real DNS names or
IPs work directly; there is no `/etc/hosts` convention to satisfy.

The launcher retries if peers are not up yet (`--attempts`, `--retries`), so exact
simultaneity is not required.

## Ports

For `H` parties and `-t T` worker threads, the pair (i→j) uses ports
`base_port + H*T*i + T*j`, plus `100 * engine_index` when `-neng > 1`.

For three parties:

| Party | Must accept inbound | Must reach outbound |
|---|---|---|
| **0** | **nothing** | parties 1 and 2 |
| 1 | `[base+T, base+2T)` from party 0 | party 2 |
| 2 | `[base+2T, base+3T)` from 0, `[base+5T, base+6T)` from 1 | — |

`check-manifest.sh` prints the exact numbers, so you can give your network team a
minimal request rather than "open 10000-51000".

This was verified empirically: during a three-machine run with `base_port=26000` and
`-t 1`, rank 0 opened **no listening sockets at all**, rank 1 listened on 26001, and
rank 2 on 26002 and 26005 — exactly what the formula predicts.

## Private data

Each organization mounts only its own data, read-only, at `/data`:

```bash
--data-dir /srv/org-a/private        # becomes /data:ro in the container
```

`inputCSVTableData(path, input_party)`
(`include/core/containers/tabular/encoded_table.h`) has **only** the designated party
open the file; every other party allocates zero-filled columns and receives secret
shares. The plaintext never crosses the network.

See [`examples/ex6_three_party_private_input.cpp`](../examples/ex6_three_party_private_input.cpp)
for a worked three-party program. The schema is public and must be identical on every
party — keep it in the manifest. Note that columns parse as integers, so values where
leading zeros matter (ZIP codes, identifiers) must be encoded numerically beforehand.

Open only the aggregate the parties agreed to learn. Opening a table reveals it to
everyone.

## Troubleshooting

**`Peer certificate is not pinned`.** The peer presented a certificate whose SHA-256 is
not in your pinned set. Either you have the wrong `party<N>.crt`, or someone is
impersonating a party. The connection is refused rather than falling back — that is
intended. Compare fingerprints out of band before assuming it is a configuration mistake.

**`Could not load certificate ... Permission denied`.** The container could not read
your mounted TLS directory. The entrypoint stages the material as the container user,
so this normally means the path given to `--tls-dir` is wrong.

**Everything hangs at startup with no error.** Almost always a manifest mismatch —
usually `-t`, because the port arithmetic depends on it. Run `check-manifest.sh` on
every party and compare the configuration fingerprints from `build-party.sh`.

**`Failed to connect to server` on rank 0.** The higher-ranked parties are not
listening yet. Start in descending rank order, and raise `--retries`.

**TLS is on but performance dropped.** Expected: record encryption is on the critical
path. Never compare TLS and non-TLS numbers in the same table; re-baseline.

**A party restarts mid-run.** The computation does not recover; all parties must
restart. There is no checkpointing.
