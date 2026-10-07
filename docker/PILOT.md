# Pilot run: what each party runs

Three organizations each run one command on their own machine. **UMass holds the data
and is the only party that sees the results.** BU and the insurer see neither.

| Party | Rank | Address |
|---|---|---|
| BU | 0 | `wormhole.bu.edu` (128.197.11.60) |
| UMass | 1 | `<UMASS_IP>` |
| Insurer | 2 | `<INSURER_IP>` |

Fill in `<UMASS_IP>` and `<INSURER_IP>` everywhere below. Use the same values on every
machine.

## 1. Setup (every party, once)

You need a Linux machine with 16+ cores, `git`, `openssl`, and Docker
(`curl -fsSL https://get.docker.com | sh`).

```bash
git clone https://github.com/CASP-Systems-BU/CryptDough.git
cd CryptDough
git checkout <COMMIT>            # BU sends you this

# Build (about 15 minutes)
docker build -t cryptdough:pilot -f docker/Dockerfile .
docker run --rm -v cdough-pilot:/opt/cryptdough/build cryptdough:pilot bash -c \
  'cmake .. -DPROTOCOL=3 -DCOMM=NOCOPY -DTLS=ON -Wno-dev && make -j8 mpc-analysis'

# Make your key and certificate. YOUR_RANK is 0, 1 or 2, from the table above.
./docker/gen-party-certs.sh --party YOUR_RANK --out ~/pilot-tls
```

Email your `~/pilot-tls/partyYOUR_RANK.crt` to the other two parties, and read them the
printed fingerprint over the phone. Put the two certificates you receive into
`~/pilot-tls/`. **Never send the `.key` file to anyone.**

## 2. Before the run

**UMass:** put the data at `~/pilot-data/base_owner.csv`, then count its rows:

```bash
python3 docker/count-analysis-rows.py ~/pilot-data/base_owner.csv
```

Send the printed number to BU and the insurer. In the commands below it is `<ROWS>`.
That number is the only thing about the data that leaves UMass.

**Firewalls.** Allow these inbound TCP connections (nothing else is needed):

| Who | Ports | From |
|---|---|---|
| UMass | 31007–31013 | 128.197.11.60 (BU) |
| Insurer | 31014–31020 | 128.197.11.60 (BU) |
| Insurer | 31035–31041 | `<UMASS_IP>` |
| BU | none | |

## 3. Run (start in this order: Insurer, then UMass, then BU)

Run these from the `CryptDough` folder. Each party waits for the others, so start within
a few minutes of each other. The run could take a long time, so use `tmux` or `screen`.

**Insurer (rank 2):**

```bash
./docker/run-party-external.sh --rank 2 \
  --hosts wormhole.bu.edu,<UMASS_IP>,<INSURER_IP> --base-port 31000 \
  --tls-dir ~/pilot-tls --image cryptdough:pilot --build-volume cdough-pilot \
  -- ./mpc-analysis -S all -ow 1 -D /data -rs <ROWS> -t 1 -neng 7
```

**UMass (rank 1):**

```bash
./docker/run-party-external.sh --rank 1 \
  --hosts wormhole.bu.edu,<UMASS_IP>,<INSURER_IP> --base-port 31000 \
  --tls-dir ~/pilot-tls --data-dir ~/pilot-data \
  --image cryptdough:pilot --build-volume cdough-pilot \
  -- ./mpc-analysis -S all -ow 1 -D /data -rs <ROWS> -t 1 -neng 7 \
  | tee ~/pilot-results.txt
```

The results go to `~/pilot-results.txt`, on UMass only.

**BU (rank 0).** The party runs on `blinky.bu.edu`, which cannot be reached from outside.
All its traffic goes out through `wormhole.bu.edu`. Start the two relays first,
blinky's before wormhole's:

```bash
# On blinky: one local port per connection; only wormhole may reach the relay side.
for p in $(seq 31007 31020); do
  (setsid nohup bash -c "while :; do socat TCP4-LISTEN:$p,bind=127.0.0.1,reuseaddr \
    TCP4-LISTEN:$((p + 100)),reuseaddr,nodelay,range=128.197.11.60/32; sleep 0.2; done" \
    </dev/null >/dev/null 2>&1 &)
done

# On wormhole: dial blinky, then dial UMass (31007-31013) or the insurer (31014-31020).
for p in $(seq 31007 31020); do
  host=<UMASS_IP>; (( p >= 31014 )) && host=<INSURER_IP>
  (setsid nohup bash -c "while :; do $HOME/bin/socat \
    TCP4:blinky.bu.edu:$((p + 100)),forever,interval=0.5,nodelay \
    TCP4:$host:$p,retry=120,interval=1,nodelay; sleep 0.2; done" \
    </dev/null >/dev/null 2>&1 &)
done
```

Then, on blinky, start the party. Its host list differs from the others' on purpose: it
reaches UMass and the insurer through the local relay ports.

```bash
./docker/run-party-external.sh --rank 0 \
  --hosts blinky.bu.edu,127.0.0.1,127.0.0.1 --base-port 31000 \
  --tls-dir ~/pilot-tls --image cryptdough:pilot --build-volume cdough-pilot \
  -- ./mpc-analysis -S all -ow 1 -D /data -rs <ROWS> -t 1 -neng 7
```

When the run ends, stop the relays on both machines with
`pkill -f '[w]hile :; do .*socat'; pkill -x socat`.

## 4. Done, or not

- Success: each party prints `=== party N completed successfully ===`.
- If any party stops or crashes, **all three** must stop and start step 3 again.
- `Peer certificate is not pinned`: a certificate in `~/pilot-tls/` is wrong. Compare
  fingerprints by phone again.
- Everything hangs at the start: one party's command differs from the others' (an
  address, the port, or `<ROWS>`), or a firewall port is still closed.
