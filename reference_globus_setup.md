---
name: Globus setup for Fir↔cs-theia-gpu01 transfers
description: How to use Globus to move data between Compute Canada (Fir/Cedar) and the SFU lab — endpoint IDs, daemon, auth quirks, and the pty helpers that let me drive the interactive logins from chat
type: reference
originSessionId: d8e49144-a8ff-4d16-bb15-a3dd8e7354a5
---
Globus is set up on cs-theia-gpu01 for moving data **both directions** between DRAC clusters (Fir, Cedar) and the SFU lab. Use this when SSH/scp to the cluster is unavailable, or for very large transfers (Globus is much faster and resumable). Verified working 2026-05-01 (~18 MB/s pulling 7.3 GB of checkpoints from Fir).

## Endpoints

- **DRAC source/destination** (GCS5 mapped collection): `8dec4129-9ab4-451d-a45f-5b4b8471f7a3` — display name *"computecanada#cedar-globus & alliancecan#fir-globus"*. Paths on this endpoint look like `/home/zla247/...`, `/scratch/zla247/...`, `/project/.../zla247/...` — they mirror the cluster filesystem layout. The same collection covers both Fir and Cedar.
- **Local GCP on cs-theia-gpu01**: `c0822582-4501-11f1-ad4e-0afffe4617ab` (name: `cs-theia-gpu01-zla247`). Exposes `~/` and `/media/raid/` rw via `~/.globusonline/lta/config-paths`. Add lines (`<path>,0,1`) for more paths.

## Tooling on cs-theia-gpu01

- **globus-cli**: `~/.local/bin/globus` — installed via `pip install --user globus-cli`. Tokens persist in `~/.globus/`.
- **Globus Connect Personal**: `~/globus/globusconnectpersonal-3.2.8/`. Config lives in `~/.globusonline/`.
- **GCP daemon**: must be running for transfers to/from the local endpoint.
  - Start: `~/globus/globusconnectpersonal-3.2.8/globusconnectpersonal -start &` (nohup-style; does NOT auto-start on reboot)
  - Status: `~/globus/globusconnectpersonal-3.2.8/globusconnectpersonal -status` (expect `Globus Online: connected`)
  - Stop: `~/globus/globusconnectpersonal-3.2.8/globusconnectpersonal -stop`
- **pty helpers** (so I can drive interactive auth flows across chat turns — Globus prints a URL and reads the code from stdin):
  - `~/globus/pty_helper.py <tag> -- <cmd> [args...]` — generic. Writes URL to `/tmp/<tag>_url.txt`, reads code from `/tmp/<tag>_code.txt`, marks done at `/tmp/<tag>_done.txt`. Use this for all interactive Globus auth.
  - `~/globus/globus_login_helper.py` and `~/globus/gcp_setup_helper.py` — older hardcoded variants kept for reference.

## Auth quirks — the gotcha

Accessing DRAC GCS5 collections requires THREE separate auth flows the first time. Each one prints a URL and waits for an auth code on stdin — drive them through `pty_helper.py`:

1. `globus login --no-local-server` — initial CLI auth.
2. `globus session consent --no-local-server 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/<COLLECTION_ID>/data_access]'` — grants `data_access` for the specific collection. Required per collection.
3. `globus session update --no-local-server computecanada.ca` — forces session reauth via the `@computecanada.ca` identity (NOT `@sfu.ca`). DRAC enforces this domain check.

Symptoms when each is missing:
- Step 1 missing → `MissingLoginError: Missing login for Globus Auth.`
- Step 2 missing → `The collection you are trying to access data on requires you to grant consent...`
- Step 3 missing → `Session reauthentication required (Globus Transfer)` / `globus session update computecanada.ca`

After the first time, tokens persist for ~6 months. Re-auth only needed for new collections, expired tokens, or different DRAC sites.

## Transfer commands

**Pull from DRAC → lab** (download, e.g., checkpoints):
```bash
~/.local/bin/globus transfer --recursive \
  '8dec4129-9ab4-451d-a45f-5b4b8471f7a3:/home/zla247/scratch/SOMETHING/' \
  'c0822582-4501-11f1-ad4e-0afffe4617ab:/media/raid/cloth/SOMETHING/'
```

**Push from lab → DRAC** (upload, e.g., dataset, source code):
```bash
~/.local/bin/globus transfer --recursive \
  'c0822582-4501-11f1-ad4e-0afffe4617ab:/media/raid/cloth/SOMETHING/' \
  '8dec4129-9ab4-451d-a45f-5b4b8471f7a3:/home/zla247/projects/zla247/SOMETHING/'
```

Useful flags: `--label "..."` (for tracking), `--notify off` (suppress email), `--sync-level mtime` (rsync-like skip-if-unchanged on re-run), `--preserve-mtime`, `--encrypt-data`.

## Listing / monitoring / canceling

- List a remote dir: `globus ls 8dec4129-...:/home/zla247/`
- List recursively: `globus ls --recursive 8dec4129-...:/path/`
- Show task status: `globus task show <TASK_ID>` (look for `Status: SUCCEEDED|ACTIVE|FAILED`, `Bytes Transferred`, `Bytes Per Second`)
- Wait for a task to finish: `globus task wait <TASK_ID>`
- Recent tasks: `globus task list`
- Cancel: `globus task cancel <TASK_ID>`
- Quick auth check: `globus whoami` and `globus session show`

## How to apply

For a new transfer, work top-to-bottom:
1. Confirm GCP daemon is running (`...globusconnectpersonal -status`); start it if not.
2. If targeting a *new* DRAC collection (different UUID), run step 2 of the auth flow for it.
3. If you get a session reauth error, run step 3 — but only when the error explicitly says so; don't re-run prophylactically.
4. Submit `globus transfer --recursive ...`. Save the task ID; poll with `globus task show`.

If the transfer is large (>100 GB) or might run overnight, attach `--label` so it shows up usefully in `globus task list`. Globus retries automatically on transient failures and resumes from the last completed file on restart.
