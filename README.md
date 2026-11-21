# JH Super Resolution — Modal backend

GPU super-resolution backend for the JHenTai remote-SR feature. One
scale-to-zero Modal container per account serves one fixed model
(RealESRGAN_x4plus_anime_6B, native 4x; 2x/3x are Lanczos downscales of
the 4x output). A Cloudflare Worker scheduler (single file, deployed
separately via the Cloudflare dashboard) wakes the account with the
most monthly credit left, mints a session here, and hands the client
this backend's Cloudflare Tunnel address. Clients only ever talk to
the tunnel hostname; the modal.run URL exposes nothing but the
scheduler's secret-gated door.

## One-time setup per Modal account

1. Install the Modal CLI and sign in to the account:
   `pip install modal && modal setup`
2. Create the secret (values: pick a long random `BACKEND_SECRET` —
   the same value goes into the scheduler; `TUNNEL_HOST` is this
   account's public hostname without scheme; `CF_TUNNEL_TOKEN` comes
   from step 3):

   ```
   modal secret create jh-sr-secrets \
     BACKEND_SECRET=<long random> \
     TUNNEL_HOST=ehsr-1.fur.boats \
     CF_TUNNEL_TOKEN=<tunnel token>
   ```

3. In the Cloudflare Zero Trust dashboard create a tunnel, add the
   public hostname (`ehsr-1.fur.boats`) routed to
   `http://localhost:8080`, and copy the tunnel token into the secret
   above. (Create the tunnel before or after deploying — the backend
   starts cloudflared with whatever token the secret holds at boot.)
4. Deploy: `modal deploy app.py`
5. Note the app's URL (`https://<workspace>--jh-super-resolution.modal.run`)
   — that is the account's `wakeBase` in the scheduler's `ACCOUNTS`.

## Scheduler configuration (per account)

```json
{"id": "acc1",
 "tunnelBase": "https://ehsr-1.fur.boats",
 "wakeBase": "https://<workspace>--jh-super-resolution.modal.run",
 "monthlyCreditUsd": 30,
 "gpuRatePerSec": 0.000164}
```

`gpuRatePerSec`: T4 = 0.000164, L4 = 0.000222, A10 = 0.000306. The GPU
type itself is `GPU_TYPE` at the top of `app.py`.

## HTTP surface

Scheduler door (modal.run host only, header `X-Backend-Secret`):

| Route | Purpose |
|---|---|
| `POST /__wake` | The request itself wakes the container; also (re)starts cloudflared |
| `POST /__session` | Mint a session id for a dispatched client |
| `GET /__status` | busy flag + this month's busy GPU seconds (the scheduler's quota ledger source) |

Client API (tunnel host only, header `X-Session-Id`):

| Route | Purpose |
|---|---|
| `GET /health` | Liveness probe (also how the scheduler detects a running container) |
| `POST /sr/submit-urls` | `{"urls": [...], "scale": 2\|3\|4}` — backend fetches the images; original-image links (`fullimg` / `or=`) are rejected |
| `POST /sr/submit-archive` | `{"url": "...", "scale": ...}` — backend downloads the zip, upscales every image, repacks |
| `POST /sr/begin-upload` + `POST /sr/upload-one` | Upload local images one by one (for galleries already on the phone) |
| `GET /sr/status` | `{stage, total, done, failed, processing, queuedAhead}` — stages: receiving / queued / processing / packing / done / error |
| `GET /sr/result` | The result zip (original file names preserved; failures listed in `_errors.txt`) |
| `DELETE /sr/session` | Cancel and wipe the session |

Any other Host is rejected with 403.

## Notes and limits

- Queueing is VRAM-gated: a worker only starts an image when free VRAM
  is above ~3.2 GB, so bursts queue instead of OOMing the card.
- Sessions expire after 2 h; a downloaded result is kept 30 min.
- Busy GPU seconds are persisted on the `jh-sr-data` volume
  (`/data/usage.json`) and survive scale-to-zero.
- The result zip is streamed through the tunnel; extremely large
  archives are bounded by Cloudflare/Modal request durations — if a
  download is cut, just fetch `/sr/result` again within the keep
  window.
- If `modal deploy` complains about the `@modal.concurrent` decorator,
  your Modal SDK is older than the decorator — delete that one line;
  it only raises the concurrent-input cap.
