# IBB HLS Relay — Cloudflare Worker

The `kamerayayin.ibb.istanbul` CDN refuses every Google Cloud IP range
(HTTP 403). It answers normally from any other address, so proxying the
IBB requests through Cloudflare's edge — a different ASN — restores the
four Istanbul cameras (`taksim_yeni`, `sultanahmet_1_yeni`,
`eyup_sultan_yeni`, `beyazit_meydan_yeni`) with no paid infrastructure.

Cloudflare's free plan gives 100,000 Worker requests per day; our load
is ~26k/day for four cameras sampled every 40 seconds, so headroom is
comfortable.

## One-time setup (~5 minutes)

1. **Create a free Cloudflare account** at <https://dash.cloudflare.com/sign-up>
   if you don't have one. No card required.
2. **Install `wrangler`** on your local machine (once):
   ```bash
   npm install -g wrangler
   wrangler login
   ```
   The `login` command opens a browser tab; grant access and return.
3. **Deploy the worker**. From the repo root:
   ```bash
   cd src/deploy/cloudflare-proxy
   wrangler deploy
   ```
   The output shows the deployed URL, e.g.
   `https://ibb-proxy.<your-subdomain>.workers.dev`.
4. **Set the shared secret** (any random string). Wrangler prompts you
   to type it when you run:
   ```bash
   wrangler secret put PROXY_SECRET
   ```
   Pick something like `openssl rand -hex 24` and paste it in. This
   prevents strangers from burning your 100k/day quota.
5. **Configure the VM**. SSH into the collector and drop the URL +
   secret into `/etc/turkey-footfall/proxy.env`:
   ```bash
   sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<EOF
   IBB_PROXY_URL=https://ibb-proxy.<your-subdomain>.workers.dev
   IBB_PROXY_SECRET=<the same secret you just set>
   EOF
   sudo chmod 600 /etc/turkey-footfall/proxy.env
   sudo systemctl restart collector
   ```

## Verify

After the restart, from the VM:

```bash
cd /opt/turkey-footfall/src && sudo -E .venv/bin/python -m tools.probe_country --country turkey
```

Before this change: `0/24 live` (or `3/24 live` if the YouTube tier is
already installed). After: the four IBB cameras should also flip to
`LIVE` — total `7/24 live`.

You can also spot-check the worker directly:

```bash
curl -s -H "X-Proxy-Secret: <your secret>" \
  "https://ibb-proxy.<you>.workers.dev/https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" \
  | head -3
```

Expect an `#EXTM3U` playlist body. `403 forbidden` from the worker means
the secret is wrong; `403` with an IBB body header means Cloudflare
itself is now blocked (rare — swap to a different Worker region or fall
back to Option B in the digest report).

## What the worker does not do

- **No caching that would break liveness.** `cf.cacheTtl: 4` matches the
  ~4-second HLS segment rotation; longer would show stale frames.
- **No proxying of other hosts.** Only `kamerayayin.ibb.istanbul` is
  allowed; requests for anything else return 403 immediately.
- **No proxying of tvkur.com** (the Konya / Otogar / other webcamera24
  Turkish cameras). tvkur restricts even residential ASNs — a
  Cloudflare edge would face the same 403 the VM does. Those cameras
  need a Turkish-IP proxy specifically, and the operator has declined
  that path for the free-tier budget.
