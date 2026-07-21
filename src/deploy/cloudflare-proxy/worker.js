// Cloudflare Worker — IBB HLS relay for the turkey-collector VM.
//
// The IBB CDN (kamerayayin.ibb.istanbul) returns HTTP 403 to every GCP
// IP range - Google Cloud is treated as a scraping ASN. From ANY other
// address (residential, Cloudflare edge, most cloud regions outside
// Google) IBB responds 200 with the full HLS chain. This worker fetches
// IBB from Cloudflare's edge (non-GCP IPs) and hands the bytes back to
// the collector, so all four Istanbul cameras become reachable again
// with no change to the IBB catalog and no paid infrastructure.
//
// Contract:
//   GET  https://<worker>/<encoded-target-url>
//   Header:  X-Proxy-Secret: <shared secret>
//
// Only kamerayayin.ibb.istanbul URLs are proxied - anything else is
// refused. The shared secret keeps the free 100k-req/day tier ours,
// not the whole internet's.

const ALLOWED_HOSTS = new Set([
  "kamerayayin.ibb.istanbul",
]);

export default {
  async fetch(request, env, ctx) {
    // ---- 1. Auth (shared secret) --------------------------------------
    const secret = request.headers.get("X-Proxy-Secret");
    if (!env.PROXY_SECRET || secret !== env.PROXY_SECRET) {
      return new Response("forbidden", { status: 403 });
    }

    // ---- 2. Extract the target URL (everything after the worker root) -
    const url = new URL(request.url);
    // Strip the leading "/" and append any query the collector sent - the
    // collector calls _http_get(url) which is the target verbatim, so the
    // worker sees it as the pathname.
    const target = url.pathname.slice(1) + (url.search || "");
    if (!target) {
      return new Response("missing target url", { status: 400 });
    }
    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch (_) {
      return new Response("invalid target url", { status: 400 });
    }
    if (!ALLOWED_HOSTS.has(targetUrl.hostname)) {
      return new Response(
        `host not allowed: ${targetUrl.hostname}`, { status: 403 });
    }
    if (targetUrl.protocol !== "https:") {
      return new Response("https only", { status: 400 });
    }

    // ---- 3. Fetch from the edge ---------------------------------------
    // Preserve the collector's own User-Agent when it sent one; otherwise
    // hand IBB a plain browser UA. IBB does not check Referer/Origin -
    // we verified that from a residential IP with three header sets and
    // got 200 every time.
    const upstreamHeaders = {
      "User-Agent": request.headers.get("User-Agent") ||
        "Mozilla/5.0 (compatible; ibb-proxy)",
      "Accept": "*/*",
    };
    let upstream;
    try {
      upstream = await fetch(targetUrl.toString(), {
        headers: upstreamHeaders,
        // Cloudflare's built-in cache: short TTL because playlists rotate
        // segment names every ~4s. Segments themselves are immutable and
        // benefit slightly, but the collector rarely re-fetches the same
        // one. cacheEverything=true trades a small hit-rate lift for the
        // egress savings this proxy is here to protect.
        cf: { cacheTtl: 4, cacheEverything: true },
      });
    } catch (e) {
      return new Response(`upstream error: ${e.message}`, { status: 502 });
    }

    // ---- 4. Pass status + Content-Type through unchanged --------------
    const respHeaders = new Headers();
    for (const h of ["Content-Type", "Cache-Control", "Content-Length"]) {
      const v = upstream.headers.get(h);
      if (v) respHeaders.set(h, v);
    }
    // Permissive CORS so a browser-side dashboard could also hit the
    // worker directly if it ever wanted to - the collector's Python
    // client ignores CORS entirely.
    respHeaders.set("Access-Control-Allow-Origin", "*");
    return new Response(upstream.body, {
      status: upstream.status,
      headers: respHeaders,
    });
  },
};
