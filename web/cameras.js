// Camera metadata for the HTML dashboard grid.
// Kept in sync with app/cameras.py GRID_CAMERAS by hand — when you add/rename a
// grid camera in the Python catalog, mirror the change here.
//
// Per camera:
//   id    : the cam_id used in Firestore (doc id under `latest/` and the
//           cam_id field on `footfall/` docs).
//   name  : display name in the tile header.
//   city  : small grey subtitle.
//   embed : iframe URL for the live player (all four are tvkur, globally reachable).
//   page  : the public webcam page (clickable fallback / source link).

// Helper: tvkur player URL with autoplay+mute (Chrome blocks unmuted autoplay).
const tvkur = (id) => `https://player.tvkur.com/l/${id}?autoplay=true&mute=true`;

// Helper: route iframe content through corsproxy.io to strip X-Frame-Options.
// Skylinewebcams and istanbuluseyret.ibb.gov.tr both set frame-ancestors CSP that
// blocks our localhost from iframing them directly — going through a public
// CORS/XFO proxy makes the iframe load. Caveats: rate-limited (free tier), and
// some JS inside the proxied page may still try to talk back to the original
// origin and fail. For production, replace with a self-hosted proxy.
const PROXY = (url) => "https://corsproxy.io/?" + encodeURIComponent(url);

export const GRID_CAMERAS = [
  {
    id:    "konya_hukumet",
    name:  "Konya — Hükümet Meydanı / Sarraflar Yeraltı Çarşısı",
    city:  "Konya",
    embed: tvkur("c77i84vbb2nj4i0fr80g"),
    page:  "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
  },
  {
    id:    "giresun_gazi",
    name:  "Giresun — Gazi Caddesi",
    city:  "Giresun",
    embed: PROXY("https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html"),
    page:  "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
  },
  {
    id:    "otogar_kavsagi",
    name:  "Konya — Otogar Kavşağı",
    city:  "Konya",
    embed: tvkur("c77i91vbb2nj4i0fr81g"),
    page:  "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
  },
  {
    id:    "kadikoy",
    name:  "Kadıköy",
    city:  "Istanbul",
    embed: PROXY("https://istanbuluseyret.ibb.gov.tr/kadikoy/"),
    page:  "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
  },
];
