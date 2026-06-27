// Camera metadata for the HTML dashboard grid.
// Kept in sync with app/cameras.py GRID_CAMERAS by hand - when you add/rename a
// grid camera in the Python catalog, mirror the change here.
//
// Per camera (one of two shapes):
//   id    : the cam_id used in Firestore (doc id under `latest/` and the
//           cam_id field on `footfall/` docs).
//   name  : display name in the tile header.
//   city  : small grey subtitle.
//   page  : the public webcam page (clickable fallback / source link).
//   embed : iframe URL for the live player (use when the host is iframe-friendly,
//           e.g. tvkur). Mutually exclusive with `hls`.
//   hls   : direct HLS .m3u8 URL the dashboard plays in a <video> via hls.js
//           (use when the host blocks iframes but exposes CORS-open HLS).
//
// History (2026-06-27): replaced konya_kulturpark + konya_millet_caddesi with
// sultanahmet_1_yeni + taksim_yeni. The new IBB pages (istanbuluseyret.ibb.gov.tr/
// *-yeni/) set X-Frame-Options: SAMEORIGIN so we can't iframe them, but their
// HLS at kamerayayin.ibb.istanbul returns Access-Control-Allow-Origin: * and is
// reachable from any country - so the dashboard plays the m3u8 directly with
// hls.js, and the Python collector reads it via _grab_via_segment.

// Helper: tvkur HLS via our local proxy (/tvkur/<id>/master.m3u8 -> content.tvkur.com).
// We can't <video src=tvkur> directly: the CDN refuses requests without a
// Referer it trusts AND sends no CORS headers, so hls.js's fetch is blocked.
// serve.py / the notebook server relay the request with the right Referer
// and add Access-Control-Allow-Origin:* so hls.js plays the stream natively.
const tvkurHls = (id) => `/tvkur/${id}/master.m3u8`;

export const GRID_CAMERAS = [
  {
    id:    "konya_hukumet",
    name:  "Konya - Hükümet Meydanı / Sarraflar Yeraltı Çarşısı",
    city:  "Konya",
    hls:   tvkurHls("c77i84vbb2nj4i0fr80g"),
    page:  "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
  },
  {
    id:    "otogar_kavsagi",
    name:  "Konya - Otogar Kavşağı",
    city:  "Konya",
    hls:   tvkurHls("c77i91vbb2nj4i0fr81g"),
    page:  "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
  },
  {
    id:    "sultanahmet_1_yeni",
    name:  "Sultanahmet",
    city:  "Istanbul",
    hls:   "https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet1.stream/playlist.m3u8",
    page:  "https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/",
  },
  {
    id:    "taksim_yeni",
    name:  "Taksim Meydanı",
    city:  "Istanbul",
    hls:   "https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8",
    page:  "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/",
  },
];
