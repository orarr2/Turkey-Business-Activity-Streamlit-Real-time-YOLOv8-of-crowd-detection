// Camera metadata for the HTML dashboard grid.
// Kept in sync with app/cameras.py GRID_CAMERAS by hand — when you add/rename a
// grid camera in the Python catalog, mirror the change here.
//
// Per camera:
//   id    : the cam_id used in Firestore (doc id under `latest/` and the
//           cam_id field on `footfall/` docs).
//   name  : display name in the tile header.
//   city  : small grey subtitle.
//   embed : iframe URL for the live player, or null if the platform sets
//           X-Frame-Options and blocks iframing.
//   page  : the public webcam page (clickable fallback when embed is null).

export const GRID_CAMERAS = [
  {
    id:    "konya_hukumet",
    name:  "Konya — Hükümet Meydanı / Sarraflar Yeraltı Çarşısı",
    city:  "Konya",
    embed: "https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g",
    page:  "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
  },
  {
    id:    "giresun_gazi",
    name:  "Giresun — Gazi Caddesi",
    city:  "Giresun",
    embed: "https://www.skylinewebcams.com/en/embed/turkey/giresun/giresun/gazi-street.html",
    page:  "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
  },
  {
    id:    "otogar_kavsagi",
    name:  "Otogar Kavşağı",
    city:  "Turkey",
    embed: null,            // webcamera24 page sets X-Frame-Options
    page:  "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
  },
  {
    id:    "kadikoy",
    name:  "Kadıköy",
    city:  "Istanbul",
    embed: null,            // İBB hosts set X-Frame-Options
    page:  "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
  },
];
