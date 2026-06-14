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

export const GRID_CAMERAS = [
  {
    id:    "konya_hukumet",
    name:  "Konya — Hükümet Meydanı / Sarraflar Yeraltı Çarşısı",
    city:  "Konya",
    embed: "https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g",
    page:  "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
  },
  {
    id:    "otogar_kavsagi",
    name:  "Konya — Otogar Kavşağı",
    city:  "Konya",
    embed: "https://player.tvkur.com/l/c77i91vbb2nj4i0fr81g",
    page:  "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
  },
  {
    id:    "konya_kulturpark",
    name:  "Konya — Kültürpark",
    city:  "Konya",
    embed: "https://player.tvkur.com/l/c77i6hb84cnrb6mlji3g",
    page:  "https://webcamera24.com/camera/turkey/8058-kulturpark/",
  },
  {
    id:    "konya_millet_caddesi",
    name:  "Konya — Millet Caddesi / Hastane Kavşağı",
    city:  "Konya",
    embed: "https://player.tvkur.com/l/c77i9cfbb2nj4i0fr82g",
    page:  "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
  },
];
