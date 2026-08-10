<div dir="rtl">

# מדריך הפרויקט — פעילות מסחרית / Live Footfall

מסמך הפעלה מאוחד ומפורט לפרויקט כולו: איך המערכת עובדת, איך הניתוחים עובדים,
איך ה-VM עובד ואילו פקודות מפעילות אותו. מאחד לכדי קובץ אחד את כל מה שהיה
פזור בשישה מסמכים נפרדים (`deploy/gcp-vm/README.md`, `deploy/REBUILD.md`,
`deploy/cloudflare-proxy/README.md`, `deploy/gcp-billing-killswitch/README.md`,
`docs/firebase_setup.md` ו-`MODEL_GUIDE_HE.md` הישן). התאום האנגלי של המסמך
הזה — עם אותו מספר פרקים ואותם עוגנים — נמצא ב-[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md),
כך שהפניות עוברות בשני הכיוונים.

</div>

---

<div dir="rtl">

## ניווט מהיר

1. [מה המערכת עושה בסך הכל](#1-מה-המערכת-עושה-בסך-הכל)
2. [ארכיטקטורה — איפה כל דבר רץ](#2-ארכיטקטורה--איפה-כל-דבר-רץ)
3. [ה-VM — צלילה מלאה](#3-ה-vm--צלילה-מלאה)
4. [שולחן הפקודות של ה-VM](#4-שולחן-הפקודות-של-ה-vm)
5. [7 שכבות הניתוח החי](#5-7-שכבות-הניתוח-החי)
6. [ניתוח חלון עמוק (`behavior.analyze_window`)](#6-ניתוח-חלון-עמוק-behavioranalyze_window)
7. [המחברת — אנליטיקה מקומית](#7-המחברת--אנליטיקה-מקומית)
8. [בחירת המודל והפרמטרים](#8-בחירת-המודל-והפרמטרים)
9. [אנומליות ודיווח](#9-אנומליות-ודיווח)
10. [לולאת הלמידה הפעילה](#10-לולאת-הלמידה-הפעילה)
11. [הגדרות פרויקט Firebase](#11-הגדרות-פרויקט-firebase)
12. [‏Cloudflare Worker ל-IBB](#12-cloudflare-worker-ל-ibb)
13. [ה-Killswitch של החיוב ב-GCP](#13-ה-killswitch-של-החיוב-ב-gcp)
14. [תקלות נפוצות ותשובות](#14-תקלות-נפוצות-ותשובות)
15. [נספח: החלטות עיצוב שהתקבלו](#15-נספח-החלטות-עיצוב-שהתקבלו)

</div>

---

<div dir="rtl">

## 1. מה המערכת עושה בסך הכל

המערכת ממירה 4 מצלמות רחוב פומביות לזרם נתונים כמותי:

> **‏Live HLS stream → YOLOv8 frame inference → counts + appearance re-ID →
> Firestore → real-time web dashboard + Jupyter analytics.**

בכל סבב דגימה (‏40 שניות בפריסת הענן שנכתבה בקוד; ‏`--interval` קובע אותו),
הקולקטור אוסף burst של פריימים מכל מצלמה פעילה, מריץ YOLO על כל פריים,
מחשב ספירות מדוללות + חתימות מראה + שערי אנומליה, ואז כותב את התוצאה
ל-Firestore. הדשבורד באתר נרשם עם `onSnapshot` — בלי polling, בלי refresh;
כל כתיבה של הקולקטור מופיעה מיד.

הגריד הוא **גנרי-מדינתי**: הוא תמיד מריץ 4 מצלמות ממדינה **אחת** וסובב לפי
סולם עדיפויות (**טורקיה ← תאילנד ← יפן ← ארה"ב**), נופל למדינה הבאה רק
כשהמדינה הפעילה כולה חשוכה. טורקיה היא נושא הפרויקט; מ-GCP הזרם של IBB
איסטנבול חסום גיאוגרפית, לכן בפועל הגריד רץ בדרך כלל על ספסלי החו"ל
(מצלמות רחוב מגובות YouTube-Live) עד שגם Cloudflare Worker ב-ASN שאיננו של
גוגל מחזיר את IBB (ראה פרק 12).

**מה מפיקה המערכת:**

- **‏Footfall** — כמה אנשים / כלי רכב פר מצלמה פר סבב.
- **‏Re-identification** — זיהוי-מחדש של אותה ישות (אדם או רכב) לאורך זמן
  (חתימות OSNet כאשר קובץ ה-ONNX קיים, hist HSV אחרת).
- **אנומליות** — עומס קיצוני, חסימת מצלמה, החשכה, שהייה ממושכת, מבקר חוזר,
  חפץ נטוש, חשד לנפילה.
- **מהירות אופיינית של רכבים בקמ"ש** (מוצגת רק כשלמצלמה יש מסה סטטיסטית
  אמיתית: לפחות 5 דגימות ולפחות 10% מהסבבים; רחבת הולכי-רגל בלי תנועה אמיתית
  מקבלת "-" במקום מספר מומצא).
- **דוח PDF** במייל לפי דרישה (כפתור "Send Report" בדשבורד); דיגסט מתוזמן
  פעמיים ביום נשלח אך ורק לתיבת הארכיון של הפרויקט.

הכל רץ על מסלולים חינמיים: מודל בקוד פתוח, GCP `e2-micro` במסלול Always
Free ($0/חודש), GitHub Actions על ריפו ציבורי, Firebase Spark plan.

### 1.1 שני זמני-ריצה, אותו קוד בסיס

| | ‏VM בענן (‏פרודקשן 24/7) | מחברת מקומית (הפניה לדיוק) |
|---|---|---|
| מארח | GCP `e2-micro` (1 GB RAM) | כל מחשב נייד |
| דטקטור | `yolov8s.pt` @ `imgsz 640` (‏fallback לזיכרון: `yolov8n.pt` @ 768) | `yolo26m.pt` @ `imgsz 960` |
| ייעוד | רץ לנצח, מזין את הדשבורד + הדוחות | ניתוח עמוק; גם ההפניה של ה-ground-truth ל-calibration |
| זרימת נתונים | Firestore + Firebase Storage | ‏CSV cache מקומי הנמשך מ-Firestore |
| קובץ המחברת | — | `turkey_business_activity.ipynb` (בגיט, `MODEL_WEIGHTS = 'yolo26m.pt'`) |
| מחברת תאומה | — | `turkey_business_activity_yolov8n.ipynb` (מקומית בלבד, ב-`.gitignore`; משקפת את מודל ה-VM כדי להשוות ‏apples-to-apples על אותה מצלמה) |

התאומה מכוונת לא-להיות-בגיט: היא עותק ידני של המחברת הראשית עם
`MODEL_WEIGHTS = 'yolov8s.pt'` (או `yolov8n.pt`), כדי שהמפעיל יראה על
המחשב שלו בדיוק מה שה-VM היה רואה על אותו פריים.

### 1.2 סולם ה-fallback של המדינות

הקולקטור לעולם לא נעול על קבוצת מצלמות קבועה. מחלקה בשם `CountryDirector`
מנהלת שני סולמות מקוננים:

- **סולם המדינות (עדיפות):** טורקיה ← תאילנד ← יפן ← ארה"ב. הגריד מציג 4
  מצלמות ממדינה אחת ועובר למדינה הבאה **רק** כשהמדינה הפעילה לא יכולה
  להעמיד ולו מצלמה חיה אחת. מצלמה בודדת שנפלה לא מזיזה את הגריד — מצלמה
  מהספסל של אותה מדינה מחליפה אותה.
- **סולם המצלמות (בתוך כל מדינה):** מחלקת `CameraPool` עוברת על רשימת
  המצלמות של המדינה ומקצה כל סבב את 4 המצלמות החיות הראשונות (תמיד שונות);
  מצלמה שהחטיאה 3 דגימות ברצף נחה 15 דקות; מצלמות `tvkur` (קוניה) הן מסלול
  fail-fast — החמצה אחת מספיקה כדי להרגיע אותן.
- **מנתק-זרם ברמת ה-host** (`HostBreaker`): 4 סירובי גישה רצופים (‏HTTP
  403/429) — כל מצלמות ה-host נחות 20 דקות ובקשת גישוש בודדת מחזירה אותן.
  ‏CDN חוסם מקבל ~3 בקשות בשעה במקום ~120.
- **התאוששות לפני הדוח:** כמה דקות לפני כל דוח מתוזמן (‏12:00 ו-20:00 שעון
  ישראל) הקולקטור מגשש שוב את המדינות בעדיפות גבוהה יותר, כדי שטורקיה תחזור
  לגריד ברגע ש-IBB משתחררת.

שדות הדוח בוחרים day/night לפי **אזור-הזמן של כל מצלמה** (הספסל האמריקאי לבדו
משתרע על Eastern / Central / Pacific).

</div>

---

<div dir="rtl">

## 2. ארכיטקטורה — איפה כל דבר רץ

</div>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GCP e2-micro VM (1 GB RAM, 24/7)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ collector.py:  loop { grab_frame → YOLO → count → re-ID → events }   │  │
│  │ 4 cameras in parallel, ~40 s per round                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                             ↓  Firestore + Firebase Storage                 │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓                                                              ↓
┌──────────────────┐                                     ┌─────────────────────┐
│  Dashboard in    │                                     │ On-demand report    │
│  the browser     │                                     │ (dashboard button   │
│  localhost:8000  │                                     │ → PDF to inbox)     │
└──────────────────┘                                     └─────────────────────┘
       ↓ operator labels                                          ↓ push
┌──────────────────┐          ┌───────────────────────┐    ┌─────────────────┐
│ training_sync    │  ──→     │ GitHub Actions        │    │ Inbox           │
│ pushes labels    │          │ train-head (free)     │    │ notification    │
│ to Storage       │          │ promotion gate → SA   │    └─────────────────┘
└──────────────────┘          └───────────────────────┘
                                        ↓ hot-load
                              ┌───────────────────────┐
                              │ Collector pulls new   │
                              │ head without restart  │
                              └───────────────────────┘
```

<div dir="rtl">

מונחי-מסגרת:

- **‏VM** — Virtual Machine בענן של גוגל. הפרויקט משתמש ב-`e2-micro`,
  המסלול הקטן ביותר של GCP (‏2 vCPU שיתופיים, 1 GB RAM), שנכלל ב-Always Free.
- **‏Firestore** — מסד נתונים NoSQL של גוגל. הדשבורד נרשם ב-`onSnapshot`
  וכל כתיבה של הקולקטור מגיעה לדפדפן ללא polling.
- **‏Firebase Storage** — דלי אחסון אובייקטים עבור snapshots JPEG וייצואי
  JSON של heatmap. יש כלל lifecycle של 24 שעות על `snapshots/`.
- **‏GitHub Actions** — ה-CI החינמי של GitHub. הפייתון של אימון הראש רץ שם
  על דקות של ריפו ציבורי; הראש שקודם נוחת ב-Storage והקולקטור מבצע hot-swap
  שלו בלי restart.

**עיקרון מרכזי:** הדשבורד הוא **צרכן טהור**. כל מי שmakes-clone של הריפו
יכול להגיש `web/` ולראות את אותו גריד חי — כי כל המצב חי ב-Firestore,
וה-TTL של Firestore מנקה כל אוסף היסטוריה אחרי 24 שעות.

</div>

---

<div dir="rtl">

## 3. ה-VM — צלילה מלאה

### 3.1 המכונה עצמה

</div>

```
Provider    : Google Cloud Platform
Machine     : e2-micro (Always Free)
CPU         : 2 vCPU shared (0.25 vCPU guaranteed, ~1 vCPU burst)
RAM         : 1024 MB total (~950 MB usable after kernel)
Disk        : 30 GB SSD (Standard persistent disk)
OS          : Debian 12 (bookworm)
Zone        : us-east1-c (Virginia)
Public IP   : ephemeral (rotates on stop/start; a static IP costs money)
Instance ID : turkey-collector
Project     : turkey-footfall
Cost        : $0/month (Always Free — one e2-micro per account,
              us-central1 / us-east1 / us-west1 only)
```

<div dir="rtl">

**מה שחשוב לשים לב:**

- **‏"0.25 vCPU guaranteed"** אומר שברירת המחדל היא רבע ליבה; יש burst עד
  ~1 vCPU כאשר יש חלון פנוי בשרת המשותף. סבבים לפעמים לוקחים ~25% יותר זמן
  בלי סיבה נראית לעין — זה מס-דיירים, לא באג.
- **‏1 GB RAM זה מעט מאוד** למודל של 11M פרמטרים + 4 זרמי HLS פעילים
  + OSNet. נוסף `/swapfile` של 2 GB ידנית (לא ב-`install.sh`) כביטחון;
  שיא נמדד ~273 MB.
- **‏us-east1-c, לא קרוב לטורקיה** — Always Free מוגבל לאזורים us-central1
  ‏/ us-east1 / us-west1. ה-RTT לאיסטנבול הוא ~150ms שלא חשוב כאן — דגימה
  היא כל 40 שניות, latency לא משפיע.

### 3.2 מבנה התיקיות ב-VM

</div>

```
/opt/turkey-footfall/              ← הריפו נמצא כאן, קלון מ-main
├── src/
│   ├── app/                       ← כל מודולי הפייתון (collector, detect, tracker, reid, heatmap, faces, pose, gestures, behavior, live_analysis, dashboard_server, alerts, adapters ...)
│   ├── tools/                     ← כלי CLI (analyze_window, calibrate_conf, probe_country, daily_digest, train_head, promote_adapter, fetch_training_data ...)
│   ├── data/
│   │   ├── reid.db                ← SQLite של OSNet embeddings
│   │   ├── osnet_x0_25_msmt17.onnx ← מודל ה-re-ID (~5 MB)
│   │   ├── confidence_boost.json  ← "learned" per-(cam,cls) gate nudges
│   │   ├── blacklist_auto.json    ← polygons של auto-blacklist
│   │   ├── per_camera_conf.json   ← פלט WS4: gates מכוילי-precision
│   │   └── adapters/              ← ראשים אחרי fine-tune (‏head-only)
│   │       ├── current.json       ← מצביע לראש הפעיל
│   │       ├── history.jsonl      ← יומן קידומים/דחיות
│   │       └── head_run<N>.pt     ← קובץ הראש עצמו
│   ├── web/
│   │   ├── snapshots/             ← מטמון snapshots חי
│   │   │   ├── review_frames/     ← פריימים בתור לתיוג (LRU 500)
│   │   │   ├── live_samples/      ← crops לחיפוש חזותי (LRU 1000)
│   │   │   ├── entities/          ← per-entity crops (LRU 400)
│   │   │   ├── anomalies/         ← snapshots של אנומליות (24h TTL)
│   │   │   └── heatmaps/          ← overlays של heat + <cam>.json
│   │   └── firebase-config.js     ← מפתחות web SDK ציבוריים (בטוח בגיט)
│   ├── .venv/                     ← Python virtualenv (~2 GB, torch)
│   └── deploy/                    ← install.sh, systemd unit templates, worker.js
├── yolov8s.pt                     ← משקולות הזיהוי (ultralytics מוריד בהרצה הראשונה)

/etc/turkey-footfall/              ← קונפיגורציה מוגנת (root:root)
├── serviceAccount.json            ← מפתח Firebase Admin SDK (0400)
├── proxy.env                      ← IBB_PROXY_URL, IBB_PROXY_SECRET (0600)
└── digest.env                     ← GMAIL_USER, GMAIL_APP_PASSWORD (0600)

/etc/systemd/system/
├── collector.service
├── digest.service
└── digest.timer

/var/log/journal/                  ← יומני systemd (מתחלפים אוטומטית)
/swapfile                          ← 2 GB swap (שלב ידני; ראה 3.7)
```

<div dir="rtl">

הערות:

- הקולקטור רץ כ-root כי הוא צריך לקרוא את `serviceAccount.json` (מצב 0400
  root:root). שום דבר אחר ב-VM לא זקוק להרשאות מוגברות.
- `.venv/` נשלט על ידי `torch` (~1.2 GB) + `ultralytics` (~200 MB). אל
  תנסה לגזום — שניהם load-bearing.
- כל מה שתחת `web/snapshots/` מתחדש תוך סבב אחד — בטוח למחוק לצורך restart נקי.

### 3.3 התקנה

**מקדימים (פעם אחת, בקונסולת GCP):**

1. הפעל חיוב על הפרויקט (נדרש גם למכונות ‏free-tier).
2. הפעל APIs: Compute Engine, Secret Manager, Cloud Storage.
3. ‏Secret Manager ← Create secret `firebase-sa`, הדבק את ה-JSON של מפתח
   ‏Firebase Admin SDK כערך הסוד.
4. הענק ל-`<PROJECT_NUMBER>-compute@developer.gserviceaccount.com` את
   התפקיד **‏Secret Manager Secret Accessor** על `firebase-sa`.
5. ‏Firestore console ← Time-to-live ← הוסף TTL על `footfall.expire_at`
   וגם על `events.expire_at`.
6. ‏Firebase Console ← Storage ← Get started; ואז GCP ← Cloud Storage ←
   הדלי ← Lifecycle ← מחק קבצים תחת `snapshots/` אחרי יום.

**‏Create the VM (‏Console ← Compute Engine ← Create instance):**

- שם: `turkey-collector`
- אזור: `us-east1` (או `us-central1` / `us-west1`)
- ‏Machine type: `e2-micro`
- ‏Boot disk: Debian 12, Standard persistent disk, 30 GB
- ‏Firewall: השאר HTTP/HTTPS לא מסומן (הקולקטור לא מאזין לשום דבר)
- ‏Identity & API access: השאר את חשבון השירות הדיפולטיבי

**להתקין את הקולקטור (ברגע שה-VM עלה):**

לחץ SSH ליד ה-VM בקונסולה, ואז:

</div>

```bash
curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh \
  | sudo bash
```

<div dir="rtl">

הסקריפט idempotent — הרצה שוב היא הדרך הסטנדרטית לרענן קוד. ששת השלבים:

1. ‏`apt-get install` חבילות מערכת: `git`, `python3-venv`, `ffmpeg`,
   ה-‏shared libs של OpenCV, `fonts-dejavu-core`.
2. ‏Clone (או fetch + reset) `/opt/turkey-footfall`.
3. יצירת `.venv`, `pip install -r requirements.txt` עם `TMPDIR=/var/tmp`
   (מונע מיצוי ה-`/tmp` שהוא RAM-backed באמצע ההתקנה).
4. משיכת מפתח Firebase Admin מ-Secret Manager אל
   `/etc/turkey-footfall/serviceAccount.json` (‏mode 0400 root:root).
5. זיהוי הדלי של Firebase Storage (`<project>.firebasestorage.app` תחילה,
   אז ה-legacy `<project>.appspot.com`); רינדור של תבניות ה-systemd units
   עם `sed`; התקנה ל-`/etc/systemd/system/`.
6. ‏`systemctl enable --now collector.service`; התקנת `digest.service`
   ו-`digest.timer` (ה-timer מופעל רק אם `/etc/turkey-footfall/digest.env`
   קיים).

**שלבים חד-פעמיים לאחר ההתקנה ש-`install.sh` **‏לא** מכסה:**

</div>

```bash
# swap של 2 GB — ה-VM של 1 GB זקוק לזה (שיא נמדד: 273 MB בשימוש)
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# env של הסוד ל-digest של Firebase Storage
sudo tee /etc/turkey-footfall/digest.env > /dev/null <<'EOF'
GMAIL_USER=<gmail address>
GMAIL_APP_PASSWORD=<16-char app password from myaccount.google.com/apppasswords>
EOF
sudo chmod 600 /etc/turkey-footfall/digest.env
sudo systemctl enable --now digest.timer

# ‏Cloudflare-Worker relay ל-IBB (ראה פרק 12 להגדרת wrangler)
sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<'EOF'
IBB_PROXY_URL=https://ibb-proxy.<subdomain>.workers.dev
IBB_PROXY_SECRET=<same secret you set on the worker>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env
sudo systemctl restart collector

# מודל OSNet re-ID (~5 MB, אופציונלי אבל מומלץ)
sudo bash /opt/turkey-footfall/src/tools/setup_reid.sh
```

<div dir="rtl">

### 3.4 ‏`collector.service` — יחידת ה-systemd

התבנית נמצאת ב-`src/deploy/gcp-vm/collector.service`; ‏`install.sh` מרנדר
אותה עם `sed -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' ...` לפני
כתיבה ל-`/etc/systemd/system/`. השורות המרכזיות ומה הן עושות:

</div>

```ini
Environment=OMP_NUM_THREADS=2
# תואם לשני vCPU השיתופיים של ה-e2-micro. ברירת המחדל של torch
# oversubscribes ומשליכה context switches על המכונה הזאת.

Environment=MALLOC_ARENA_MAX=2
# glibc מגדל arena אחת ל-thread כברירת מחדל (~50-150 MB של RSS על תהליך
# פייתון מרובה threads). כסף אמיתי במכונה של 1 GB שנהרגה בעבר על ידי
# kernel oom-killer ב-696 MB peak.

Environment=FIREBASE_CREDENTIALS=/etc/turkey-footfall/serviceAccount.json
Environment=FIREBASE_STORAGE_BUCKET=<detected at install time>
Environment=REID_MODEL=/opt/turkey-footfall/src/data/osnet_x0_25_msmt17.onnx

EnvironmentFile=-/etc/turkey-footfall/proxy.env
# ‏IBB relay + secret אופציונלי; כשהקובץ חסר הקולקטור רץ ללא שינוי
# (IBB נשאר 403 מ-GCP, גריד טורקיה תלוי בשכבת ה-YouTube בלבד).

Environment=EXTRA_CLASSES=bird:0.35,dog:0.35,cat:0.40,backpack:0.35,handbag:0.40,suitcase:0.35,umbrella:0.35
# ‏fix1: תיקים מזינים את מעקב-חפץ-נטוש; חיות + שמשיות מקבלות ספירה נפרדת
# (שורת "other objects" בדוח). זיהוי בלבד, לעולם לא אימון.

Environment=FALL_CHECK=1
# ‏fix1-A11: אירוע person-loiter מפעיל pose pass אחד על ה-crop של אותו אדם;
# torso אופקי מעלה את ההתראה ל-"Possible FALL".

ExecStart=/opt/turkey-footfall/src/.venv/bin/python \
  -m app.collector --weights yolov8s.pt --interval 40 --imgsz 640 \
  --burst 2 --burst-stride 13

MemoryHigh=760M
MemoryMax=900M
# מגבלות ה-cgroup. אם היומן מראה reclaim throttling או oom-kills, נפילת
# הגיבוי היא `--weights yolov8n.pt --imgsz 768` ב-ExecStart לפני שנוגעים
# בשום דבר אחר.
```

<div dir="rtl">

**חשוב לעדכוני קוד:** ‏`git pull` + ‏`systemctl restart` לבד מספיק — ה-unit
המותקן נושא שורות **‏Environment=** ‏machine-local (‏`FIREBASE_CREDENTIALS`,
‏`FIREBASE_STORAGE_BUCKET`, ‏`REID_MODEL`) שבמכוון אינן ב-template בגיט.
דריסה של ה-unit המותקן מה-template **מוחקת** את השורות האלה, והקולקטור נכנס
ל-crash-loop עם `FileNotFoundError: Firebase service-account JSON not found`.
כדי לשנות דגל, ערוך את ה-unit המותקן במקום:

</div>

```bash
sudo sed -i 's#--weights yolov8s.pt --interval 40 --imgsz 640#--weights yolov8s.pt --interval 40 --imgsz 512#' \
  /etc/systemd/system/collector.service
sudo systemctl daemon-reload && sudo systemctl restart collector
```

<div dir="rtl">

### 3.5 ‏`digest.service` + ‏`digest.timer`

הדיגסט היומי הפך ל-on-demand בלבד (דרך כפתור בדשבורד); ה-timer שרץ פעמיים
ביום כותב עכשיו רק לתיבת הארכיון של הפרויקט. הפעלתו דורשת
`/etc/turkey-footfall/digest.env` עם `GMAIL_USER` ו-`GMAIL_APP_PASSWORD`.

### 3.6 הלולאה הראשית — זרימת דגימה

מפושט מ-`app/collector.py`:

</div>

```python
while True:
    round_start = time.time()
    for slot_id, cam_id in director.current_grid():
        try:
            counts, boxes, frames = sample_slot(cam_id)
        except Exception as e:
            record_miss(cam_id, e)
            continue
        write_firestore(slot_id, cam_id, counts)
        heatmap.accumulate(cam_id, boxes, frames[-1].shape)
        run_anomaly_gates(cam_id, counts, boxes, frames)
        maybe_capture_review_frame(cam_id, frames[-1], boxes)
    reload_review_overrides_if_due()
    reload_adapter_if_due()
    time.sleep(max(0, INTERVAL - (time.time() - round_start)))
```

<div dir="rtl">

כל ‏`sample_slot` תופס `burst` (ברירת מחדל 2 פריימים, `--burst-stride 13`
פריימים במרווח = ~0.5s ב-25fps), מריץ `detect_and_count`, מפעיל ‏ROI +
פילטרי gate + polygons של auto-blacklist, ואז מאחד את ה-burst על ידי חציון
(תיבות מזוייפות בפריים בודד לא יכולות למשוך את הבין למעלה).

### 3.7 מודל הזיכרון

המגבלות ‏`MemoryHigh=760M / MemoryMax=900M` יחד עם ה-envs `MALLOC_ARENA_MAX=2`
ו-`OMP_NUM_THREADS=2` וה-`/swapfile` של 2 GB — הן שגורמות לכל הדבר להיכנס
פיזית ל-RAM של 1 GB. סטטוס יציב שנמדד לאחר עדכון הדיוק של 2026-08-05
(4 מצלמות, ‏`burst 2`, ‏`yolov8s @ 640`): ‏RSS של ~410 MB, ‏96-100% CPU idle,
‏swap 0-30 MB.

אם היומן מראה אחד מהמצבים הבאים, נפול ל-`yolov8n @ 768`:

- ‏Reclaim throttling (הודעות `memory pressure` ב-`journalctl -u collector`)
- ‏Kernel oom-kill loops (`Killed process ... (python)` ב-`journalctl -k`)
- סבבים שנמתחים מעבר ל-interval (`! round took Ns > interval` ב-app log;
  התווית "counts from Ns ago" בדשבורד מסמנת אדום)

### 3.8 מבנה Firestore

| קולקציה | ‏Rows/day | צורת document | גישת client |
|---|---|---|---|
| ‏`footfall/{auto}` | ~17k | `{cam_id, ts, person, vehicles, ok, night, expire_at}` (TTL 24h) | קריאה בלבד |
| ‏`latest/{cam_id}` | ~2.16k (‏upserts) | דגימה אחרונה פר מצלמה | קריאה בלבד |
| ‏`reid_stats/{cam_id}` | ‏upsert יומי | ספירות unique / seen-again | קריאה בלבד |
| ‏`events/{auto}` | ~0-50 | אירועי אנומליה (‏TTL 24h) | קריאה בלבד |
| ‏`config/grid` | ‏1 doc | 4 המצלמות הנוכחיות + מדינה פעילה | קריאה בלבד |
| ‏`training_events/{auto}` | 1 לקידום | ‏mAP + labels_total לעקומת ה-AL | קריאה בלבד |

מכסת Spark plan: 20,000 כתיבות/יום. ב-`--interval 40` הקולקטור משתמש ב-~17k
(4 slots × 2 writes/round × 2160 samples/day). העלאת `--interval` ל-60 שניות
מכניסה את זה בהחלט מתחת למכסה.

### 3.9 מבנה Firebase Storage

</div>

```
snapshots/
├── review_frames/<cam>/<ts>_uNN.jpg   ← תור התיוג (LRU בדיסק ה-VM, נשלח ל-Storage באצווה)
├── live_samples/<cam>/<ts>_<cls>.jpg  ← crops לחיפוש חזותי
├── entities/<eid>/<ts>.jpg            ← per-entity crops (עד 6 ליישות)
├── anomalies/<cam>/<ts>.jpg           ← ראיות אנומליה (‏24h lifecycle)
├── heatmaps/<cam>.jpg                 ← ‏overlay פר-מצלמה (רענון אחרון)
└── heatmaps/<cam>.json                ← ה-grid המלא per-daypart × per-layer
training/
├── labels/                             ← verdicts של המפעיל שהועלו על ידי training_sync
├── frames/                             ← פריימים שהתיוגים מתייחסים אליהם
└── adapters/<run>/head.pt              ← ראש שקודם + metadata
```

<div dir="rtl">

### 3.10 ‏Hot-swap של ה-Detect head

כל 30 סבבים הקולקטור בודק את `data/adapters/current.json`, ואם המצביע
השתנה — קורא ל-`adapters.overlay_head(model, head_state_dict)` — זה
מטייל על ה-Detect module ומעתיק את ה-tensors של הראש במקום. אין restart,
אין קפיצת זיכרון, אין הפרעה לדגימה. גיבוי: אם `current.json` חסר או לא
קריא, מודל הבסיס רץ ללא שינוי (byte-identical).

</div>

---

<div dir="rtl">

## 4. שולחן הפקודות של ה-VM

### 4.1 ‏SSH פנימה

שלוש דרכים כניסה, כולן מגיעות לאותו `turkey-collector`:

</div>

```bash
# מהמחשב שלך (פעם אחת: gcloud auth login && gcloud config set project turkey-footfall)
gcloud compute ssh turkey-collector --zone=us-east1-c

# Browser SSH: Console → Compute Engine → כפתור SSH ליד ה-instance
# Mobile: אפליקציית Google Cloud → Compute Engine → SSH
```

<div dir="rtl">

### 4.2 ניהול השירותים

</div>

```bash
sudo systemctl status  collector          # רץ? (want: active (running))
sudo systemctl restart collector          # אחרי שינוי קוד או env
sudo systemctl stop    collector          # השהיה מכוונת (חיוב ממשיך)
sudo systemctl start   collector          # חידוש
sudo systemctl status  digest.timer       # timer דוח מתוזמן
sudo systemctl list-timers                # כל ה-timers, מועדי fire הבאים
```

<div dir="rtl">

### 4.3 יומנים

</div>

```bash
sudo journalctl -u collector -f                              # tail חי
sudo journalctl -u collector -n 200                          # 200 שורות אחרונות
sudo journalctl -u collector --since "15 min ago"            # חלון
sudo journalctl -u collector --since "6h" | grep -iE "oom|Killed"   # ציד oom

# one-liner: ספירות success/miss ב-15 דקות האחרונות
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn
```

<div dir="rtl">

### 4.4 פריסת קוד חדש

</div>

```bash
# המסלול הסטנדרטי — בטוח גם עם היסטוריה שנדחפה force-pushed
sudo git -C /opt/turkey-footfall fetch origin main && \
  sudo git -C /opt/turkey-footfall reset --hard origin/main && \
  sudo systemctl restart collector

# חלופה: הרצת install.sh (idempotent; גם מרענן venv deps)
sudo /opt/turkey-footfall/src/deploy/gcp-vm/install.sh
```

<div dir="rtl">

לעולם אל תעשה ‏`sed ... | tee /etc/systemd/system/collector.service` מה-template
בגיט — ה-unit המותקן נושא שורות ‏Environment= ‏machine-local (ראה 3.4).

### 4.5 בטריית health-check — האם ה-VM באמת מזין את הדוח?

הרץ את זה לפני שאתה סומך על דוח כלשהו:

</div>

```bash
# 1. השירות חי
sudo systemctl status collector --no-pager | head -12

# 2. דגימה חיה — want slot_1..4 עם ספירות אמיתיות מתגלגלות כל ~40 שניות
sudo journalctl -u collector -f --no-hostname | grep --line-buffered -E "slot_|MISS|country"

# 3. יחס success/miss, 15 דקות אחרונות
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn

# 4. מרחב זיכרון
sudo systemctl show collector -p MemoryCurrent -p MemoryMax && free -h

# 5. oom kills אמיתיים בלבד (התעלם מרעש URL של googlevideo HLS)
sudo journalctl -u collector --since "6h" | grep -iE "oom-kill|Killed process|out of memory"

# 6. env של IBB proxy מחוברים (ערכים חסויים; want 2)
sudo grep -c -E "IBB_PROXY_URL|IBB_PROXY_SECRET" /etc/turkey-footfall/proxy.env

# 7. DECISIVE end-to-end — תופס פריים אמיתי מטורקיה עכשיו
sudo bash -c 'set -a; . /etc/turkey-footfall/proxy.env; set +a; \
  cd /opt/turkey-footfall/src && timeout 90 .venv/bin/python - <<PY
from app.cameras import CAMERAS
from app.detect_core import resolve_stream, grab_burst
url = resolve_stream(CAMERAS["taksim_yeni"])
frames = grab_burst(url, n=2, stride=10)
print("frames grabbed:", len(frames), "shape:", frames[0].shape if frames else None)
PY'
# want: frames grabbed: 2 shape: (1080, 1920, 3)

# 8. הקוד שנפרס עדכני
sudo git -C /opt/turkey-footfall log --oneline -1
```

<div dir="rtl">

### 4.6 זיכרון / דיסק / swap

</div>

```bash
free -h                                # RAM + swap use
df -h /                                # root disk (30 GB סה"כ, expect <10 GB used)
sudo du -sh /opt/turkey-footfall/src/.venv     # ~2 GB, נורמלי
sudo du -sh /opt/turkey-footfall/src/web/snapshots  # מטמון אנומליות + review
```

<div dir="rtl">

### 4.7 החלפת ה-IP החיצוני

לפעמים IP חדש מנקה rate-block של CDN. מחיקה והוספה מחדש של ה-access config
מחליפה את ה-IP בלי reboot:

</div>

```bash
NAME=$(gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].name)")
gcloud compute instances delete-access-config turkey-collector \
  --zone=us-east1-c --access-config-name="$NAME"
gcloud compute instances add-access-config turkey-collector --zone=us-east1-c
gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

<div dir="rtl">

### 4.8 בנייה מחדש מאפס

ה-VM חד-פעמי במטרה. שלושה סודות שורדים: מפתח Firebase Admin (מנפיקים
מחדש מ-Firebase Console ← Service accounts ← Generate new private key),
סוד ה-IBB relay (`wrangler secret put PROXY_SECRET`), וסיסמת אפליקציה
של Gmail (‏`myaccount.google.com/apppasswords`).

**מסלול א' — בנייה מחדש ב-GCP (מכונה זהה):**

</div>

```bash
# 1. יצירת המכונה
gcloud compute instances create turkey-collector \
  --project=turkey-footfall --zone=us-east1-c \
  --machine-type=e2-micro \
  --image-family=debian-13 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard

# 2. bootstrap (חבילות, ריפו, venv, מפתח Firebase מ-Secret Manager, יחידות)
gcloud compute ssh turkey-collector --zone=us-east1-c --project=turkey-footfall \
  --command='curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh | sudo bash'

# 3. swap, env files, digest timer (ראה 3.3)
```

<div dir="rtl">

**מסלול ב' — בנייה מחדש בכל ספק לינוקס אחר:**

דרישות: ‏Debian 12/13 או Ubuntu, x86_64, 1 GB+ RAM (עם שלב ה-swap),
~20 GB דיסק, אינטרנט יוצא. הקולקטור לא מאזין לשום דבר. ‏`install.sh`
מניח image של GCP; אצל ספק אחר, הרץ את השלבים ידנית ושים את המפתח של
Firebase ביד:

</div>

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip \
  ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  ca-certificates curl fonts-dejavu-core

sudo git clone --depth 1 https://github.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection.git /opt/turkey-footfall
cd /opt/turkey-footfall/src
sudo python3 -m venv .venv
sudo TMPDIR=/var/tmp .venv/bin/pip install --no-cache-dir -r requirements.txt

sudo mkdir -p /etc/turkey-footfall
sudo install -m 0400 -o root -g root ~/serviceAccount.json /etc/turkey-footfall/serviceAccount.json

for unit in collector.service digest.service digest.timer; do
  sudo sed -e 's|__STORAGE_BUCKET__|turkey-footfall.firebasestorage.app|g' \
           -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' \
           -e 's|__SA_PATH__|/etc/turkey-footfall/serviceAccount.json|g' \
    /opt/turkey-footfall/src/deploy/gcp-vm/$unit | sudo tee /etc/systemd/system/$unit > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now collector.service
```

<div dir="rtl">

ואז יישם את שלבי ה-swap + env files + timer של מסלול א'. מצלמות איסטנבול
עובדות מכל ASN ש-Cloudflare מגיע אליו; ‏ASN ‏residential לא בהכרח זקוקים
ל-relay — בדוק עם `python -m tools.probe_country --country turkey`.

### 4.9 הסרה

</div>

```bash
sudo systemctl disable --now collector digest.timer
sudo rm /etc/systemd/system/collector.service /etc/systemd/system/digest.{service,timer}
sudo rm -rf /opt/turkey-footfall /etc/turkey-footfall
sudo systemctl daemon-reload
```

<div dir="rtl">

ואז מחק את ה-VM מהקונסולה.

</div>

---

<div dir="rtl">

## 5. 7 שכבות הניתוח החי

מאז fix 2 (‏2026-08), כל אריח בדשבורד יכול להתמזג במקום לזרם של פריימים
מנותחים חיים על אותה מצלמה שהוא מנגן. לחיצה על 🔬, בחירת שכבה, השרת מרים
`LiveSession` ודוחף JPEGs מנותחים ב-~‏1/s; הלקוח מרנדר אותם בתוך האריח.
עד 4 סשנים בגריד (אחד לכל אריח). מעבר בין שכבות באריח פועל **משנה** את
הסשן — הזרם, ה-tracker, וכל המצברים (‏heat, מונים, מונה מחוות) שורדים
את המעבר. ה-VM לא מעורב; הכל רץ ב-`app/live_analysis.py` על מכונת המפעיל.

**הצינור המשותף (‏`LiveSession.run`, כל tick ≈ ‏TICK_TARGET_S = 0.8s):**

</div>

```python
frame = self._grab()                              # (א) שליפת פריים
if frame is None: continue
boxes = self._infer(frame)                        # (ב) YOLO + gates/ROI
self.tracker.update(boxes, now)                   # (ג) BurstTracker
if layer in ("pose", "gestures", "body"):
    self._pose_pass(frame, boxes)                 # (ד) top-down pose, רק כשצריך
faces_list = self._faces_pass(frame) if layer == "faces" else []
self._accumulate(frame.shape, boxes, now)         # (ה) heat + מונה קו
img = self._render(frame, faces_list, layer)      # (ו) ציור השכבה
self._publish(img)                                # (ז) JPEG ללקוח
```

<div dir="rtl">

‏`INFER_LOCK` מסדר בטור כל קריאה למודל בתהליך הזה (‏Ultralytics `predict`
לא thread-safe על מודל משותף). על CPU: סשן חי אחד — 1-2 fps, ארבעה במקביל —
0.3-0.5 fps כל אחד — יורד בחן במקום להפיל את המודל.

**זיהוי המצלמה לאריח הנבחר:** ‏`resolve_cam` קודם מחפש `cam_id` בקטלוג
(‏`app/cameras.py`); אם זה מפספס, קורא את `web/local_grid.json` (נכתב על ידי
תא 32 במחברת) וממפה את ה-slot ל-dict של ‏`kind ∈ {youtube, hls, webcamera24, skyline}`.
המצלמה שהמפעיל רואה בפועל היא בדיוק זו שנפתחת לניתוח.

### 5.1 ‏Paths & speeds — ‏`draw_paths_layer`

מסלולים + תיבות פר-track עם id + shvatzim של km/h. השכבה היחידה שמציגה
תיבות זיהוי לכל המחלקות. היסטוריית ה-track מוגבלת ל-‏`TRAIL_MAX_PTS = 40`
נקודות centroid; קו צבעוני נמתח דרך ה-centroids בצבע ה-track (יציב פר-id).

אומדן קמ"ש לרכבים מגיע מ-`track_stats`:

</div>

```python
real_len = VEHICLE_LENGTH_M.get(cls or "")       # 4.5 לרכב, 12 לאוטובוס, ...
if real_len and speeds:
    exts    = [max(b["x2"]-b["x1"], b["y2"]-b["y1"]) for b in boxes if ...]
    m_per_px = real_len / (sum(exts) / len(exts))     # קנה מידה של פיקסלים
    kmh     = round(sum(speeds)/len(speeds) * m_per_px * 3.6, 1)
```

<div dir="rtl">

טווח שגיאה כן ±30-50% (הרכב לא תמיד מקביל למישור התמונה). הדוח מציג את זה
רק כשלמצלמה יש מסה סטטיסטית מספיקה (≥5 דגימות **וגם** ≥10% מהסבבים).

### 5.2 ‏Pose & skeleton — ‏`draw_pose_layer`

שלדים בלבד, על אנשים קרובים מספיק ל-pose pass — אין תיבות זיהוי, אין
רכבים. מפני שאדם ברחוב תופס 30-120 פיקסלים ו-pose pass על הפריים המלא
ב-640 נותן למודל ~15px של אדם ולא מוצא כלום, השכבה הזאת מריצה **top-down pose**:
לכל תיבת person של הדטקטור, קוצצים את השכנות עם 25% padding ומריצים
YOLOv8n-pose על ה-crop בלבד. ה-pose-person הטוב ביותר בכל crop זוכה
בתיבה הזאת:

</div>

```python
def attach_keypoints_crops(model, frame, boxes,
                           imgsz=256, pad_frac=0.25,
                           min_box_h=40, conf=0.25) -> int:
    persons = [b for b in boxes if b.get("cls") == "person"
               and (b["y2"] - b["y1"]) >= min_box_h]
    crops, offsets = [], []
    for b in persons:
        bw, bh = b["x2"]-b["x1"], b["y2"]-b["y1"]
        px, py = bw*pad_frac, bh*pad_frac
        x1 = max(0, int(b["x1"]-px)); y1 = max(0, int(b["y1"]-py))
        x2 = min(W, int(b["x2"]+px)); y2 = min(H, int(b["y2"]+py))
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((b, x1, y1))
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
    for (b, ox, oy), res in zip(offsets, results):
        if not len(res.boxes): continue
        qi = max(range(len(res.boxes.conf)), key=lambda i: res.boxes.conf[i])
        kps = res.keypoints.data.tolist()[qi]
        b["kps"] = [[x+ox, y+oy, c] for x, y, c in kps]     # חזרה למרחב הפריים
```

<div dir="rtl">

הפלט: 17 מפרקי COCO (אף, עיניים, אוזניים, כתפיים, מרפקים, שורשי כף היד,
אגן, ברכיים, קרסוליים) מצוירים על כל אדם מספיק קרוב. מה שרחוק מדי כתוב
ביושר: "skeletons on 3 of 12 people, rest too far".

### 5.3 ‏Hand gestures — ‏`draw_gestures_layer` + ‏`app/gestures.py`

שלוש מחוות ברמת זרוע על אותם שלדים: ‏`hand_raised` (שורש כף היד מעל
הכתף למשך ≥3 פריימי pose), ‏`both_hands_up` (שני השורשים מעל שתי הכתפיים),
‏`wave` (שורש כף היד עובר את המרפק ≥2 פעמים). הסשן שומר מונה מצטבר
(‏`self.gesture_counts`) כדי שהכיתוב יקרא "session: hand_raised x3, wave x1"
ברגע שמישהו עשה משהו. סצנה ריקה תקרא "no gestures detected right now" —
זה תקין, לא באג.

### 5.4 ‏Body anomalies — ‏`draw_body_layer`

תצוגת חי בסגנון Fall Detection. ‏`label_track` (‏`app/behavior_labels.py`)
מריץ שלוש שכבות פר-‏track ומחזיר תווית **אחת** בדיוק:

1. **‏Pose flags מהשלד** (‏`pose_flags_of`): התיל בין מרכז הכתפיים למרכז
   האגן — הזווית מהאנך; ‏> ‏`FALL_TORSO_DEG = 60°` למשך ‏≥ ‏`POSE_FLAG_MIN_FRAMES = 2`
   פריימים ‏→ ‏`fall_suspect`.
2. **‏Course reversals** (‏`heading_turns`): כמה חזרות אחורה חדות של >90°
   על פני המסלול; ‏≥3 ‏→ ‏`erratic`.
3. **קינמטיקה טהורה:** מהירות ממוצעת, moving fraction, תזוזה נטו לאורך
   המסלול — מקבלים ‏`running` / ‏`walking` / ‏`standing` / ‏`dwelling` /
   ‏`driving` / ‏`parked` / ‏`normal`.

השכבה מציגה **רק** את התוויות ברמת התראה
(‏`BODY_ANOMALY_LABELS = {"fall_suspect", "erratic", "running"}`): תיבה
אדומה או כתומה + overlay של שלד + shvatz של תווית על אנשים שסומנו, HUD
בפינה שמאלית עליונה (‏`persons in view: N, flagged: M`), ובאנר ALERT
אדום בזמן שדגל של `fall`/`erratic` חי.

### 5.5 ‏Face detection — ‏`draw_faces_layer_img` + ‏`app/faces.py`

מלבני זיהוי בלבד (בלי embeddings, בלי מסד נתונים). הדטקטור הוא
**‏YuNet** (‏OpenCV Zoo, ~230KB ONNX), ‏CPU בלבד, ~15ms על פריים
‏960×540. במרחק רחוב פנים לרוב מתחת לרזולוציית הדטקטור — הכיתוב
"no faces at this distance/resolution" הוא כנה.

### 5.6 ‏Heat vision — ‏`draw_heat_layer`

התמונה כולה משתנה כשבוחרים את השכבה הזאת (‏fix 3 requirement).
לא חיישן תרמי — מפת צבע מסוגננת שמונעת על ידי בהירות בתוספת הצטברות
ה-dwell של הסשן:

</div>

```python
gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
signal = gray * 0.72                                      # בסיס: בהירות
peak   = float(np.asarray(grid).max())
if peak > 0:
    dwell  = np.sqrt(grid / peak)                         # sqrt gamma מיישר peaks
    dwell  = cv2.resize(dwell, (W, H), INTER_LINEAR)
    dwell  = cv2.GaussianBlur(dwell, (0,0), sigmaX=max(2, W/96))
    signal = np.clip(signal + dwell*0.55, 0, 1)           # אזורים ששהו בהם רצים חמים יותר
out = cv2.applyColorMap((signal*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
```

<div dir="rtl">

‏`grid` היא מטריצת GRID_H × GRID_W (‏27 × 48) של שניות שהייה פר-cell;
‏`bump_heat` שם באמצע כל tick את משקל הזמן שעבר מה-tick הקודם בכל cell
שמתחת לרגלי אנשים/כלי-רכב. מעבר בין שכבות אחורה ל-heat שומר את הצבירה
— המפה ממשיכה לגדול.

### 5.7 ‏Line crossing — ‏`draw_line_layer` + ‏`update_crossings`

קו סופר (מוגדר פר-מצלמה ב-`app/cameras.py`, או ה-default האופקי
‏`DEFAULT_LINE = [[0.10, 0.62], [0.90, 0.62]]` — רצועת מדרכה). כל שינוי
סימן קפדני של foot point של track על פני הקו הוא אירוע חצייה; כיוון
(‏in/out) הולך לפי סדר הנקודות A→B (צד שלילי → צד חיובי = "in"):

</div>

```python
def update_crossings(side_state, tracks, frame_shape, line, cross):
    H, W = frame_shape[:2]
    for tr in tracks:
        if tr.misses: continue
        fx = (tr.boxes[-1]["x1"] + tr.boxes[-1]["x2"]) / 2
        fy = tr.boxes[-1]["y2"]
        side = _line_side(fx/W, fy/H, line)          # signed cross-product
        if side == 0: continue                       # בדיוק על הקו: אמביוולנטי, דלג
        prev = side_state.get(tr.tid)
        side_state[tr.tid] = side
        if prev is None or prev == 0: continue
        if prev < 0 and side > 0:  cross["in"]  += 1
        elif prev > 0 and side < 0: cross["out"] += 1
```

<div dir="rtl">

‏`side == 0` (נחיתה בדיוק על הקו) מדולג בכוונה — אמביוולנטי והיה גורם ל-jitter
כפול סביב הגבול. הכיתוב מציג "IN x / OUT y (session total)".

</div>

---

<div dir="rtl">

## 6. ניתוח חלון עמוק (`behavior.analyze_window`)

מנוע נפרד, לפי דרישה: תופס חלון ארוך יותר (ברירת מחדל 12 פריימים ב-stride 12
≈ ‏0.5s בין פריימים) ממצלמה אחת, מריץ את אותו זיהוי מדולל פר-פריים, שוזר
אותם לתוך tracks פר-פרט עם `BurstTracker`, ומחזיר פרופיל פר-פרט:

- ‏`path` — מסלול foot-point (מנורמל, JSON-safe)
- ‏`distance / speed` — אורך מסלול, תזוזה נטו, mean/max px/s, ואומדן km/h
  לרכבים (אותו טווח ±30-50% כמו השכבה החיה)
- ‏`moving_frac` — חלק הצעדים שאכן זזו ("עמד שקט 80% מהחלון")
- ‏`direction` — כיוון מסך דומיננטי של התזוזה הנטו
- ‏`zones` — cells של heatmap שביקר בהם (קושר את המסלול למפת ה-dwell
  ארוכת-הטווח)
- ‏`nn_min / mean_px` — השכן הקרוב ביותר מאותה מחלקה על פני החלון (אות
  צפיפות / זוגיות)
- ‏`label` — תווית התנהגות אחת קריאה פר-פרט (מ-`label_track`) עם הראיה
  שלה ב-`label_reasons`
- ‏`gestures` — מחוות ברמת זרוע על פני החלון, במצב pose בלבד

שכבות אופציונליות לפי בקשה:

- ‏`pose=1` — מריץ pose pass top-down, מעשיר את `label` ומאכלס `gestures`.
- ‏`want_faces=1` — זיהוי פנים על הפריים האחרון.
- ‏`lock=auto` או ‏`lock=<track_id>` — מצייר crosshair target lock על אותו
  פרט ומחזיר את ה-offset המנורמל ממרכז הפריים (‏`dx, dy` ∈ ‏`[-0.5, 0.5]` —
  בדיוק האות שבקר pan/tilt היה צורך אם הייתה חומרה מקומית).

צורת CLI:

</div>

```bash
cd /path/to/repo/src
python -m tools.analyze_window --cam taksim_yeni --pose --faces --lock auto
```

<div dir="rtl">

פלט: JPEG מסומן + פרופיל JSON תחת `web/snapshots/behavior/`, ‏LRU של 40
קבצים. כפתור "Analyze window" בדשבורד קורא ל-`POST /api/deep-analyze?cam=<id>`.

</div>

---

<div dir="rtl">

## 7. המחברת — אנליטיקה מקומית

המחברת הראשית היא `turkey_business_activity.ipynb` (בשורש הריפו;
ה-imports מוצאים את `src/app/` אוטומטית). היא משתמשת באותם מודולי
‏`detect_core` ו-`reid` כמו הקולקטור, כך שהמספרים מתאזנים. שנים-עשר סעיפים:

| # | נושא התא | מה הוא עושה |
|---|---|---|
| 0 | ‏Setup | בדיקת תלויות + `MODEL_WEIGHTS = 'yolo26m.pt'` + `load_model` פעם אחת |
| 1 | ‏Camera picker | קטלוג ממוספר על פני כל המדינות; המפעיל בוחר 4 לפי מספר (כולן חייבות לחלוק מדינה); בדיקת חי דוחה בחירות מתות |
| 2 | ‏Single-frame check | תופס פריים אחד מהבחירה הראשונה ומסמן אותו |
| 3 | ‏Footfall time series | דגימה דלילה כל `interval_s`; DataFrame + peak-hour chart |
| 4 | אנומליות + פרופיל peak-hour | ‏Robust rolling z (median + MAD × 1.4826) מסמן אנומליות |
| 5 | ‏Dwell / prolonged stops | ‏burst צפוף + ByteTrack לחלון קצר; per-track dwell + movement |
| 5b | ‏Re-identification | ‏ReidStore על N פריימים; per-class unique / seen-again / regulars (≥3) |
| 6 | ‏Business score | הרכב `volume_median × w0 + linger_rate × w1 + consistency × w2` (data ריק ‏→ `None` כן + note) |
| 7 | ‏Live cloud dashboard | כותב `web/local_grid.json` = המצלמות שנבחרו, מרים `http.server` ב-`localhost:8000`, פותח דפדפן |
| 8 | השוואה בין אתרים | מדרג את המצלמות שנבחרו לפי פעילות |
| 9 | ‏Live summary | סיכום הסשן + גרף מתמיד של footfall/anomaly |
| 10 | כיול דיוק | ‏10a תופס פריימים + חיזויים ב-640/960; 10b תיוג אינטראקטיבי; 10c MAE + bias פר-מצלמה פר-גודל |
| 11 | חיזוי | ‏11a delta fetch מ-Firestore ל-CSV cache; 11b grid של 15-min + eligibility; 11c persistence / seasonal-naive / hour-of-week profile / closed-form ridge; 11d GRU קטן |

### 7.1 המחברת התאומה המקומית

‏`turkey_business_activity_yolov8n.ipynb` (מגיט-מתעלם — אף פעם לא בריפו)
היא עותק ידני של המחברת הראשית עם ‏`MODEL_WEIGHTS = 'yolov8s.pt'` (או
`yolov8n.pt`) כדי שהמפעיל יראה את מה שה-VM רואה על אותם פריימים.
מתועד ב-`.gitignore:54`.

### 7.2 חלק 11 חיזוי — איך מחליט

כל מודל מתוחכם חייב להביס את "אותו זמן אתמול" (‏seasonal-naive) על MAE
לאורך 25% האחרונים של ה-cache (מעולם לא נגעו בו במהלך fitting). הסולם:

- **‏persistence** — ‏`y_{t+h} = y_t`
- **‏seasnaive24** — ‏`y_{t+h} = y_{t+h - 24h}` (בייסליין הכנות)
- **‏profile** — חציון פר-slot local (hour-of-day), או (hour-of-week)
  ברגע שה-cache נושא ≥7 ימים
- **‏ridge** — ‏closed-form numpy ridge על lags (‏1, 2, 3, 4, 96), ‏rolling
  means (‏4, 12), ‏sin/cos של שעת ה-hour-of-day היעד, ‏one-hots פר-מצלמה
- **‏gru** — ‏GRU קטן (hidden 32, ~15k params) קורא 24h ופולט 12h; מתאמן
  על CPU בפחות מדקה; מצטרף לסולם ברגע שה-cache נושא מספיק חלונות

‏`skill = 1 - mae / mae['seasnaive24']` (חיובי = טוב יותר). זרם יציב לגמרי
(‏`seasnaive24 MAE = 0`) נותן ‏`n/a` במקום infinities מטעים.

</div>

---

<div dir="rtl">

## 8. בחירת המודל והפרמטרים

### 8.1 שני דטקטורים, שני זמני-ריצה

‏`yolov8s.pt` על ה-VM (מאז 2026-08-05, ‏`@ 640`); ‏`yolo26m.pt` במחברת
(‏`@ 960`). המחברת היא ההפניה לדיוק; על אותם פריימים חיים: ‏`yolov8n@512`
(תצורת VM לפני 2026-08) מצא 0 אנשים ב-Taksim ו-0 רכבים ב-Sarachane;
‏`yolov8s@960` מצא 5 ו-7; ‏`yolo26m@960` מצא 6 ו-16 + אוטובוס.
תת-הספירות היו התצורה, לא המצלמות.

סולם גדלי המודל (‏COCO):

| מודל | פרמטרים | ‏mAP50 | מעבר CPU 1080p | פסק דין |
|---|---|---|---|---|
| ‏`yolov8n` | 3.2M | 37.3 | ~120ms | תצורת VM ישנה; recall נמוך על מצלמות רחוב רחבות |
| **‏`yolov8s`** | **11.2M** | **44.9** | **~280ms** | **‏VM הנוכחי (@640, מאז 2026-08-05)** |
| ‏`yolov8m` | 25.9M | 50.2 | ~700ms | ‏Peak RSS > 900 MB → oom-kill על e2-micro |
| ‏`yolov8l` | 43.7M | 52.9 | ~1400ms | לא ריאלי על e2-micro |
| ‏`yolo26m` | ~30M | ~50 (NMS-free) | ~800ms CPU | מחברת בלבד |

### 8.2 הכפתורים המרכזיים

| ‏Env / flag | ערך | למה |
|---|---|---|
| ‏`--imgsz 640` | היה 512 עד 2026-08-05 | ‏640 מחזיר אובייקטים קטנים שה-512 pass איבד; ~‏0.39s / pass על ה-VM (× 2 frames × 4 cams = ~3s / 40s round) |
| ‏`--burst 2 --burst-stride 13` | שני פריימים ~0.5s אחד מהשני | חציון הורג flicker של פריים בודד; שתי נקודות מזינות את אומדן המהירות |
| ‏`--interval 40` | שניות | ‏bound על ידי מכסת Firestore (~17k writes/day מתוך 20k) |
| ‏`MemoryHigh=760M / MemoryMax=900M` | מגבלות cgroup | מתאים ל-‏e2-micro של 1 GB עם מרווח |
| ‏`OMP_NUM_THREADS=2` | | תואם למספר ה-vCPUs השיתופיים; default של torch oversubscribes |
| ‏`MALLOC_ARENA_MAX=2` | | ‏glibc per-thread arenas עולים 50-150 MB של RSS על פייתון מרובה threads |
| ‏`DEFAULT_PER_CLASS_CONF` (ב-‏`detect_core.py`) | מפת gate פר-מחלקה | ‏night_adjusted_conf(+0.08) בלילה + boosts פר-מצלמה שנלמדים ב-review |
| ‏`EXTRA_CLASSES` env | ‏`bird, dog, cat, backpack, handbag, suitcase, umbrella` | מזין את מעקב-חפץ-נטוש + שורת "other objects" בדוח |
| ‏`FALL_CHECK=1` env | | ‏person-loiter ‏→ ‏pose pass אחד על ה-crop; ‏torso אופקי ‏→ "Possible FALL" |

### 8.3 כיול confidence פר-מצלמה

‏`tools/calibrate_conf.py` קורא את היסטוריית ה-verdicts של המפעיל, מחשב
מטריצת בלבול פר-`(camera, class)`, ובוחר את הסף הנמוך ביותר שמשיג
**‏precision ≥ 0.90** עם **≥ 30 verdicts** — כותב ל-‏`data/per_camera_conf.json`.
‏`cameras._merge_per_camera_conf()` רץ אחרי ‏`_merge_confidence_boost` וגובר
עליו פר-זוג. זוג שכויל מסומן ‏`source=calibration` בפאנל Learning-proof.

</div>

---

<div dir="rtl">

## 9. אנומליות ודיווח

הקולקטור מריץ סט של שערי אנומליה דטרמיניסטיים פר-סבב ופר-מצלמה. לכל שער
יש טריגר מפורש + חלון debounce כדי שהדיגסט לא יציף.

| שער | טריגר | ‏Debounce |
|---|---|---|
| עומס קיצוני | ספירת אנשים / רכבים מעל סף rolling robust-z | 3 סבבים |
| מצלמה חסומה | הבהירות הממוצעת יורדת מתחת ל-night floor בזמן שהשעון אומר יום | 5 סבבים |
| מצלמה חשוכה | ‏MISSes של דגימה חורגים מתזמון המנוחה-והבדיקה | 3 סבבים |
| ‏Loiter | אותו track של אדם נשאר בתוך תיבה למשך ‏≥ ‏`loiter_s` של המצלמה | תקרה ‏10/יום/מצלמה |
| מבקר חוזר | אותה זהות OSNet נראית במרחק ≥ 1.2× box-scale מהמפגש הקודם | אדם בלבד, ‏≥ 64px floor |
| חפץ נטוש | תיק / מזוודה בלי אדם קרוב-בעל למשך ‏≥ 90s | שער owner-nearby |
| חשד לנפילה | ‏person-loiter + torso אופקי מ-pose pass אחד | תחת ‏`FALL_CHECK=1` |
| ‏Crowd rush | ‏spike פתאומי של speed × density | 2 סבבים |

ראיות אנומליה נלכדות כ-JPEG מסומן תחת ‏`snapshots/anomalies/<cam>/<ts>.jpg`
(‏24h lifecycle ב-Storage). כל אירוע גם נוחת ב-`events/` ב-Firestore
(גם 24h TTL) כדי שרצועת ה-"Events" בדשבורד תוכל להראות אותו חי.

**דיווח:**

- דיגסט ארכיון פעמיים ביום (12:00 + 20:00 שעון ישראל) → תיבת הארכיון של
  הפרויקט בלבד, דרך ‏`digest.timer`. משתמש ב-`tools/daily_digest.py`.
- ‏PDF לפי דרישה מכותרת הדשבורד (אריח פרטי: "Send Report From VM" ←
  `POST /api/send-report`; אריח ציבורי: GitHub Actions workflow dispatch).
  אותו composer של PDF (`tools/report_pdf.py`), sender שונה.

הדוח כן לגבי מסה סטטיסטית: שדות ה-km/h מתפרסמים רק כשיש ‏≥ ‏5 דגימות
מהירות **וגם** ‏≥ 10% מהסבבים נושאים כאלה — אחרת ‏`-`.

</div>

---

<div dir="rtl">

## 10. לולאת הלמידה הפעילה

כל פריים שעבר תיוג הופך לנתוני אימון; לילית (או לפי דרישה) רץ fine-tune
של הראש בלבד ב-GitHub Actions; הראש שקודם נוחת ב-Storage; הקולקטור מבצע
hot-swap בלי restart. הלולאה כולה משתמשת באפס משאבים בתשלום.

### 10.1 תור פריימים uncertainty-first

כל תיבה שנשמרה נושאת ‏`uncertainty ∈ [0,1]` מ-`app/uncertainty.py`:

```
uncertainty = 0.6 * margin + 0.4 * flip_delta
```

- ‏`margin(conf, gate, span=0.25)` — ‏1.0 על ה-gate של המחלקה, יורד לינארית
  ל-0 ב-‏`gate ± span`. זול: כל תיבה כבר יש לה `conf` וה-gate האפקטיבי
  שה-burst רץ איתו.
- ‏`flip_delta` (אופציונלי, על bursts שנדגמו בלבד) — pass נוסף אחד על
  הפריים ההפוך אופקית; ‏per-box IoU-matched conf delta. עולה pass אחד על
  ~‏1-מתוך-5 bursts על מצלמה אחת כאשר ‏`UNCERTAINTY_FLIP=1`.

נשמר: פריימים ‏→ ‏sidecar JSON `metadata.boxes[i].uncertainty`; crops ‏→
‏suffix בשם הקובץ ‏`_uNN` (למשל ‏`..._u87.jpg` = 0.87). ה-‏labels.frame_uncertainty
של ה-review UI מעדיפה את הערך שנשמר; חסר ‏→ fallback ל-margin. הסמפלר
הנאיבי-אקראי של פריימים לא קיים יותר.

### 10.2 ‏BADGE crop sampler

‏`app/badge.py`: ‏k-means++ init עצמי בוחר batch מגוון משוקלל לפי
uncertainty — ‏OSNet embeddings ככיוון, uncertainty כגודל. ‏env switch
‏`REVIEW_SAMPLER=badge|naive` (ברירת מחדל ‏`naive`); דריסה פר-בקשה
‏`?strategy=` על ‏`/api/review-sample`. שורות review מתעדות ‏`sampler` +
‏`uncertainty_at_selection` כדי ש-replay של יעילות naive-vs-BADGE יוכל
לרוץ offline.

### 10.3 ‏Fine-tune של ראש בלבד + שער קידום

‏`tools/train_head.py` עוטף את ‏`yolo detect train` עם backbone קפוא
(‏`freeze=<all-but-head>`), ‏mosaic/mixup כבויים, ‏HSV + flip פעילים, ‏≤ 10
epochs עם early-stop. פולט ‏`data/adapters/<cam>/head_<ts>.pt` — ‏tensors
של Detect head בלבד (~4-6 MB).

‏`tools/promote_adapter.py` מריץ ‏`val` על ה-split הכרונולוגי 90/10 של
ה-exporter לגם ה-baseline וגם ה-candidate; שער:

- ‏`ΔmAP50 ≥ +0.5` נקודות אחוז, **וגם**
- אין מחלקות שיורדות ‏> ‏2pp (‏person / car: ‏0pp — הספירות שמניעות כל דוח).

עבר ‏→ עדכון atomic של המצביע ‏`current` + הוספה ל-‏`history.jsonl`.
נכשל ‏→ שורה ‏`gate.log`. ‏`--rollback` משיב את המצביע הקודם.

תעבורה: התוויות + פריימים המקומיים של המפעיל זורמים ל-Storage דרך
‏`app/training_sync.py` (‏batched, ‏ledger-diffed); ‏GitHub Actions
(‏`.github/workflows/train.yml`) מאמן על runners של ריפו ציבורי חינם;
הראש שקודם נוחת ב-Storage; הקולקטור בודק ‏`current.json` כל 30 סבבים
ומבצע hot-swap במקום.

**‏Fallback byte-identical:** חסר / לא קריא ‏`current.json` ‏→ מודל הבסיס
רץ ללא שינוי. אין adapter פעיל at-rest; הראש נטען לזיכרון רק אם קיים
כזה שקודם ואומת.

### 10.4 עקומת "‏Labels vs quality"

‏`GET /api/al-curve` קורא ‏`history.jsonl` (‏+ מראה Firestore
‏`training_events`, ‏TTL 30d, כתיבה אחת לקידום) והדשבורד מצייר קו
‏Chart.js: ‏labels_total על X, ‏mAP50 על Y, ‏candidates שנדחו באפור,
‏baseline מקווקו. הצ'רט מתמלא אחרי שבוע של ריצות ליליות.

</div>

---

<div dir="rtl">

## 11. הגדרות פרויקט Firebase

הדשבורד ‏onSnapshot חי — כל כתיבה של הקולקטור מגיעה לדפדפן מיד, בלי
polling. ההגדרות, פעם אחת לפרויקט:

**1. יצירת הפרויקט.** ‏`console.firebase.google.com` ‏← ‏Add project ‏←
הפעל Firestore במצב test (יינעל לפני הפריסה הציבורית — ראה שלב 5).

**2. אישורי backend לקולקטור.**

</div>

```bash
# Project settings (גלגל) → Service accounts → Generate new private key.
# שמור את ה-JSON מחוץ ל-git (מגיט-מתעלם אם קרוי firebase-service-account.json)
export FIREBASE_CREDENTIALS=/path/to/serviceAccount.json
pip install firebase-admin
```

<div dir="rtl">

**3. הרצת הקולקטור מול Firebase.**

</div>

```bash
python -m app.collector --backend firebase --interval 20 \
  --only konya_hukumet,kapali_carsi,misir_carsisi,eminonu,istiklal_1
```

<div dir="rtl">

כל סבב כותב doc היסטוריה אחד פר-מצלמה ל-`footfall` ודורס את `latest/{cam_id}`.
הרץ ברשת פתוחה (‏IBB/YouTube חוסמים sandboxes מוגבלים). שמור על החיים
עם systemd / Docker / `nohup`.

**4. ‏Web frontend.** ‏Firebase Console ‏← ‏Project settings ‏← ‏Web app ‏←
העתק את קונפיג ה-SDK. צור ‏`web/firebase-config.js` עם
‏`export const firebaseConfig = {...}`. ואז:

</div>

```bash
cd src/web && python -m http.server 8000     # http://localhost:8000
```

<div dir="rtl">

הדף נרשם עם ‏`onSnapshot` וכל כתיבה של הקולקטור מופיעה מיד.

**5. כללי אבטחה — זה מה שמגן על ה-DB.** מצב test מאפשר לכל אחד באינטרנט
לקרוא **וגם לכתוב**. קונפיג ה-web-SDK הציבורי (‏`apiKey`, ‏`projectId`)
נשלח בדפדפן של כל מבקר ו**אינו** סוד; כללי האבטחה כן.

הכללים המצומצמים חיים ב-`src/firestore.rules`: קריאה ציבורית על קולקציות
הדשבורד (‏`footfall`, ‏`latest`, ‏`reid_stats`, ‏`events`, ‏`config`), כל
כתיבות client נדחות (ה-Admin SDK עוקף כללים, לכן הקולקטור לא נפגע).
פרוס אותם:

</div>

```bash
npm install -g firebase-tools    # פעם אחת
firebase login
# .firebaserc: {"projects":{"default":"<your-project-id>"}}
firebase deploy --only firestore:rules
```

<div dir="rtl">

ואז ב-Firebase Console ← Firestore ← Rules, אמת שהכתיבות מציגות `if false`.

**6. מדיניות TTL.** ‏Firebase Console ‏← ‏Firestore ‏← ‏Time-to-live ‏←
הוסף TTL על ‏`footfall.expire_at` **וגם** ‏`events.expire_at` (שניהם
מתנקים אחרי 24 שעות).

**7. ‏App Check (הגנה מהתעללות / מכסת קריאה).** כללים הופכים את הנתונים
לקריאה-בלבד אך scraper עדיין יכול לשרוף את מכסת הקריאה. ‏App Check דורש
שכל בקשה תישא attestation של reCAPTCHA v3.

‏Firebase Console ‏← ‏App Check ‏← ‏Apps ‏← רשום את ה-web app עם provider
‏reCAPTCHA v3. העתק את ה-site key ל-`web/firebase-config.js` בתור
‏`recaptchaSiteKey`; ‏`web/app.js` מאתחל את App Check אוטומטית ברגע שהוא
מוגדר. כשאתה בטוח, ‏App Check ‏← ‏Firestore ‏← ‏Enforce.

הפעל אכיפה **רק** לאחר שה-site key חי בעמוד — אחרת קריאות מאוכפות
נדחות והדשבורד נהיה ריק.

**8. ‏Rate limit + cost cap.** ‏Firestore Spark tier ‏≈ 20k writes/day.
הקולקטור מדפיס את ספירת הכתיבות היומית הצפויה בהפעלה ומגביל את
‏`--interval` ל-5s. הגדר ‏budget alert ב-Google Cloud ‏← ‏Billing; במסלול
‏Blaze, הגדר גם App Engine daily spending limit — זה ה-hard cap האמיתי.
ל-Firestore אין rate-limit פר-user משלו.

</div>

---

<div dir="rtl">

## 12. ‏Cloudflare Worker ל-IBB

‏`kamerayayin.ibb.istanbul` מסרב לכל טווח IP של Google Cloud (‏HTTP 403)
אבל עונה בנורמליות מכל כתובת אחרת. ‏Cloudflare Worker במסלול החינמי (‏100k
requests/day, העומס שלנו ~‏26k/day) מגלגל את בקשות IBB דרך edge של Cloudflare
— ‏ASN שונה — ומשיב את ‏`taksim_yeni`, ‏`sultanahmet_1_yeni`,
‏`eyup_sultan_yeni`, ‏`beyazit_meydan_yeni`.

מקור ה-worker וקבצי הפריסה נשארים ב-`src/deploy/cloudflare-proxy/`:
‏`worker.js` (ה-fetch handler) ו-`wrangler.toml` (קונפיג הפריסה).

**הגדרה חד-פעמית (~5 דקות):**

</div>

```bash
# 1. חשבון Cloudflare חינמי: https://dash.cloudflare.com/sign-up (ללא כרטיס).

# 2. התקן wrangler (פעם אחת)
npm install -g wrangler
wrangler login                # פותח לשונית דפדפן, אשר גישה

# 3. פרוס את ה-worker
cd src/deploy/cloudflare-proxy
wrangler deploy               # מדפיס https://ibb-proxy.<your-subdomain>.workers.dev

# 4. קבע את הסוד המשותף (כל מחרוזת אקראית)
wrangler secret put PROXY_SECRET
# הצעה: openssl rand -hex 24

# 5. חבר את ה-VM
sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<EOF
IBB_PROXY_URL=https://ibb-proxy.<your-subdomain>.workers.dev
IBB_PROXY_SECRET=<the same secret you just set>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env
sudo systemctl restart collector
```

<div dir="rtl">

**אמת (מה-VM):**

</div>

```bash
cd /opt/turkey-footfall/src && sudo -E .venv/bin/python -m tools.probe_country --country turkey
# want: ארבע מצלמות IBB עוברות ל-LIVE; סה"כ 7/24 live אם גם שכבת ה-YouTube מותקנת

# בדיקה נקודתית של ה-worker
curl -s -H "X-Proxy-Secret: <your secret>" \
  "https://ibb-proxy.<you>.workers.dev/https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" \
  | head -3
# expect: #EXTM3U ...
# 403 מה-worker → הסוד שגוי
# 403 עם body של IBB → Cloudflare עצמו חסום (נדיר)
```

<div dir="rtl">

**מה ה-worker **‏לא** עושה:**

- אין caching שישבור liveness (‏`cf.cacheTtl: 4` תואם לסבב HLS ‏~4s
  segment rotation).
- אין proxying של hosts אחרים (רק ‏`kamerayayin.ibb.istanbul`; כל דבר
  אחר מחזיר 403).
- אין proxying של ‏`tvkur.com` (‏Konya, ‏Otogar ומצלמות תורכיות אחרות של
  ‏webcamera24 — ‏tvkur מגביל גם ‏ASNs residential, ו-edge של Cloudflare
  עומד באותו 403; אלו זקוקות ל-proxy עם IP תורכי ספציפי, מחוץ לתקציב
  ה-‏free-tier).

</div>

---

<div dir="rtl">

## 13. ה-Killswitch של החיוב ב-GCP

מכבה חיוב אוטומטית ב-`turkey-footfall` ברגע שסף תקציב של Cloud Billing
נחצה. זה ההבדל בין "מייל ב-3 בבוקר שאתה מעל תקציב" (התראת תקציב פשוטה)
לבין "השירותים הפסיקו לחייב אותך שלוש דקות אחרי שחצית $5" (זה).

מקור נשאר ב-`src/deploy/gcp-billing-killswitch/`: ‏`main.py` (ה-Cloud
Function), ‏`requirements.txt`.

**מקדימים (פעם אחת):**

1. **הפעל APIs**: ‏Cloud Pub/Sub, ‏Cloud Functions, ‏Cloud Build, ‏Cloud Billing.
2. **צור נושא Pub/Sub** שהתקציב יפרסם אליו:
   ```bash
   gcloud pubsub topics create budget-alerts --project=turkey-footfall
   ```
3. **חבר את הנושא לתקציב:** ‏GCP Console ‏← ‏Billing ‏← ‏Budgets & alerts ‏←
   פתח את התקציב ‏← ‏Manage notifications ‏← ‏Connect a Pub/Sub topic ‏← בחר
   ‏`projects/turkey-footfall/topics/budget-alerts`.
4. **צור את ה-runtime SA:**
   ```bash
   gcloud iam service-accounts create billing-killswitch \
     --display-name "Billing kill-switch runtime" \
     --project=turkey-footfall
   ```
5. **הענק לו שני תפקידים ברמת הפרויקט:**
   ```bash
   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/billing.projectManager

   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/browser
   ```
   ‏`billing.projectManager` יש לו ‏`deleteBillingAssignment` (ה-unlink
   בפועל). ‏`browser` יש לו ‏`resourcemanager.projects.get` לבדיקת
   ה-‏idempotency שרצה לפני ה-unlink.
6. **הפעל בכפייה את ה-Pub/Sub service agent** (דלג רק אם הפרויקט השתמש
   בעבר ב-Pub/Sub push-subscriptions):
   ```bash
   gcloud beta services identity create --service=pubsub.googleapis.com \
     --project=turkey-footfall
   ```

**פרוס:**

</div>

```bash
cd src/deploy/gcp-billing-killswitch
gcloud functions deploy billing-killswitch \
    --gen2 \
    --project=turkey-footfall \
    --region=us-east1 \
    --runtime=python312 \
    --source=. \
    --entry-point=stop_billing \
    --trigger-topic=budget-alerts \
    --set-env-vars=PROJECT_ID=turkey-footfall \
    --service-account=billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
    --memory=256Mi \
    --timeout=60s \
    --max-instances=1

# 2-4 דקות. ואז:
gcloud functions describe billing-killswitch --gen2 --region=us-east1
# want: state: ACTIVE
```

<div dir="rtl">

**הענק ל-trigger SA `run.invoker`** על שירות ה-Cloud Run שעומד מאחורי
ה-function של gen2 (אחרת כל delivery של Pub/Sub נדחה):

</div>

```bash
gcloud functions add-invoker-policy-binding billing-killswitch \
  --gen2 --region=us-east1 \
  --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com
```

<div dir="rtl">

**הוכח שזה עובד:**

</div>

```bash
gcloud pubsub topics publish budget-alerts \
    --message='{"budgetDisplayName":"test","costAmount":999,"budgetAmount":1}'

gcloud functions logs read billing-killswitch --gen2 --region=us-east1 --limit=20
# want: "billing DISABLED on turkey-footfall"
# קונסולת Billing אמורה להראות "Billing account: None"
```

<div dir="rtl">

**הפעל מחדש את החיוב אחרי הבדיקה:** ‏GCP Console ‏← ‏Billing ‏← ‏Link this
project to a billing account.

**מה זה **‏לא** עושה:** לא מוחק resources (ה-VM, נתוני Firestore, דלי
Storage, ה-function עצמה — כולם נשארים; הם פשוט מפסיקים לייצר אירועים
חייבים עד ש-billing account מחובר מחדש); לא נוגע בשירותי free-tier
(ה-e2-micro ממשיך לרוץ); לא אכפת לו איזה סף נחצה (‏Google מפרסמת בכל
סף מוגדר — ‏50/90/100/120%; ה-function מנתקת רק כאשר ‏`costAmount ≥ budgetAmount`).

עלות ה-killswitch עצמו: אפס — הודעת Pub/Sub אחת לחצייה, ‏Cloud Function
invocation אחת (‏2M/חודש חינמיים), ‏Cloud Storage לקוד ה-function (‏Always Free).

</div>

---

<div dir="rtl">

## 14. תקלות נפוצות ותשובות

**"הספירות פר-אריח בדשבורד הן ‏'‏from Ns ago' וההתווית אדומה."**
הקולקטור לא עומד בקצב של ‏`--interval`. הרץ את בטריית ה-health-check
(§4.5); אם הזיכרון בסדר אבל ה-CPU רווי, נפול ל-‏`--weights yolov8n.pt
--imgsz 768` בשורת ה-ExecStart (§3.4).

**"ניתוח חי על מצלמת skyline שנבחרה מקבל 404."** תוקן בסבב האודיט —
‏`_cam_from_slot` עכשיו מטפל ב-slots של ‏`kind="skyline"` מ-
‏`web/local_grid.json`. גרסאות ישנות לפני התיקון נכשלו עם
‏`ValueError("no analyzable stream")`.

**"מייל הדיגסט היומי לא מגיע."** בדוק ‏`/etc/turkey-footfall/digest.env`
קיים עם app password אמיתי של Gmail (‏`sudo cat` כ-root); בדוק
‏`sudo systemctl status digest.timer` לזמן ה-fire הבא; הרץ
‏`sudo systemctl start digest.service` להרצה ידנית מיידית.

**"מצלמות טורקיה תמיד ‏MISS מה-VM."** ‏IBB חוסם גיאוגרפית ‏ASNs של Google
Cloud. או שאתה מחבר את ה-Cloudflare Worker (§12), או שאתה מקבל שהגריד
נופל לתאילנד / יפן / ארה"ב עד ש-IBB משתחררת. ‏`tools/probe_country
--country turkey` מראה מצב חי של כל מצלמת טורקיה.

**"איך אני מוסיף מצלמה חדשה?"** ערוך את ‏`src/app/cameras.py`: בחר
‏`cam_id` יציב, מלא את ה-‏`kind` (‏`hls | youtube | webcamera24 | skyline`),
את ה-URL ואת ה-page, שם התצוגה, וכל ‏`roi` / ‏`roi_exclude` / ‏`line`
overrides. ‏`python -m tools.probe_country --country <c>` מאמת אותה.
לא צריך שינוי ב-VM — ‏`git pull + systemctl restart` תופס את זה.

**"איך לוקחים את ה-VM offline לשבוע?"**
‏`gcloud compute instances stop turkey-collector --zone=us-east1-c` —
Firestore שומרת את 24 השעות האחרונות (‏TTL), הדשבורד מציג את המצב הידוע
האחרון. ‏`gcloud compute instances start ...` כשחוזרים.

**"כמה זה באמת עולה בחודש?"** ‏$0 בהפעלה נורמלית. ה-e2-micro הוא Always
Free; ‏Firestore Spark tier נשארת מתחת ל-20k writes/day; ‏Firebase Storage
נשארת מתחת ל-5 GB חינמיים (~50 MB פעילים עם TTL 24h); ‏egress מ-GCP
ל-Firebase (אותו region) חינמי. ה-killswitch שומר מפני חריגות מפתיעות
(§13).

**"המחברת התאומה חסרה אחרי clone."** במטרה — ראה ‏`.gitignore:54`.
התאומה (‏`turkey_business_activity_yolov8n.ipynb`) מקומית בלבד: העתק את
המחברת הראשית ושנה ‏`MODEL_WEIGHTS = 'yolov8s.pt'`.

</div>

---

<div dir="rtl">

## 15. נספח: החלטות עיצוב שהתקבלו

רטרוספקטיבי. סקירה הנדסית שורה-שורה של ‏SPEC המקורי של הלמידה הפעילה מול
מה שהקוד וסביבת הפרודקשן באמת דרשו. כל verdict אומר מה שרד, מה הוחלף,
ולמה. שמור כאן לקורא שרוצה את ה-‏WHY מאחורי הצורה הנוכחית.

**‏D1 — ‏MC-Dropout uncertainty:** ‏REJECTED. ל-YOLOv8 detection יש
**אפס** ‏`nn.Dropout` modules — variance של T=10 stochastic-pass היה
בדיוק 0. החלפה (‏WS1, שנשלחה): ‏margin מול ה-gate האפקטיבי פר-מחלקה (‏0.6)
+ ‏one-pass flip delta על bursts שנדגמו (‏0.4). אותו contract downstream,
עלות כמעט אפסית.

**‏D2 — ‏LoRA-via-peft:** ‏REPLACED. ‏peft-wrapping של ה-Detect של
‏Ultralytics שובר גישה ל-attributes (‏`stride`, ‏`nc`, ‏`reg_max`), ‏EMA
deep-copies ו-checkpoint pickling. ‏`yolo detect train freeze=<all-but-head>`
native מספק את אותה תוצאה של "‏artifact קטן, ‏backbone קפוא" בלי deps
אקזוטיים. ה-head-only `.pt` (~4-6 MB) הוא ה-"‏adapter".

**‏D3 — ‏COCO export:** ‏SUPERSEDED. ‏`tools/export_labels.py` כבר פולט
‏dataset בפורמט YOLO (פיצול כרונולוגי 90/10, מיפוי verdict כולל
‏relabel + operator-added misses). ‏Ultralytics מתאמן מזה native; ניתן
להוסיף converter ל-COCO מאוחר יותר אם איזה כלי חיצוני יזדקק לו.

**‏D4 — ‏BADGE embeddings:** ‏UPGRADED input. ‏OSNet ONNX עכשיו נשלח
בריפו והוא ה-embedder הדיפולטיבי בכל מקום (‏auto-detected). ‏BADGE מקבל
וקטורים ‏identity-grade של ‏512-d מהיום הראשון; ‏k-means++ init הוא
עצמי (~30 שורות) — ‏sklearn נשאר מחוץ ל-VM.

**‏D5 — ‏Architecture option B (‏split VM / external trainer):** ‏CONFIRMED.
‏`app/pool_sync.py` כבר מזיז artefacts ‏VM↔Storage↔operator עם manifests,
batching, ו-public URLs; ‏round-trip האימון משתמש בו מחדש תחת קידומת
‏`training/`.

**‏D6 — ‏Bit-identical fallback:** ‏TRIVIALLY SATISFIED. ‏Head-overlay
loading פירושו "אין adapter file" = מודל בסיס ללא שינוי — ‏byte-identical.
אין ‏identity-LoRA gymnastics.

**‏D7 — ‏VM resource envelope:** ‏TIGHTENED אחרי אירועי oom-kill חיים.
מעטפה סטנדרטית לכל תוספת VM: ‏`MALLOC_ARENA_MAX=2`, ‏`OMP_NUM_THREADS=2`,
‏`/swapfile` של 2 GB, כל חישוב חדש פר-סבב שומר את הסבב שנמדד מתחת ל-~‏30s,
כל upload path חדש עושה batching (‏≤ 40 objects/pass), ‏Firestore נשארת
מתחת ל-20k writes/day.

**‏D8 — ניסוח mAP:** ‏ADJUSTED. יעדי ‏mAP נמדדים עם ‏Ultralytics ‏`val`
על ה-split הכרונולוגי של ה-exporter. הכותרת של "‏40% פחות labels"
נמדדת ‏naive-vs-BADGE על checkpoints מותאמים-‏label-count בהשוואה
כרונולוגית; ‏two-camera A/B הוא stretch אופציונלי, לא שער.

**‏D9 — ‏Trainer host + ‏adapter retention:** נחתם על ידי המפעיל בקיקאוף.
ברירות מחדל נוכחיות: ‏GitHub Actions לאימון (‏runners של ריפו ציבורי,
חינם); ‏adapter retention = היסטוריה מלאה ב-‏`history.jsonl`.

</div>

---
