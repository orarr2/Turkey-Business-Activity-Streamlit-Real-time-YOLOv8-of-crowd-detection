# מדריך המודל והמערכת (עברית)

מסמך הפרודוקציה של הפרויקט הוא ה-README באנגלית בשורש הריפו. המסמך הזה הוא המדריך המנטלי — למה בחרנו מה שבחרנו, מה כל פרמטר משמעותי עושה, ואיזה מהלכי שיפור פתוחים. כתוב כדי שתוכל לקבל החלטות מיודעות כשיוצא לך רעש בדוח, כשהמודל מפספס, או כשמתחשק לשנות מודל.

## תוכן עניינים

1. [מה המערכת עושה בסך הכל](#1-מה-המערכת-עושה-בסך-הכל)
2. [ארכיטקטורה — איפה כל דבר רץ](#2-ארכיטקטורה--איפה-כל-דבר-רץ)
3. [למה YOLOv8s ומה החלופות](#3-למה-yolov8s-ומה-החלופות)
4. [הפרמטרים המרכזיים ואיזון עלות-דיוק](#4-הפרמטרים-המרכזיים-ואיזון-עלות-דיוק)
5. [הליכה על המחברת תא-אחר-תא](#5-הליכה-על-המחברת-תא-אחר-תא)
6. [מפת src/app/ — מה כל קובץ שם](#6-מפת-srcapp--מה-כל-קובץ-שם)
7. [מפת src/tools/ — הכלים בשורת הפקודה](#7-מפת-srctools--הכלים-בשורת-הפקודה)
8. [מדוע נוצרות "הזיות" ואיך לצמצם אותן](#8-מדוע-נוצרות-הזיות-ואיך-לצמצם-אותן)
9. [מסלולי שיפור מעשיים](#9-מסלולי-שיפור-מעשיים)

---

## 1. מה המערכת עושה בסך הכל

המערכת ממירה 4 מצלמות רחוב פומביות בקוניה (טורקיה) לזרם נתונים כמותי:

- כמה אנשים וכמה כלי רכב יש בכל מצלמה בכל רגע (`footfall`)
- מי חוזר לאותו מקום (`re-identification`)
- מהי חריגה תפעולית — עומס קיצוני, חסימת מצלמה, החשכה, שהייה ממושכת מול המצלמה, מבקר חוזר (`events` / anomalies)
- מהי המהירות האופיינית של כלי הרכב (~קמ"ש)
- לקבל דוח PDF פעמיים ביום למייל, מבלי לפתוח מחשב.

הכל רץ **בחינם**: מודל בקוד פתוח, מכונה וירטואלית של GCP במסלול Always Free, GitHub Actions על ריפו ציבורי (חינם ללא הגבלה), Firebase Spark (חינם עד 20K קריאות/דקה).

---

## 2. ארכיטקטורה — איפה כל דבר רץ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GCP e2-micro VM (1 GB RAM, 24/7)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ collector.py:  loop { grab_frame → YOLO → count → re-ID → events }  │  │
│  │ 4 מצלמות במקביל, ~40 שניות לסבב                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                             ↓  Firestore + Storage                          │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓                                                              ↓
┌──────────────────┐                                     ┌─────────────────────┐
│ Dashboard בדפדפן │                                     │ digest.timer 12/20  │
│ localhost:8000   │                                     │ שולח PDF ל-Gmail   │
│ (המחשב שלך)      │                                     │ (רץ ב-VM)           │
└──────────────────┘                                     └─────────────────────┘
       ↓ תיוגים                                                     ↓ PUSH
┌──────────────────┐          ┌───────────────────────┐    ┌─────────────────┐
│ training_sync    │  ──→     │ GitHub Actions        │    │ Gmail בטלפון    │
│ מעלה לענן        │          │ train-head (חינם)     │    │ התראה         │
└──────────────────┘          │ שער קידום → Storage   │    └─────────────────┘
                              └───────────────────────┘
                                        ↓ hot-load
                              ┌───────────────────────┐
                              │ Collector מושך מודל  │
                              │ חדש בלי restart       │
                              └───────────────────────┘
```

**נקודות עקרון:**

- ה-VM עושה את העבודה הכבדה: יכולה להסתיים בעצמה — התכנון לא מניח שהמחשב שלך פעיל.
- הדשבורד בדפדפן הוא **צרכן בלבד** — קורא ישירות מ-Firestore. אין שרת עורפי מקומי חוץ ממשרת קבצים סטטיים.
- לוקאלית אתה מתייג. תיוגים עולים אוטומטית לענן, ה-GitHub Actions לוקח משם, מאמן, ואם השער עובר — הראש (Detect head) של המודל מוחלף ב-VM בלי restart.

---

## 3. למה YOLOv8s ומה החלופות

### הבחירה

`yolov8s` (small, ~11.2M פרמטרים, 21MB). לא הכי מדויק, לא הכי מהיר — הכי מאוזן על CPU של e2-micro.

**מה נבחן:**

| מודל | פרמטרים | mAP50 (COCO) | זמן פריים @ 512px (CPU) | מסקנה |
|---|---|---|---|---|
| **yolov8n** ‏(nano) | 3.2M | 37.3 | ‎~120ms | ‎recall נמוך על מצלמות רחוב רחבות — מפספס אנשים רחוקים, מבלבל עמודים באנשים |
| **yolov8s** ‏(small) | **11.2M** | **44.9** | **‎~280ms** | **הבחירה** — ‎recall מספיק על אנשים רחוקים, לא מתפוצץ RAM |
| yolov8m ‏(medium) | 25.9M | 50.2 | ‎~700ms | חצי מהסבב שלנו, ‎RAM peak ‎> 900MB → oom-kill |
| yolov8l ‏(large) | 43.7M | 52.9 | ‎~1400ms | לא ריאלי על ‎e2-micro |

**זה נמדד על ה-VM עצמו.** לא על יומרות התיאוריה — על הרעש והזיכרון האמיתיים.

### חלופות שוות שיקול

- **‏yolov9c / yolov10s / yolo11s** — משפחות חדשות של Ultralytics. mAP גבוה ב-‎2-3 נקודות על אותו טווח פרמטרים. שווה החלפה עתידית, אבל עדכון האלגוריתם עצמו לא פותר את ההזיות הקטגוריאליות שאנחנו רואים בפועל ‏(‎תמרור-כ-`person`, עמוד-כ-`motorcycle`) — אלה תלויות דומיין ולא ארכיטקטורה.
- **RT-DETR** של DAMO / RT-DETRv2 — טרנספורמר לזיהוי. מדויק יותר בסצנות עמוסות אבל דורש GPU כדי לרוץ במהירות סבירה.
- **PP-YOLOE-S** של PaddlePaddle — ביצועים דומים ל-yolov8s, קצת קשה יותר לפרוס כי אקוסיסטם קטן יותר. לא שווה החלפה.
- **YOLO-World / OWL-ViT** — Open-vocabulary. אתה נותן טקסט "person carrying bag" והמודל מזהה. יקר מדי ל-CPU. שווה למקרים ספציפיים (למשל חיפוש "אישה בחליפה אדומה").
- **DINOv2 embeddings + linear head** — לזיהוי-מחדש (re-ID) לא לזיהוי אובייקטים. מדויק בהרבה מ-OSNet שאנחנו משתמשים בו, אבל דורש GPU להיות שימושי.

**המלצה מעשית:** להישאר עם yolov8s עוד ~3-6 חודשים. השדרוג הבא הנכון הוא yolo11s (זהה בגודל, mAP+2), לא מודל אחר.

### מה כן משפיע יותר מהחלפת ארכיטקטורה

1. **imgsz**: 512 → 640 שיפור MAE של ~30% על אנשים רחוקים, עלות RAM +30% (במקום 696MB → 900MB). על e2-micro פשוט לא מתאפשר.
2. **fine-tune על 200-500 פריימים מהמצלמות שלך**: זה מה שהלולאה של Reinforcement Learning עושה. השדרוג הכי גדול לדיוק בטווח הקצר.
3. **per-camera confidence gates**: לכל מצלמה יש RAM שונה, גובה שונה, זווית שונה. הרף האחיד 0.30 לא נכון בכל מקום.

---

## 4. הפרמטרים המרכזיים ואיזון עלות-דיוק

הכל בקובץ `src/deploy/gcp-vm/collector.service` (מה שהוא מבצע ב-VM). ההקשר הוא e2-micro: 1 GB RAM, 2 שיתופיות vCPU, /tmp על tmpfs (רם, לא דיסק).

### 4.1 `--imgsz 512`

**מה:** אורך הצלע הארוכה בה YOLO רץ. הפריים המקורי HD ‏(1920×1080) מוקטן ל-512×288 לפני אינפרנס.
**הכרעה:** 512 במקום 640 המקורי.
**מדוע:** 512 קיצוץ ~36% בעומס FLOPs לעומת 640, פחות הפעלה מקסימלית של הזיכרון. מדדתי חי — 640 גרם ל-oom-kill loop אחרי הצטרפות המצלמה החמישית וה-OSNet.
**עלות:** אנשים רחוקים מאוד (< 30px גובה בפריים המקורי) מתפספסים לפעמים. במצלמות הרחוב הרלוונטיות זה 1-2 אחוז מהאנשים.
**איך לחזור ל-640:** אם תעבור ל-e2-small (2GB) העלה ל-640. שנה `collector.service` בשורת ExecStart ו-`sudo systemctl restart collector`.

### 4.2 `--burst 2 --burst-stride 13`

**מה:** בכל סבב איסוף לוקחים ‎burst של 2 פריימים במרווח 13 פריימים בין אחד לשני (‎‎‎~0.5 שניות בקצב ‎25fps).
**מדוע:** רעש של פריים בודד — המודל יכול פתאום "לראות" אדם ואז לא לראות אותו בפריים הבא בגלל תאורה או ‎compression. הרעש הזה מנוטרל דרך `median`: לוקחים את החציון של הספירות מה-burst. עמיד למקריות.
**עלות:** ‎2 hits רצופים על המודל ‎= כפול זמן. `burst 3` הוא הפורמט המקורי, אבל `512 × burst=2` מכניס אותנו לחלון הבטוח של ‎e2-micro.
**גם משמש לזיהוי מהירות:** המרחק בין 2 הפריימים ב-burst הוא הבסיס לחישוב הקמ"ש (‎ראה `detect_core.estimate_speeds`).

### 4.3 `--interval 40` (שם דגל, בפועל bounded by round-time)

**מה:** מרווח מבוקש בין סבבים. בפועל הסבב לוקח את הזמן שלוקח (הרשת + הפענוח + האינפרנס), ואם הוא חורג מ-40 שניות — הסבב הבא מתחיל מיד. באזור 40-50 שניות בפועל על ה-VM שלנו.
**מדוע דווקא 40:** מספיק צפוף לתפוס אירועים סבירים בסצנת רחוב, מספיק דליל שהמכונה לא ניצתת.

### 4.4 `MemoryHigh=760M / MemoryMax=900M` (בקובץ .service)

**מה:** cgroup limits של systemd — אזהרה בגובה 760, kill חזק ב-900.
**מדוע:** ה-e2-micro מבטיח 1024MB אבל בפועל ~950-1000MB זמין (הקרנל לוקח את שלו). Buffer של 100MB חובה.
**מה שקורה אם חורגים:** קרנל הורג את התהליך, systemd מפעיל מחדש. אם קורה בלולאה = יש בעיה.

### 4.5 `OMP_NUM_THREADS=2` ו-`MALLOC_ARENA_MAX=2`

**מה:** מספר ‎threads של ‎OpenMP (‎עבור torch) ו-arenas של glibc malloc.
**מדוע:** למכונה יש 2 vCPU. ‏‎torch כברירת מחדל מנסה להשתמש בכל הליבות, ובמכונה עם 2 בלבד ההשתלטות הזו כרוכה ב-context-switch thrash. הבעיה של ‎malloc arenas עוד יותר עדינה — ‎glibc יוצר ‎arena לכל ‎thread, וכל ‎arena לוקח ‎32-64MB. עם ‎8 threads זה ‎300MB בזבוז ‎RSS.
**זו הבעיה השקטה שהפילה אותנו לפני התיקון.**

### 4.6 `DEFAULT_PER_CLASS_CONF` (`app/detect_core.py`)

**מה:** לכל class יש רף conf נפרד. person=0.30, car=0.30, bus=0.25, train=0.25, truck=0.32, motorcycle=0.28, bicycle=0.28.
**מדוע:** MS COCO מלמד על 80 classes, אבל הדומיין שלנו הוא 7 classes. Detectors מודרניים מכוילים על "מרחק מ-ground truth הכי קרוב"; לכל class יש קליבר טבעי שונה.
**איך זה משתפר לאורך זמן:** confidence_boost.py לומד מ-verdicts שלך ב-Reinforcement Learning tab. "correct" מוריד את הרף לקלאס הזה במצלמה הזו, "wrong" מעלה. השינויים נשמרים ב-`data/confidence_boost.json` ונטענים מחדש כל 10 סבבים.

### 4.7 מהי אנומליה — מה נמדד ומה לא

הוגדר במפורש (בעיצוב חדש): אנומליה = הגדרה תפעולית, לא שיטת סטטיסטיקה.

| Kind | הגדרה מדויקת | ספי הגדרה | Cooldown |
|---|---|---|---|
| `extreme_load` | ‎50+ אנשים בפריים אחד, או weighted vehicle load ≥ 38 (בכפולה: car=1, bus=2.5, truck=2.5, motorcycle=0.5, bicycle=0.3, train=3.0) | 50 / 38 | 30 דק' |
| `camera_obstructed` | תיבת זיהוי אחת תופסת ≥ 50% מהפריים **ו**-conf ≥ 0.45 | 50% / 0.45 | 30 דק' |
| `camera_dark` | luma של הפריים היה ≥ 90 ואז ירד ל-≤ 25 (מעבר, לא ערך מוחלט) | 90 → 25 | 30 דק' |
| `loiter` | אותה entity_id (מ-re-ID) נשארה במקום ≥ 5 דקות (person) או ≥ 15 דקות (רכב). "במקום" = תזוזת מרכז ≤ 60px | 5 / 15 min | לפי entity |
| `returning` | אותה entity_id נראתה שוב אחרי ≥ 5 דקות מהיעלמות, עם similarity ≥ 0.96 (OSNet), רק אם היו ≥ 3 sightings קודמות. לא לרכבת/אוטובוס | 5 min / sim 0.96 | לפי entity |

**חוקים סטטיסטיים (spike/drop) לא מייצרים אנומליות!** הם רק מעדכנים את הפרופיל השעתי (HourlyProfile). זה מהלך מכוון: פרויקטים סטטיסטיים של אנומליות מציפים אלארמים על שגרות שאתה לא רוצה לדעת עליהם.

### 4.8 היקף Firestore — לכן זה נשאר בחינם

מכסה חופשית של Firestore (Spark): 20,000 writes/day, 50,000 reads/day, 1 GiB storage.

השימוש שלנו:
- Footfall: 4 מצלמות × 3 writes לסבב (footfall, latest, events) × ~2200 סבבים/יום ≈ **26,400 writes/day** — קרוב לגבול.
- כדי לרדת מהגבול: `reid_stats` כותב פעם ב-5 סבבים (משתנה `REID_STATS_EVERY_ROUNDS=5`), חוסך 4×2200÷5 = 1760 writes/day. סה"כ סביב **19,000/day** — מתחת ל-20K.

**מזה חשוב לך:** כל תוספת של write לכל סבב = יכולה להרים אותך מעל המכסה. אם תרצה להוסיף מעקב חדש — או שהוא throttled או שהוא מקבץ פעולות.

---

## 5. הליכה על המחברת תא-אחר-תא

הקובץ: `turkey_business_activity.ipynb` בשורש הריפו. סה"כ 35 תאים, מתוכם 16 תאי קוד ו-19 תאי טקסט (`markdown`). כל תת-סעיף כאן ממופה לתא בודד או לצמד תאים צמודים, עם הקוד עצמו, פירוש שורה-שורה של הפרמטרים, ומה חשוב לשים לב לפני שאתה משנה משהו.

> **הצעה לפני שממשיכים:** פתח את המחברת במקביל למסך הזה. כל סעיף פותח עם מספר התא כדי שקל למצוא אותו. תא ריק בסוף (מספר 34) נועד להוספת קוד משלך — אל תשים לב אליו.

---

### תא 0 — כותרת המחברת (markdown)

תוכן: תיאור על-קצה-המזלג של מה שהמחברת עושה, קפיצה מהירה בין הסעיפים. אין קוד — רק אוריינטציה. שים לב לשורה החשובה בסוף התא: **אין צורך בהרשאות Firebase כדי להריץ את המחברת**. הדשבורד קריא ציבורית, האיסוף רץ במקום אחר (ב-VM).

**מה חשוב לזכור:** המחברת היא **צרכן בלבד** של הנתונים — לא כותבת ל-Firestore, לא משנה state בענן, לא צריכה מפתחות. אם תשנה משהו במחברת, זה נשאר אצלך מקומית.

---

### תא 1 — מציאות הרשת (markdown)

הבהרה חד-פעמית שלא ניתן להתעלם ממנה: מצלמות `livestream.ibb.gov.tr` (עיריית איסטנבול) חסומות ל-IPs מחוץ לטורקיה. סנדבוקסים מוגבלים (למשל סביבות ריצה של Colab לפעמים) חוסמות גם הן. מצלמות `tvkur` (השירות הפרטי) פתוחות בכל מקום, מה שהופך אותן לבחירה המעשית.

**מסקנה לפרויקט:** ‎4 מצלמות הפרודקשן שלנו כולן על `tvkur`. אין תלות בגיאוגרפיה של המשתמש. אם תרצה להוסיף מצלמת IBB — היא תרוץ מה-VM ב-GCP ‏(שהוא בארה"ב אבל יש לו מסלול לטורקיה), אבל לא בהכרח מהמחשב הביתי שלך.

---

### תא 2 — כותרת "Setup" (markdown)

כותרת בלבד. סמן חזותי שהחלק הבא מתקין תלויות.

---

### תא 3 — התקנת חבילות (code)

```python
%pip install -q ultralytics opencv-python-headless yt-dlp pandas numpy matplotlib firebase-admin
```

התא הזה מריץ `pip install` בשקט (`-q`) על כל החבילות שהמחברת צריכה:

- `ultralytics` — הספרייה של YOLOv8/v11. כוללת גם את מסגרת ה-tracking (`ByteTrack`).
- `opencv-python-headless` — פענוח וידאו וטיפול בתמונות. הגרסה ה-headless חוסכת ~200MB של תלויות GUI (Qt, GTK) — קריטי בסביבות VM.
- `yt-dlp` — לפתרון של YouTube Live streams לכתובת HLS שאפשר לקרוא ישירות.
- `pandas`, `numpy` — עיבוד נתונים.
- `matplotlib` — גרפים.
- `firebase-admin` — SDK של Google לקריאה/כתיבה מ-Firestore ו-Storage.

**מתי מדלגים על התא הזה:** אחרי הרצה ראשונה, החבילות מותקנות ב-kernel — התא רץ שוב במהירות של תת-שנייה. אין נזק להרצה חוזרת.

**חשוב לשים לב:** ‎`opencv-python-headless` ולא `opencv-python`. אם תתקין את הרגילה (עם GUI) לצד ה-headless, תקבל התנגשות סמלית עמומה שקורסת ב-import. אם קרה — הרץ `pip uninstall opencv-python opencv-python-headless -y` ואז את התא הזה מחדש.

---

### תא 4 — Imports וטעינת המודל (code)

```python
import sys, time, datetime as dt
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the src/ tree so the `app` package imports regardless of whether the
# notebook is run from the project root (default layout) or from inside src/.
_src_dir = Path.cwd() / 'src' if (Path.cwd() / 'src' / 'app').is_dir() else Path.cwd()
sys.path.append(str(_src_dir))
from app.detect_core import load_model, detect_and_count, grab_frame, resolve_youtube, resolve_stream, VEHICLE_NAMES
from app.cameras import CAMERAS, active_cameras, GRID_CAMERAS

DATA_DIR = _src_dir / 'data'; DATA_DIR.mkdir(parents=True, exist_ok=True)
model = load_model(str(_src_dir / 'yolov8n.pt'))
print('cameras available:', list(active_cameras()))
print('dashboard grid (4 live cameras):', GRID_CAMERAS)
```

זה הגרעין של המחברת. בואי נפרק אותו.

**זיהוי מיקום הקבצים.** השורה עם ה-`_src_dir` פתרונה חלקה של בעיה טכנית: תלוי איפה הפעלת את `jupyter` (משורש הפרויקט או מתוך `src/`), נתיב ה-imports משתנה. הקוד בודק אם קיימת `src/app/` יחסית לתיקייה הנוכחית — אם כן, ה-`src_dir` הוא `src/`; אם לא, הוא התיקייה הנוכחית עצמה. אחרי זה מוסיפים אותו ל-`sys.path` כדי ש-`from app.X import Y` יעבוד.

**Imports מ-`app`.** התא מייבא 5 פונקציות ו-3 שמות מקטלוג המצלמות:
- `load_model(path)` — טוען קובץ `.pt` של YOLO.
- `detect_and_count(model, frame)` — מקבל פריים, מחזיר `{'person': N, 'vehicles': N}`.
- `grab_frame(stream_url)` — מוריד פריים בודד מ-HLS/RTSP.
- `resolve_youtube`, `resolve_stream` — ממירים URL תיאורי לכתובת ניגנת.
- `VEHICLE_NAMES` — set של שמות הקלאסים שנחשבים "כלי רכב" ‏(`car`, `bus`, `truck`, `motorcycle`, `bicycle`, `train`).
- `CAMERAS` — מילון עם כל המצלמות הידועות.
- `active_cameras()` — מייצר את המצלמות הפעילות (מסנן catalog-only).
- `GRID_CAMERAS` — רשימה של 4 המצלמות שהדשבורד מציג.

**טעינת המודל.**
```python
model = load_model(str(_src_dir / 'yolov8n.pt'))
```
זו נקודת החלטה קריטית ולעיתים מבלבלת: **המחברת טוענת `yolov8n` ‏(nano), בעוד ה-VM טוען `yolov8s` ‏(small)**. זה מכוון:

- **במחברת** אתה מנתח לוקאלית, לרוב על מחשב אישי בלי GPU. `yolov8n` רץ ב-‎120ms לפריים על CPU של laptop רגיל.
- **ב-VM** רץ קולקטור 24/7 שדורש רקול גבוה יותר על אנשים רחוקים. `yolov8s` נותן ‎mAP+7 נקודות במחיר של פי-2 זמן ריצה.

**איך לשנות ל-`yolov8s` במחברת:** רק שנה את המחרוזת. אם הקובץ `yolov8s.pt` לא קיים בתיקייה, `ultralytics` יוריד אותו בהרצה הראשונה (‎~22MB).

**מה חשוב לשים לב:** ‏`load_model` הוא ‎wrapper דק סביב `ultralytics.YOLO(...)`. הוא גם בודק אם קיים מודל מותאם ב-`data/adapters/current.json` ‏(‎Detect head שקודם בענן), ואם כן — טוען אותו על גבי המשקולות הבסיסיות. אם השדרוג נשמע לא נכון, בטל אותו זמנית עם ‎`ADAPTERS_DISABLE=1` בסביבה.

---

### תא 5 — כותרת "Pick a camera" (markdown)

מסביר איך `resolve_stream` מטפל בארבעה סוגי פרוטוקולים:
- `hls` — פורמט הזרמה סטנדרטי (IBB, tvkur). הכתובת נקראת ישירות.
- `youtube` — הפלטפורמה עם ‎URL של live-stream. `yt-dlp` שולף את ה-HLS הפנימי.
- `skyline` — סטרימינג של `hd-auth.skylinewebcams.com` עם טוקן שמסתובב כל דקה. הקוד גורף את הטוקן מדף ה-HTML.
- `webcamera24` — פורמט מיוחד שמכיל נגן `tvkur` מוטמע. שולפים את ה-ID של הנגן ובונים playlist מחדש.

---

### תא 6 — בחירת המצלמה (code)

```python
CAM_ID = 'konya_hukumet'   # works from any open network via tvkur.
cam = CAMERAS[CAM_ID]
stream_url = resolve_stream(cam)
print(cam['name'], '->', stream_url)
```

- `CAM_ID` — המפתח במילון `CAMERAS`. שנה כאן אם רוצים מצלמה אחרת. אפשרויות מקדימות בהערה: `giresun_gazi`, `otogar_kavsagi`, `kadikoy`.
- `CAMERAS[CAM_ID]` — dict עם `name`, `url`, `kind`, ולפעמים `roi_exclude`, `per_class_conf`, ועוד.
- `resolve_stream(cam)` — מטפל בכל 4 הפרוטוקולים לפי `cam['kind']`.

**מה חשוב לשים לב:**

> אם קיבלת שגיאה ‎`resolve failed`, לרוב זה אחד מהשלושה: המצלמה נפלה זמנית (טוקנים של `skyline` לא מתחדשים בשעות מסוימות), אתה על רשת חסומה, או ה-User-Agent שלך נחסם. נסה `CAM_ID = 'konya_hukumet'` שרץ בכל מצב.

---

### תא 7 — כותרת "Single-frame check" (markdown)

מסביר שלפני שמתחילים לאסוף סדרות זמן, כדאי לבדוק שהזרם באמת עובד ושהמודל רואה משהו.

---

### תא 8 — פריים בודד + ויזואליזציה (code)

```python
frame = grab_frame(stream_url)
if frame is None:
    print(f"WARN: {cam['name']} returned no frame (likely geo-blocked or stream down).")
else:
    print('frame shape:', frame.shape)
    print('counts:', detect_and_count(model, frame))
    res = model.predict(frame, conf=0.35, classes=[0,1,2,3,5,7], verbose=False)[0]
    plt.figure(figsize=(11, 6))
    plt.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(cam['name']); plt.show()
```

מה מתרחש כאן, שלב-שלב:

1. **‏‎`grab_frame`** מוריד פריים בודד. אם הזרם למטה או שהחיבור בעייתי, מחזיר `None`.
2. **‏‎`detect_and_count`** מריץ YOLO ומחזיר מילון ספירות: `{'person': 12, 'vehicles': 5}`. זהו wrapper דק סביב `model.predict()` שמסכם קלאסים לקטגוריות.
3. **‏‎`model.predict(...)`** ‏(ישירות) מחזיר את התוצאה המפורטת עם כל תיבה. ‏‎`conf=0.35` הוא סף הביטחון — כל תיבה עם `conf < 0.35` נזרקת.
4. **‏‎`classes=[0,1,2,3,5,7]`** מגביל את הקלאסים ל-`person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7` ‏(מזהי COCO). למה לא `4` (airplane) או `6` (train)? — הוצאתי בפרודקשן. train כן נשאר במודל השרת, אבל במחברת קליברציה מקלה על החיים.
5. **‏‎`res.plot()`** יוצר תמונה עם bounding boxes מצוירות עליה. ‏‎`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` הוא ה-conversion הידוע — OpenCV משתמש ב-BGR ו-matplotlib ב-RGB.

**דברים חשובים שכדאי לשים לב:**

> **הרפים במחברת גבוהים יותר.** ‏‎`conf=0.35` במחברת, לעומת `conf=0.30` בפרודקשן. הסיבה: EDA חד-פעמית — אתה רוצה תוצאות נקיות שקל להבין. בפרודקשן אתה רוצה recall גבוה כי ה-confidence_boost הלמידה תסנן FPs לאורך זמן.

> **אם `detect_and_count` מחזיר ספירות שנראות נמוכות** — לרוב זה כי ה-conf גבוה מדי בהתאם לזווית של המצלמה שלך. נסה ‏‎`conf=0.20` והשווה.

---

### תא 9 — כותרת "Footfall time series" (markdown)

הרעיון: לשאלה **"כמה אנשים / מתי"** לא צריך פריים כל שנייה. דגימה כל 15-30 שניות מספיקה, וגם מונעת עומס על השרת הצד השני. זה בדיוק מה שה-collector עושה בפרודקשן, רק במסגרת 24/7.

---

### תא 10 — פונקציית `footfall_series` והרצה קצרה (code)

```python
def footfall_series(stream_url, cam_name, interval_s=20, duration_min=1.0):
    rows, t_end = [], time.time() + duration_min * 60
    while time.time() < t_end:
        ts = dt.datetime.now(dt.timezone.utc)
        f = grab_frame(stream_url)
        c = detect_and_count(model, f) if f is not None else {'person': np.nan, 'vehicles': np.nan}
        rows.append({'ts': ts, 'cam': cam_name, 'person': c.get('person'), 'vehicles': c.get('vehicles')})
        print(f"[{ts:%H:%M:%S}] person={c.get('person')} vehicles={c.get('vehicles')}")
        time.sleep(interval_s)
    return pd.DataFrame(rows)

df = footfall_series(stream_url, cam['name'], interval_s=10, duration_min=1.0)
df.to_csv(DATA_DIR / f'footfall_{CAM_ID}.csv', index=False)
df.head()
```

**מבנה הלולאה:**

1. שמור זמן סיום: `t_end = time.time() + duration_min * 60`.
2. בכל איטרציה: תפוס פריים, ספור, שמור, ישן שניות נותרות.
3. אם `grab_frame` החזיר `None` — שים `NaN` בעמודות במקום לזרוק חריגה. זה מאפשר לגרפים להמשיך לעבוד.
4. סוף — מחזיר `DataFrame` ושומר CSV לתיקיית `data/`.

**הפרמטרים שאתה שולט בהם:**

- **‏‎`interval_s`** — כל כמה שניות דוגמים. במחברת 10 שניות (חלון קצר). ב-VM זה 40 (סבב לא מסתיים לפני זה בגלל 4 מצלמות ברצף). ערך אופטימלי לניתוח: 15-30 שניות.
- **‏‎`duration_min`** — משך הכולל. 1 דקה = 6 דגימות ב-‎10s intervals. מספיק לראות טרנד, לא מספיק לאנומליות רציניות. הרם ל-10 דקות אם רוצים אנומליות.

**דברים חשובים שכדאי לשים לב:**

> **דגימה דלילה זה החלטת עלות מודעת.** אנחנו סופרים, לא עוקבים אחר תנועה. פריים כל שנייה יעלה פי 20 בעלות רשת ואינפרנס בלי לשפר את איכות המדד.

> **‏‎`time.sleep(interval_s)`** אינו מדויק — הוא בעצם `interval + processing_time`. אם `grab_frame + detect_and_count` לוקחים 3 שניות, המרווח האמיתי בין דגימות הוא 13 שניות. אם אתה צריך timestamps מדויקים, החלף ל-`while time.time() < next_tick: sleep(0.1)`.

---

### תא 11 — כותרת "Anomalies + peak-hour profile" (markdown)

מסביר שאנומליה מוגדרת כאן כ-**rolling z-score > 2.5** על סדרת ה-footfall: זינוק פתאומי או ירידה חדה. גם: פרופיל שעות שיא — באיזו שעה יש הכי הרבה תנועה.

---

### תא 12 — `flag_anomalies` וגרפים (code)

```python
def flag_anomalies(s, window=12, z=3.5, min_delta=3):
    """Robust rolling z: median + MAD (x1.4826), the same statistic the cloud
    collector uses. Outliers already inside the window inflate a mean/std
    baseline and mask the next event; a median/MAD baseline barely moves."""
    med = s.rolling(window, min_periods=4).median()
    mad = (s - med).abs().rolling(window, min_periods=4).median() * 1.4826
    spread = mad.clip(lower=1.0)   # counts are integers; floor the spread
    robust_z = (s - med) / spread
    return (robust_z.abs() > z) & ((s - med).abs() >= min_delta)
```

**זה הלב הסטטיסטי של המחברת** — ולכן כדאי להבין כל שורה.

**‏‎`window=12`** — גודל חלון גלגול. אם דוגמים כל 20 שניות, זה 4 דקות אחורה. הבסיס נלמד מ-4 הדקות האחרונות. אם החלון קצן מדי (2-3 דגימות), הבסיס מקפץ מדגימה לדגימה — פחות אינפורמטיבי. אם ארוך מדי (100+ דגימות), טרנדים אמיתיים נקברים בבסיס.

**‏‎`min_periods=4`** — מספר הדגימות המינימלי לחשב בסיס. פחות מזה — מחזיר `NaN` ולא מדגיש אנומליה.

**‏‎`med = s.rolling(window).median()`** — החציון של 12 הדגימות האחרונות. אמיד כלפי outliers.

**‏‎`mad = (s - med).abs().rolling(window).median() * 1.4826`** — MAD ‏(‎Median Absolute Deviation) מומר לסטיית תקן שקולה. המקדם `1.4826` הוא ‎`1 / Φ⁻¹(0.75)` בהתפלגות נורמלית — הופך את MAD לאומדן ל-`σ`.

**‏‎`spread = mad.clip(lower=1.0)`** — רצפה של 1. ספירות הן שלמים; אם MAD יוצא 0 (‎N דגימות זהות), הרצפה מונעת חלוקה באפס וגם מונעת רגישות יתר לתזוזות של מדגם 1 אנשים.

**‏‎`robust_z = (s - med) / spread`** — ה-z-score החזק. אם הערך רחוק מהחציון ב-`z` יחידות "MAD-סקאלד", הוא חשוד.

**התנאי הכפול בהחזרה:**
```python
return (robust_z.abs() > z) & ((s - med).abs() >= min_delta)
```
- **‏‎`robust_z > z`** — סטטיסטית חורג.
- **‏‎`(s - med) >= min_delta`** — ההפרש המוחלט לפחות `min_delta` אנשים.

**למה שני התנאים?** בסצנה שקטה עם 1-2 אנשים ממוצע, ‎‎`spread` הוא 1 (הרצפה). אז ‎`z=3.5` נעבור עם הפרש של 3.5 אנשים בלבד — יכול לקרות בטעות. `min_delta` מציב סף מוחלט.

**‏‎`z=3.5`** — כברירת מחדל. בהתפלגות נורמלית זה מתאים לכ-0.05% שוליות. בפועל על סדרות רחוב (heavy tails) זה יוצא 1-2% חריגים אמיתיים. הרם ל-4.0 אם יש רעש; הורד ל-3.0 אם מפספסים.

**מה חשוב לזכור:**

> **זה לא זהה לפרודקשן.** ה-VM לא מסמן אנומליות עם `z-score`. הוא מסמן חמישה סוגי אנומליות תפעוליות מוגדרות: `extreme_load`, `camera_obstructed`, `camera_dark`, `loiter`, `returning`. ה-`z-score` בפרודקשן מעדכן רק את `HourlyProfile` — הוא לא מתריע. זה מכוון (ראה סעיף 4.7).

> **הגרפים שהתא מייצר:** צד שמאל — סדרת הזמן עם anomalies בכתומים. צד ימין — פרופיל שעות: ממוצע אנשים לפי שעת יום. עם דגימה של דקה אחת, פרופיל השעות משמעותי רק אם אתה מריץ את התא מספר פעמים לאורך היום.

---

### תא 13 — כותרת "Dwell-time / prolonged stops" (markdown)

מסביר את המעבר מדגימה דלילה למעקב (tracking): לענות על השאלה "כמה זמן אדם שוהה כאן?" צריך IDs יציבים בין פריימים, שזה עובד רק ב-consecutive frames. לכן — burst קצר צפוף במקום דגימה דלילה.

---

### תא 14 — `dwell_analysis` עם ByteTrack (code)

```python
from app.detect_core import iter_frames, NAME_BY_ID

def dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.35):
    frames_seen = defaultdict(int)
    centroids = defaultdict(list)
    track_cls = {}
    n_frames = int(seconds * target_fps)
    for frame in iter_frames(stream_url, max_frames=n_frames):
        r = model.track(frame, persist=True, conf=conf, classes=[0,2,3,5,7],
                        tracker='bytetrack.yaml', verbose=False)[0]
        if r.boxes.id is not None:
            for box, tid, cl in zip(r.boxes.xywh.cpu().numpy(),
                                    r.boxes.id.int().cpu().tolist(),
                                    r.boxes.cls.int().cpu().tolist()):
                frames_seen[tid] += 1
                centroids[tid].append((float(box[0]), float(box[1])))
                track_cls[tid] = cl

    rows = []
    for tid, n in frames_seen.items():
        pts = np.array(centroids[tid])
        movement = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) > 1 else 0.0
        rows.append({'track_id': tid,
                     'class': NAME_BY_ID.get(track_cls[tid], str(track_cls[tid])),
                     'dwell_s': round(n / target_fps, 1),
                     'movement_px': round(movement, 1)})
    return pd.DataFrame(rows).sort_values('dwell_s', ascending=False) if rows else pd.DataFrame(
        columns=['track_id','class','dwell_s','movement_px'])

dwell = dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.25)
dwell.head(15)
```

זה חלק מפותל יותר — נפרק אותו:

**‏‎`iter_frames(stream_url, max_frames=n)`** — generator שמניב פריימים ברצף. שים לב שהוא מטפל בטריקים של hosts שדורשים headers ‏(‎`tvkur`, IBB, skylinewebcams). ה-`cv2.VideoCapture(url)` הרגיל של OpenCV לא מעביר `Referer` ו-`Origin`, ולכן `iter_frames` מוריד segments של HLS דרך ‎`requests` ומפענח מקומית.

**‏‎`target_fps=3`** — 3 פריימים בשנייה במקום ‎25 של המצלמה. חוסך פי-8 בעומס אינפרנס. עדיין מספיק צפוף שמעקב עובד (‎ByteTrack מוותר על track אחרי 30 פריימים ריקים — ‎‎10 שניות ב-3fps).

**‏‎`model.track(frame, persist=True, ...)`** — מפעיל את המודל עם עקבן. `persist=True` שומר את מצב ה-tracker בין פריימים.

**‏‎`tracker='bytetrack.yaml'`** — קובץ קונפיגורציה של Ultralytics. ‏‎ByteTrack הוא tracker קל: משייך תיבות חדשות ל-tracks קיימים לפי חפיפת IOU + Kalman filter. מאוד זול (~2ms) אבל חלש בזיהוי-מחדש: אם אדם נעלם מאחורי אוטובוס ל-5 שניות, יקבל track_id חדש.

**‏‎`classes=[0,2,3,5,7]`** — מסננים את הקלאסים הרלוונטיים (‎`person, car, motorcycle, bus, truck`).

**‏‎`r.boxes.id`** — יכול להיות `None` אם אף track לא זוהה בפריים. הבדיקה `if r.boxes.id is not None` חובה.

**חישוב תזוזה:**
```python
pts = np.array(centroids[tid])  # רשימת (x,y) של מרכזי-מסה לכל track
movement = np.linalg.norm(pts.max(0) - pts.min(0))
```
המרחק בין הפינה הפנימית הימנית-עליונה של תיבת ההגבלה של הנקודות לבין הפינה הפנימית השמאלית-תחתונה שלה. בפועל: הקוטר המקסימלי של הענן.

**‏‎`dwell_s = n / target_fps`** — משך זמן ה-track בשניות. אם ראינו את `track_id` 45 בין ‎90 פריימים ב-‎3fps → 30 שניות.

**דברים חשובים שכדאי לשים לב:**

> **‏‎`ByteTrack` נותן ID switches לרוב.** אדם שנעלם מאחורי אובייקט אחר במשך 3+ שניות יקבל ID חדש כשיחזור. לכן מספר ה-tracks הייחודיים תמיד גדול יותר ממספר האנשים בפועל. זה לא באג, זה חוסר-מודעות-סמנטי של tracker זול.

> **בפרודקשן החליפו את ByteTrack ב-OSNet re-identification** ‏(‎`app/reid.py`). OSNet מזהה את אותו אדם גם אחרי גאפ ארוך, וגם בין מצלמות. זה יקר יותר (‎‎‎‎~10ms per crop) אבל שווה את זה.

---

### תא 15 — סינון עצירות ממושכות (code)

```python
PERSON_DWELL_S, VEHICLE_DWELL_S, MAX_MOVE_PX = 25, 40, 60
if not dwell.empty:
    is_person = dwell['class'] == 'person'
    stationary = dwell[((is_person & (dwell['dwell_s'] >= PERSON_DWELL_S)) |
                        (~is_person & (dwell['dwell_s'] >= VEHICLE_DWELL_S)))
                       & (dwell['movement_px'] <= MAX_MOVE_PX)]
    print(f"Prolonged stops detected: {len(stationary)}")
    display(stationary)
    linger_rate = (is_person & (dwell['dwell_s'] >= PERSON_DWELL_S)).sum() / max(1, is_person.sum())
    print(f"Linger rate (people who stayed >= {PERSON_DWELL_S}s): {linger_rate:.0%}")
```

**הפרמטרים המרכזיים:**

- **‏‎`PERSON_DWELL_S = 25`** — 25 שניות זה סף שהייה לאדם. אדם רגיל חוצה את שדה-הראייה של המצלמה ב-‎‎5-10 שניות. 25 = הוא עוצר לזמן ממושך.
- **‏‎`VEHICLE_DWELL_S = 40`** — לרכבים הסף גבוה יותר. רכב יכול "לעצור" בכביש דו-מסלולי בזמן אור אדום (‎‎30-60 שניות) בלי לייצר עניין.
- **‏‎`MAX_MOVE_PX = 60`** — תזוזה מקסימלית בפיקסלים לתחילת הענן עד הסוף שלו. ‎60px היא ‎‎~5% מפריים HD ‏(1920px). אם אובייקט זז יותר מזה — הוא לא באמת "עומד".

**‏‎`linger_rate`** — אחוז האנשים שנשארו יותר מ-25 שניות. חשוב ל-`business_score`: קפה או חנות רוצה linger גבוה; משרד או אוטובוס לא.

**מה חשוב לשים לב:**

> **הפרמטרים בפרודקשן שונים.** ב-VM ‏‎`loiter` מוגדר כ-**5 דקות** לאדם ו-**15 דקות** לרכב, לא 25/40 שניות. הפער הוא בגלל דגימה: המחברת דוגמת ב-3fps כי היא בודקת ‎30 שניות; ה-VM דוגם כל 40 שניות אחת, כך שרף של 25 שניות נעלם.

---

### תא 16 — כותרת "Re-identification" (markdown)

מסביר מדוע צריך re-ID: ספירות בפריים מפריזות בגלל ספירות חוזרות של אותו אדם ששוהה. re-ID נותן זהות עקבית לכל אדם/רכב, ששורדת דילוגי פריימים ומצלמות.

**המחברת משתמשת בהתקנה `demo-grade`** של re-ID: ‎HSV histogram + aspect ratio + area. עובד ביום, נשבר בלילה.

---

### תא 17 — Setup של ה-Re-ID (code)

```python
from app.detect_core import load_model, grab_frame, detect_with_boxes, annotate
from app.reid import ReidStore
import cv2, time
import matplotlib.pyplot as plt

REID_DB = str(_src_dir / 'data' / 'reid_notebook.db')
Path(REID_DB).parent.mkdir(parents=True, exist_ok=True)

try:
    reid.close()
except NameError:
    pass

try:
    Path(REID_DB).unlink(missing_ok=True)
    print('reid_notebook.db cleared - fresh demo registry')
except PermissionError:
    print('reid_notebook.db is locked by another process...')

reid = ReidStore(REID_DB, threshold=0.92)
```

**מבנה ה-code הזה מיועד להיות idempotent:** אם הרצת את התא פעם והתא רץ שוב, נשארה חיבור פתוח ל-SQLite. Windows נועל את הקובץ, ואם ננסה למחוק נקבל `PermissionError`. הפתרון: לסגור אקטיבית ואז למחוק.

**‏‎`REID_DB`** — נתיב לקובץ SQLite. `reid_notebook.db` מבודד ממסד הפרודקשן (`reid.db`).

**‏‎`threshold=0.92`** — סף cosine similarity לזיהוי-מחדש. `0.92` מחמיר יחסית. הכי בטוח כשמדובר בפרויקט אמיתי:
- **‏‎0.99** — כמעט תמיד entities נפרדות. שימושי אם אתה חושד ב-over-merging.
- **‏‎0.95** — קונסרבטיבי. פחות false-merges, יותר entities חדשות של אותו אדם.
- **‏‎0.92** — ברירת המחדל. איזון.
- **‏‎0.85** — אגרסיבי. הרבה merges, כולל שגויים.

---

### תא 18 — לולאת דגימה עם Re-ID (code)

```python
N_SAMPLES, INTERVAL_S, CONF = 8, 5, 0.25

rows = []
for i in range(N_SAMPLES):
    f = grab_frame(stream_url)
    if f is None:
        print(f'[{i:02d}] miss'); time.sleep(INTERVAL_S); continue
    counts, boxes = detect_with_boxes(model, f, conf=CONF)
    results = reid.update_from_frame(CAM_ID, f, boxes)
    new = sum(r.is_new for r in results)
    seen_again = len(results) - new
    rows.append({'sample': i, 'person': counts['person'], 'vehicles': counts['vehicles'],
                 'detections': len(boxes), 'new_ids': new, 'seen_again': seen_again})
    time.sleep(INTERVAL_S)
```

**זרימת עבודה בכל איטרציה:**

1. תפוס פריים.
2. `detect_with_boxes` — כמו `detect_and_count` אבל מחזיר גם את התיבות עצמן, לא רק ספירות.
3. `reid.update_from_frame(cam_id, frame, boxes)` — לכל תיבה: מחלץ crop, חושב embedding, מחפש התאמה, מעדכן `sightings` או פותח `entity_id` חדש.
4. **‏‎`is_new`** — האם התיבה הזו יצרה entity חדשה, או הצטרפה לקיימת.
5. שומר ל-`rows`.

**מספרים לזכור:**
- **‏‎`N_SAMPLES=8`** ‏(‎8 דגימות)
- **‏‎`INTERVAL_S=5`** ‏(‎5 שניות בין דגימה לדגימה)
- **‏‎`CONF=0.25`** ‏(‎רף נמוך במיוחד — אנחנו רוצים לתפוס גם אנשים רחוקים כדי לבחון re-ID אגרסיבי)

**מה חשוב לשים לב:**

> **‏‎`INTERVAL_S=5`** צפוף יחסית (‎בפרודקשן ‎40s). מכוון: אנחנו רוצים לראות אדם אחד מספר פעמים בדגימות סמוכות כדי להבין אם ה-re-ID באמת "רואה" אותו כאותו אדם.

> **8 דגימות זה מעט מאוד לגילוי דפוסים.** להערכת איכות re-ID מציאותית, הרם ל-`N_SAMPLES=50, INTERVAL_S=10` — 8 דקות ריצה.

---

### תא 19 — סטטיסטיקות Re-ID (code)

```python
stats = reid.stats(CAM_ID)
print('Total unique entities (this camera):', stats['total_unique'])
print('Total sightings:', stats['total_sightings'])
for cls, s in stats['per_class'].items():
    print(f"  {cls:10s}  unique={s['unique']}  sightings={s['total_sightings']}  "
          f"regulars(>=3)={s['regulars']}")

print('\nTop returning entities:')
for r in reid.top_regulars(CAM_ID, n=10):
    print(f"  #{r['entity_id']:4d}  {r['cls']:8s}  sightings={r['sightings']}  "
          f"first={r['first_seen']}  last={r['last_seen']}")
```

- **‏‎`total_unique`** — מספר entities שנוצרו במסד הזה.
- **‏‎`total_sightings`** — מספר תיבות שקיבלו התאמה + entities חדשות.
- **‏‎`regulars`** — entities עם 3+ sightings. אלה הם ה-"מבקרים חוזרים".
- **‏‎`top_regulars(n=10)`** — 10 ה-entities הכי חוזרים.

**איך לפרש:**

> **‏‎`total_unique < detections`** — הרבה תיבות מוזגו לאותה entity. עדות ל-re-ID עובד. ‏‎`total_unique ≈ detections` — כל תיבה יצרה entity חדשה. או שהמצלמה מלאה באנשים חולפים, או ש-re-ID נשבר.

> **‏‎`regulars > 0`** בסצנה עם דגימה של 40 שניות בלבד זה חשוד מאוד. הרוב אמיתי רק ב-lookout ארוך.

---

### תא 20 — ויזואליזציה של Re-ID (code)

```python
if len(reid_df) >= 3:
    reid_df = reid_df.copy()
    reid_df['returning_rate'] = (reid_df['seen_again'] /
                                 reid_df['detections'].replace(0, np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(reid_df['sample'], reid_df['new_ids'], marker='o', label='new IDs')
    ax[0].plot(reid_df['sample'], reid_df['seen_again'], marker='s', label='seen again')
    ax[0].set_title('Re-ID activity per sample')

    ax[1].plot(reid_df['sample'], reid_df['returning_rate'].fillna(0), marker='o')
    ax[1].set_title('Returning-visitor rate (seen_again / detections)')
```

שני גרפים:
1. **פעילות re-ID לפי דגימה** — כמה חדשים מול חוזרים בכל דגימה.
2. **קצב חוזרים** — היחס `seen_again / detections`. עולה עם הזמן ככל שהאוכלוסייה של entities נצברת.

**מה מצפים לראות:**

> בדגימות ראשונות `new_ids` שולט (‎בסיס ריק). ככל שהדגימות מצטברות, `seen_again` מתחיל לעלות. אם `returning_rate` שטוח באזור 0 לאורך כל 8 הדגימות — יש בעיה: או שרים כל האנשים חלפו בבת-אחת, או ש-re-ID דוחה כל התאמה.

---

### תא 21 — הערת אזהרה על איכות Re-ID (markdown)

תא מדובר בעיקר על הבעיה של Konya Hukumet בלילה: אור נתרן צהוב אחיד הופך את כל האנשים לצללית זהה — HSV histogram נשבר.

**המסלול הפרודוקטיבי:** להחליף את `embed_crop()` בקריאה ל-torchreid + OSNet:

```python
pip install torchreid
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='', device='cpu')
def embed_crop(crop, cls): return extractor([crop])[0].cpu().numpy()
```

זה מה שהפרודקשן שלנו כבר עושה (‎עם `osnet_x0_25` הקטן יותר במקום `x1_0`).

---

### תא 22 — כותרת "Business score" (markdown)

מסביר את הרעיון של ציון ‎0-100 לאיכות עסק: **volume + linger + consistency** במשקולות ניתנות לכיוונון.

- **‏‎Volume** — חציון של אנשים בפריים.
- **‏‎Linger** — אחוז מהאנשים ששוהים 25 שניות+.
- **‏‎Consistency** — הפוך של ‎coefficient of variation ‏(‎`σ/μ`).

---

### תא 23 — פונקציית `business_score` (code)

```python
def business_score(footfall_df, dwell_df, w=(0.5, 0.3, 0.2)):
    people = footfall_df['person'].dropna()
    volume = float(people.median()) if len(people) else 0.0
    cv = float(people.std() / people.mean()) if people.mean() else 1.0
    consistency = max(0.0, 1 - cv)
    is_p = dwell_df['class'] == 'person'
    linger = float((is_p & (dwell_df['dwell_s'] >= 25)).sum() / max(1, is_p.sum())) if len(dwell_df) else 0.0
    vol_norm = min(1.0, volume / 40.0)  # ~40 people/frame treated as 'very busy'
    score = 100 * (w[0]*vol_norm + w[1]*linger + w[2]*consistency)
    return {'volume_median': round(volume,1), 'linger_rate': round(linger,2),
            'consistency': round(consistency,2), 'score_0_100': round(score,1)}
```

**כל מדד נורמל ל-[0,1] לפני צירוף למשקלולת:**

- **‏‎`vol_norm = min(1.0, volume / 40)`** — 40 אנשים לפריים = "עמוס מאוד". אם המצלמה שלך מכסה שדה קטן, שנה את המחלק ל-20.
- **‏‎`linger`** — כבר בין 0 ל-1 (‎יחס).
- **‏‎`consistency = max(0, 1 - CV)`** — CV נמוך → יציב → consistency גבוה. אם `CV > 1` (סטיה גדולה יותר מהממוצע), consistency = 0.

**המשקולות `(0.5, 0.3, 0.2)`:**

- **‏‎`w[0]=0.5`** — נפח 50% מהציון. הכי חשוב.
- **‏‎`w[1]=0.3`** — שהייה 30%.
- **‏‎`w[2]=0.2`** — יציבות 20%.

**איך לכוון לפי סוג עסק:**

| סוג | Volume | Linger | Consistency | הצעה למשקולות |
|---|---|---|---|---|
| קפה / מסעדה | חשוב | **הכי חשוב** | חשוב | `(0.3, 0.5, 0.2)` |
| חנות בגדים | חשוב | **הכי חשוב** | סביר | `(0.4, 0.5, 0.1)` |
| דוכן / קיוסק | **הכי חשוב** | לא חשוב | חשוב | `(0.7, 0.1, 0.2)` |
| חדר כושר | חשוב | חשוב | **הכי חשוב** | `(0.3, 0.3, 0.4)` |

---

### תא 24 — כותרת "Compare with the live cloud dashboard" (markdown)

מסביר שהמחברת עד עכשיו הראתה **ניתוח מקומי** — דקה של דגימה על מצלמה אחת. הדשבורד הענני שרץ מ-VM ב-GCP מציג את הצטברות 24 השעות. השוואה עונה על שאלות: האם הרגע שדגמתי מייצג את היום? האם אני בשיא, בשקט, או ממוצע?

---

### תא 25 — הרצת הדשבורד (code)

```python
DASHBOARD_PORT = 8000
_main = sys.modules['__main__']

if getattr(_main, '_dash_server', None) is None:
    if not port_is_free(DASHBOARD_PORT):
        print(f'Port {DASHBOARD_PORT} already in use...')
        _main._dash_server = 'external'
    else:
        factory = lambda *a, **k: DashboardHandler(*a, directory=str(WEB_DIR), **k)
        http.server.ThreadingHTTPServer.allow_reuse_address = True
        srv = http.server.ThreadingHTTPServer(('', DASHBOARD_PORT), factory)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _main._dash_server = srv

dash_url = f'http://localhost:{DASHBOARD_PORT}/'
display(HTML(f'<p><b>Live dashboard:</b> <a href="{dash_url}" target="_blank">{dash_url}</a></p>'))
display(IFrame(dash_url, width='100%', height=640))
```

**מה קורה:**

1. בודק אם השרת כבר רץ (`_main._dash_server`).
2. אם לא — יוצר `ThreadingHTTPServer` על port 8000 שמשרת קבצים מ-`src/web/`.
3. מפעיל daemon thread כדי שהשרת ירוץ ברקע. הדשבורד קורא מ-Firestore באמצעות ה-SDK של Firebase Web ‏(‎`web/firebase-config.js`).
4. מציג iframe במחברת + קישור לחיצה חיצוני.

**מה חשוב לשים לב:**

> **`ThreadingHTTPServer.allow_reuse_address = True`** — מונע `Address already in use` בהרצה חוזרת אחרי Ctrl+C. Windows יכול להחזיק את ה-port כמה דקות אחרי הסגירה.

> **‏‎`daemon_threads = True`** — סוגר את השרת כשה-kernel של המחברת מפסיק. בלי זה, השרת ממשיך לרוץ עד שסוגרים Jupyter.

---

### תא 26 — כותרת "Compare multiple commercial sites" (markdown)

מסביר את הרעיון: לעבור על כמה מצלמות ולדרג לפי פעילות. בסיס לקבלת החלטה לבחירת מיקום.

---

### תא 27 — דירוג מצלמות (code)

```python
seen = set(); SITES = []
for cid in GRID_CAMERAS + ['kapali_carsi', 'misir_carsisi']:
    if cid not in seen:
        seen.add(cid); SITES.append(cid)

summary = []
for cid in SITES:
    c = CAMERAS.get(cid)
    if not c or not c.get('url'):
        continue
    try:
        url = resolve_stream(c)
    except Exception as e:
        continue
    if grab_frame(url) is None:
        continue
    sdf = footfall_series(url, c['name'], interval_s=10, duration_min=0.5)
    summary.append({'site': c['name'],
                    'median_people': sdf['person'].median(),
                    'max_people': sdf['person'].max()})

pd.DataFrame(summary).sort_values('median_people', ascending=False)
```

**זרימת עבודה:**

1. איחוד רשימה של 4 מצלמות דשבורד + 2 שווקים (‎`kapali_carsi`, `misir_carsisi`). ‏‎`seen` מונע כפילויות.
2. לכל מצלמה: נסה `resolve_stream`, פריים בדיקה, ‎`footfall_series` של 30 שניות (‎`duration_min=0.5`).
3. אספת חציון + מקסימום.
4. מיון לפי חציון (בשלום).

**‏‎`duration_min=0.5, interval_s=10`** — 3 דגימות למצלמה. מיליום זה תיאורטי; בפועל הבדל בין 3 ל-10 אנשים בין מצלמות מספיק כדי לדרג.

**מה חשוב לשים לב:**

> **דילוגים שקטים.** כל `continue` שקוף — אם מצלמה נכשלה, פשוט מדלגים אליה. יכול לקרות ש-3 מתוך 6 מצלמות נופלות והשאר לא — התוצאה לא תכלול אותן. הוסף `print(f'{cid}: skipped')` בכל `continue` אם רוצים לדעת מה קרה.

> **הזמן הזה יקר.** 6 מצלמות × 30 שניות = 3 דקות של המתנה. בזמנים לא-משמעותיים.

---

### תא 28 — כותרת "Live summary" (markdown)

מסביר שהתא הבא מאסף את כל מה שהמחברת ראתה — אנומליות, סטטיסטיקות re-ID, תרשים משולב.

---

### תא 29 — סיכום ריצה (code)

תא ארוך עם `try/except` שאוסף שלושה מקורות:

1. **אנומליות שסומנו** — מכל DataFrame `df` פעיל, לפי `df['anomaly']`.
2. **‏‎סיכום Re-ID** — עם `reid.stats(CAM_ID)` ו-`reid.top_regulars(...)`.
3. **גרף משולב** — footfall + vehicles + anomalies על ציר זמן אחד.

הסיכום גם מדפיס הוראות איך להריץ את הפרודקשן:
```
python -m app.collector --interval 20 --only konya_hukumet,otogar_kavsagi,konya_kulturpark,konya_millet_caddesi
python serve.py        (from the project root)
```

**‏‎`try/except`** ב-outer level — מונע קריסה של המחברת בהרצה חוזרת אם `df` או `reid` לא הוגדרו.

**‏‎`if "df" in dir() and isinstance(df, pd.DataFrame)`** — הגנה בפני שם מוצל.

---

### תא 30 — כותרת "Accuracy calibration" (markdown)

**החלק הכי חשוב במחברת** אם רוצים לשפר את המודל.

מסביר את הזרימה של 3 סוב-תאים:
- **‏‎10a — Capture:** ‎6 פריימים לכל מצלמה, מריצים את המודל ב-‎`imgsz=640` וב-`imgsz=960`, שומרים תוצאות.
- **‏‎10b — Label:** ידני — פותחים כל תמונה, סופרים בעיניים, מקלידים.
- **‏‎10c — Report:** מחשבים ‎MAE ו-bias.

---

### תא 31 — 10a: אסיפת פריימים וסיווגים (code)

```python
CALIB_DIR = DATA_DIR / 'calibration'; CALIB_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_PER_CAM = 6
IMG_SIZES = (640, 960)
CALIB_CONF = 0.30

samples = []
for cam_id in GRID_CAMERAS:
    cam = CAMERAS[cam_id]
    url = resolve_stream(cam)
    for k in range(FRAMES_PER_CAM):
        frames = grab_burst(url, n=1)
        if not frames: continue
        frame = frames[0]
        stem = f'{cam_id}_{k:02d}'
        cv2.imwrite(str(CALIB_DIR / f'{stem}.jpg'), frame)
        entry = {'stem': stem, 'cam_id': cam_id}
        for size in IMG_SIZES:
            counts, _ = detect_with_boxes(model, frame, conf=CALIB_CONF, imgsz=size)
            entry[f'person_{size}']   = counts['person']
            entry[f'vehicles_{size}'] = counts['vehicles']
        cv2.imwrite(str(CALIB_DIR / f'{stem}_annotated.jpg'),
                    annotate(model, frame, conf=CALIB_CONF, imgsz=max(IMG_SIZES)))
        samples.append(entry)
        time.sleep(2)
```

**הפרמטרים:**

- **‏‎`FRAMES_PER_CAM=6`** — 4 מצלמות × 6 = 24 תמונות לתייג. חוקי-אצבע: 20-30 מספיקים כדי לקבל אומדן יציב של MAE.
- **‏‎`IMG_SIZES=(640, 960)`** — משווים ‎"ברירת המחדל הישנה" מול "ברירת המחדל של הקולקטור". שים לב: הקולקטור כרגע רץ ב-**512** (השינוי אחרי המחברת) כי היה `oom-kill`. אם רוצים להשוות ל-512, החלף את הטופל.
- **‏‎`CALIB_CONF=0.30`** — חייב להתאים ל-conf של הקולקטור. אחרת לא משווים תפוחים לתפוחים.

**‏‎`time.sleep(2)`** בין פריימים — כדי לתת לזרם החי לזוז. אחרת תקבל 6 פריימים כמעט זהים מאותה שנייה.

**מה חשוב לשים לב:**

> **תוצאות נשמרות ב-`data/calibration/predictions.json`.** תא 10b קורא משם. אם רצתם 10a והשלמתם 10b בעבר, תא 10a יחליף את הקובץ ותאבד את התיוגים. לפני הרצת 10a שנית — העתק את `labeled.json` בשם אחר.

---

### תא 32 — 10b: תיוג ידני (code)

```python
samples = _json.loads((CALIB_DIR / 'predictions.json').read_text())
labeled = []
for s in samples:
    img = cv2.cvtColor(cv2.imread(str(CALIB_DIR / f"{s['stem']}_annotated.jpg")),
                       cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 7)); plt.imshow(img); plt.axis('off')
    plt.title(f"{s['stem']}  |  model@960: person={s['person_960']} "
              f"vehicles={s['vehicles_960']}")
    plt.show()
    raw = input(f"{s['stem']} true 'people,vehicles' (Enter=skip, q=stop): ").strip()
    if raw.lower() == 'q':
        break
    if not raw:
        continue
    try:
        p_true, v_true = (int(x) for x in raw.replace(' ', '').split(','))
    except ValueError:
        continue
    labeled.append({**s, 'person_true': p_true, 'vehicles_true': v_true})

(CALIB_DIR / 'labeled.json').write_text(_json.dumps(labeled, indent=2))
```

**זרימת עבודה למשתמש:**

1. תמונה נפתחת עם הזיהוי של המודל ב-‎960 (‎מוצג בכותרת).
2. תוצאת המודל בעליון: `model@960: person=12 vehicles=5`.
3. אתה סופר בעיניים ומקליד למשל `10,5` — 10 אנשים, 5 רכבים.
4. `Enter` (שורה ריקה) → דילוג על תמונה.
5. `q` → יציאה מהלולאה עם מה שיש.

**‏‎`input(...)`** — עוצר עד להקלדת ה-user. במחברת Jupyter קלאסית — יש שדה קלט מתחת לתא. ב-Jupyter Lab חדש יותר או VSCode — נפתח דיאלוג.

**מה חשוב לשים לב:**

> **‏‎spelling `people,vehicles`.** ‏‎"אנשים" **פותחים במחרוזת** — הפורמט `12,5` הוא `people=12, vehicles=5`. סדר משמעותי.

> **מה נחשב "רכב"?** ‎`vehicles` = car + bus + truck + motorcycle + bicycle. `train` לא נכלל (‎ב-`VEHICLE_NAMES`). אם רואים רכבת בפריים — התעלם ממנה בספירה.

> **תיוג הוא סובייקטיבי.** אם יש 3 אנשים חצי-חתוכים בקצה הפריים — אתה מחליט אם לספור. יש ערך לתיוג עקבי (‎אותה מדיניות בכל התמונות).

---

### תא 33 — 10c: דוח דיוק (code)

```python
rows = _json.loads((CALIB_DIR / 'labeled.json').read_text())
assert rows, 'No labeled frames yet - run 10b first.'
cal = pd.DataFrame(rows)

overall = []
for size in IMG_SIZES:
    for metric in ('person', 'vehicles'):
        err = cal[f'{metric}_{size}'] - cal[f'{metric}_true']
        overall.append({'imgsz': size, 'metric': metric,
                        'MAE': round(err.abs().mean(), 2),
                        'bias': round(err.mean(), 2),
                        'n': len(cal)})
print('=== overall (all cameras) ===')
print(pd.DataFrame(overall).to_string(index=False))

best = max(IMG_SIZES)
per_cam = []
for cam_id, g in cal.groupby('cam_id'):
    for metric in ('person', 'vehicles'):
        err = g[f'{metric}_{best}'] - g[f'{metric}_true']
        per_cam.append({'cam': cam_id, 'metric': metric,
                        'MAE': round(err.abs().mean(), 2),
                        'bias': round(err.mean(), 2), 'n': len(g)})
```

**המדדים:**

- **‏‎MAE ‏(‎Mean Absolute Error)** = `mean(|prediction - truth|)`. תמיד חיובי. מייצג "רחוק כמה בממוצע".
- **‏‎bias** = `mean(prediction - truth)`. חיובי = overcount, שלילי = undercount.

**איך לפרש דוח לדוגמה:**

```
imgsz  metric      MAE  bias   n
  640  person     2.10 -1.30  24    ← undercount של 1.3 אנשים בממוצע
  640  vehicles   0.95  0.20  24    ← כמעט מדויק
  960  person     1.35 -0.80  24    ← שיפור משמעותי ב-960
  960  vehicles   0.85  0.15  24    ← זהה כמעט ל-640

per camera @ imgsz=960:
  cam                    metric      MAE  bias
  konya_hukumet          person     2.20 -1.80    ← המצלמה הבעייתית
  konya_kulturpark       person     0.80 -0.10    ← מדויקת
  otogar_kavsagi         person     1.10 -0.30
  konya_millet_caddesi   person     1.30 -1.00
```

**מסקנות שיעילות:**

1. **‏‎`konya_hukumet`** — undercount של 1.8 אנשים לפריים. הרם conf של person במצלמה זו: הוסף ‎`"per_class_conf": {"person": 0.25}` (‎מהברירת מחדל 0.30) ל-`app/cameras.py`.
2. **‏‎`konya_kulturpark`** — כבר טוב. אל תיגע.
3. **‏‎MAE@960 < MAE@640** — כדאי להשאיר את הקולקטור ב-‎`imgsz=960`. אבל בפרקטיקה עברנו ל-512 בגלל זיכרון, לא בגלל דיוק.

**מה חשוב לשים לב:**

> **‏‎`n=24`** — 24 תמונות. הוא מדגם מעט. הסטיה של ה-MAE ב-‎n=24 היא בערך ±20% מהערך המוצג. הרם ל-`FRAMES_PER_CAM=15` (60 תמונות) לאמינות טובה יותר.

> **הזוית של מצלמה משנה יותר מהמודל.** אם הזווית מלמעלה מאוד — אנשים רחוקים = כמה פיקסלים ‏(‎`< 20`) → מפספסים. יותר קל לשפר עם fine-tune (‎טאב Reinforcement Learning) מלהעלות `imgsz`.

**זה לא לולאה אוטומטית — זה כלי כיוונון ידני.** הלולאה האוטומטית בפרודקשן היא בטאב Reinforcement Learning בדשבורד — היא מבצעת ‎fine-tune על הראש (‎Detect head) של YOLOv8 באמצעות התיוגים שלך.

---

### תא 34 — תא ריק

מיועד לניסוי הוספת קוד משלך. השאיר ריק כי המחברת רשמית מסתיימת בקליברציה.

---

**סיכום:** המחברת היא כלי EDA — היא מאפשרת לפרק את הצינור לחלקים ולראות מה כל אחד עושה. שינויים בה **לא משפיעים על הפרודקשן**. אם משהו נראה שגוי במחברת, שקול קודם: האם זה שגיאה במחברת עצמה (‎`yolov8n` מפסיד ל-`yolov8s`), או שגיאה בקוד הבסיסי (‎`app/detect_core.py`) — בזה השני הרשמי גם.

---

## 6. מפת src/app/ — מה כל קובץ שם

| קובץ | מה הוא עושה | מתי אתה נוגע בו |
|---|---|---|
| `collector.py` | הלולאה הראשית של ה-VM. סבב אחר סבב על 4 מצלמות: grab_frame → YOLO → count → re-ID → events → write | לעולם. הוא רץ ב-VM. |
| `detect_core.py` | הבסיס: `load_model`, `grab_frame`, `detect_burst`, ‎`estimate_speeds`, `night_adjusted_conf` | כשמשנים פרמטר ברירת מחדל של המודל (imgsz, ‎burst) |
| `cameras.py` | קטלוג המצלמות: URLs, ‎display area, roi_exclude polygons | כשמוסיפים/מסירים מצלמה |
| `reid.py` | ‎Re-ID registry ‎(sqlite): מוסיף/מוצא entities, שומר sightings | כמעט לעולם |
| `reid_embed.py` | טעינת ה-OSNet ONNX + חילוץ embeddings מ-crops | כשמחליפים מודל re-ID |
| `presence.py` | ‎Tracker ל-loitering: לכל entity שומר history של box centroids + זמן | כשמעדכנים גדרי loitering (5min / 15min) |
| `firebase_store.py` | writes ל-Firestore + uploads ל-Storage. TTL 24 שעות | כשמשנים סכמת events |
| `pool_sync.py` | סנכרון pools בין ה-VM לענן (frames, crops, entities) לצורך הדשבורד המקומי | כשמעדכנים אלגוריתם bounded mirror |
| `training_sync.py` | מעלה תיוגי המשתמש (reviews.json) לענן. מופעל אחרי כל Submit בדשבורד | כמעט לעולם |
| `adapters.py` | ‎Head-only fine-tune adapters: save/load/promote/rollback. מוכן ל-hot-swap ב-VM | כשמעדכנים את חוקי השער |
| `labels.py` | ‎ReviewStore: מנהל את verdicts של המשתמש, "uncertainty-first" sample logic | כשמשנים את מדד ה-uncertainty |
| `review_frames.py` | ניהול קבצי מטא של frames מתויגים | כמעט לעולם |
| `live_samples.py` | ניהול הפול של crops חיים לתיוג | כמעט לעולם |
| `entity_gallery.py` | שומר תמונות של entities לגלריה (per-entity crops) | כמעט לעולם |
| `frame_crops.py` | לוגיקה שחוצה frames→crops לחיפוש שכן קרוב | כשמשפרים את מנוע החיפוש |
| `visual_search.py` | ‎kNN search: העלה תמונה → מצא similar crops בפולים | כשמשפרים ranking |
| `confidence_boost.py` | לומד מ-verdicts ומעדכן `per_class_conf` לכל מצלמה. persisted ל-JSON | כשמחליפים את חוק הלמידה |
| `auto_blacklist.py` | לומד מיקומים סטטיים שהמודל טועה בהם, מוסיף polygon החרגה אוטומטית | כשמכוונים סף להכללה |
| `alerts.py` | שולח התראות. Telegram/Slack/Webhook. לא מופעל כרגע. | אם תרצה telegram bot |
| `anomaly_crops.py` | חותך תיבות בודדות מפריימים של אנומליות (לצורך גלריה) | כמעט לעולם |
| `model_metrics.py` | חישוב metrics (precision, recall, F1) מ-verdicts. יוצר גם learning_curve | כשמעדכנים איך מציגים דיוק |
| `dashboard_server.py` | ‎HTTP server מקומי שמשרת את הדשבורד ומטפל ב-review APIs | כשמוסיפים API endpoint |

---

## 7. מפת src/tools/ — הכלים בשורת הפקודה

| קובץ | הרצה | מה הוא עושה |
|---|---|---|
| `export_labels.py` | `python -m tools.export_labels` | יוצר dataset YOLO מ-verdicts. משמש את trainer ב-Actions |
| `train_head.py` | `python -m tools.train_head` | מאמן את ה-Detect head של yolov8s. ‎freeze=<all-but-head>, ‎epochs≤10 |
| `promote_adapter.py` | `python -m tools.promote_adapter --candidate head.pt` | מריץ Val על baseline ועל candidate, מקבל/דוחה לפי שער |
| `fetch_training_data.py` | `python -m tools.fetch_training_data` | מוריד את reviews.json + reviewed frames מ-Storage (רץ ב-Actions) |
| `daily_digest.py` | `python -m tools.daily_digest` | מרכיב ושולח את דוח ה-PDF פעמיים ביום |
| `report_pdf.py` | ‎import — לא CLI | מנוע ה-PDF: pick_group_samples, ‎fetch_snapshots, compose_pdf, ‎draw_box |
| `roi_grid.py` | ‎helper לניפוי polygons | לא רץ ישירות |
| `search_by_image.py` | ‎legacy — לא בשימוש כרגע | ⚠ להסיר בעתיד |
| `setup_reid.sh` | ‎bash — one-shot: מוריד את OSNet ONNX ל-VM | פעם אחת ב-install |

---

## 8. מדוע נוצרות "הזיות" ואיך לצמצם אותן

השדה הזה נקרא **domain shift**: המודל אומן על COCO (סצנות שנצפו על-ידי אנשים), ואנחנו משתמשים בו על סצנות שנצפות מלמעלה במצלמות רחוב (זווית שלא הופיעה ב-training). התוצאה: המודל לפעמים "מוצא" קלאס שהוא כמעט-נכון-אבל-לא-לגמרי.

**דוגמאות ספציפיות שראינו:**

| הזיה | סיבה שורש | מה עשינו |
|---|---|---|
| תמרור באי-התנועה = person | פרופיל אנכי דק דומה לאדם עומד | ‎roi_exclude_class ל-person באזור התמרור (‎cameras.py) |
| עמוד תאורה = motorcycle | המכונה רואה גלגלים גליליים (בפועל: תחתית העמוד) | ‎auto_blacklist למד את המיקום. אחרי 5 יאמצו רצופים באותו pixel range הוא מוסיף polygon בעצמו |
| רכבת מלאה = obstruction | הרכבת תופסת 51% מהפריים בזמן מעבר, ה-conf היה 0.27 (רעש) | ‎OBSTRUCTION_MIN_CONF = 0.45. תיבה שממלאה 50% של המסך חייבת להיות בוודאות של המודל, לא ניחוש |
| אור נתרן בלילה = לא-אדם | כל הצבעים "נבלעים" בצהוב, מודל מפספס מדים אנושיים | ‎night_adjusted_conf(+0.08) — בלילה הרפים גבוהים יותר, פחות false positives בעלות של פחות true positives. שער נוסף: is_night מבוסס שעון (20:00-06:00 טורקיה) לא רק luma. |

**מה עוד עוזר להזיות:**

1. **fine-tune על הדומיין שלך.** 30 פריימים מתויגים = כבר קופצת ה-recall על הקלאסים החלשים (person רחוק, bicycle).
2. **הרם conf לקלאס הבעייתי.** אם bicycle תמיד שגוי → העלה את הרף שלו ל-0.40. אמנם תפספס אופניים אמיתיים, אבל תפסיק לראות רוחות-רפאים.
3. **הוסף roi_exclude polygons** לאזורים שאתה יודע שהם רעש (עמודי תאורה קבועים, שלטים, פנסי דרך).
4. **הרפוזציה של המצלמה עוזרת יותר מהחלפת מודל.** מצלמה נמוכה יותר על אותה כיכר יכולה להוריד את שיעור השווא ב-40%. זה לא בשליטתנו כאן — אלא של tvkur — אבל אם היה אצלך.

---

## 9. מסלולי שיפור מעשיים

מסודר לפי יחס עלות/תועלת. כל שלב עומד לבד — לא חייבים לעבור כולם.

### 9.1 תייג 30-50 פריימים בשבוע (עלות: 20 דקות/שבוע, השפעה: גבוהה)

הטאב "Reinforcement Learning" בדשבורד מציג פריימים שהמודל היה לא בטוח בהם ("uncertainty-first sampling"). כל תיוג עולה מיד לענן. ברגע שיש 20+ פריימים, ה-GitHub Action `train-head` יכול לרוץ, לאמן head חדש, ולהעלות אם הוא עובר את השער.

**זה השדרוג הכי גדול שיש לך במערכת הזו.** אחרי 3-4 סבבי אימון, הזיות ה-domain shift הנפוצות שוככות.

### 9.2 הרם ל-e2-small (עלות: ‎$6/חודש, לצאת ממסלול Free)

‎2 GB RAM (במקום 1), אותו CPU. יאפשר:
- ‎`imgsz=640` בקומפורט → MAE יורד ~30% על אנשים רחוקים
- ‎`--burst 3` במקום 2 → פחות רעש single-frame
- להריץ מודל re-ID גדול יותר (‎OSNet x0_5 במקום x0_25)

הפרויקט יכול להישאר בחינם, אז זה שיפור אופציונלי. הוא שווה אם היה בפועל בעיה של MAE.

### 9.3 החלף ל-yolo11s (עלות: 30 דקות, השפעה: בינונית)

עדכן את `detect_core.load_model()` כדי לטעון ‎yolo11s.pt במקום ‎yolov8s.pt. אותו גודל, mAP+2 נקודות על COCO. עדיין דורש lookahead בקוד שאולי מסתמך על שם ספציפי — לא ראיתי כזה. הריצה תעבוד out-of-the-box.

### 9.4 החלף לזיהוי-מחדש גדול יותר (עלות: 2 שעות, השפעה: גבוהה)

‎OSNet x1_0 (במקום x0_25) פי 4 יותר גדול, mAP +8-10 על re-ID. יעלה ~4MB במקום 907KB, אבל עדיין קטן. ‎‎Runtime לא ישתנה משמעותית (הבקבוק הוא YOLO, לא OSNet).

**אם תרצה קפיצה גדולה יותר:** ‎DINOv2 features ל-re-ID (‎linear probe). אבל דורש GPU כדי לרוץ במהירות ריאלית — לא רלוונטי לפרויקט הזה.

### 9.5 השווה מודלים מקבילים (‎ensemble, עלות: 4 שעות, השפעה: בינונית-גבוהה)

הרעיון: מריצים ‎yolov8s ו-‎yolo11s במקביל, לוקחים רק תיבות שמופיעות בשניהם (עם NMS משותף). הזיות "רק זה ראה את זה" נעלמות. עלות: כפול זמן אינפרנס = לא ריאלי על e2-micro. אבל על ‎e2-small שווה ניסוי.

### 9.6 שפר את הגדרות ה-loiter (עלות: 30 דקות, השפעה: בינונית)

הרפים כרגע: person 5 min, vehicle 15 min. במצלמות שראית "אנומליות שווא" — הרם ל-‎person 8 min. גם: שקול הוספת דרישה "‎movement ≤ 40px" (הגבל תזוזה יותר) כדי לפסול הליכה איטית באזור.

---

## נספח — טבלת קבצי VM שקוראים מ-Firestore

לצורך הבנת שרשרת התלות:

| הרכיב | קורא מ- | כותב ל- |
|---|---|---|
| `collector.py` | `config/grid` (סטטית), `reid.db` (מקומי) | `footfall`, `latest`, `events`, `reid_stats` |
| `pool_sync.py` (VM side) | קבצים מקומיים ב-`web/snapshots/` | Storage `review_sync/` |
| `pool_sync.py` (Local side) | Storage `review_sync/manifest.json` | קבצים מקומיים ב-`web/snapshots/` |
| `training_sync.py` (Local) | `data/reviews.json` | Storage `training/` |
| `train-head` (Actions) | Storage `training/` | Storage `training/adapters/`, `history.jsonl` |
| `daily_digest.py` (VM) | Firestore + Storage `training/history.jsonl` | Gmail SMTP |

---

*הקובץ הזה חי. אם השתנה משהו בקוד ולא כאן — לא סמכת עליו כמסמך הגמור.*
