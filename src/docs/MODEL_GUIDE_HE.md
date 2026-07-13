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
10. [מה יש בפועל ב-VM — צלילה מלאה](#10-מה-יש-בפועל-ב-vm--צלילה-מלאה)

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

## 10. מה יש בפועל ב-VM — צלילה מלאה

הסעיפים הקודמים דיברו על ה-VM כ"קופסה שחורה שמריצה את הקוד". הסעיף הזה פותח את הקופסה: מה בדיוק יושב על הדיסק, איזה תהליכים רצים, מה כל אחד עושה שנייה-אחר-שנייה, ואיפה יכולות להיות תקלות. הוא ארוך בכוונה — זה החלק הכי אטום למי שלא בנה את זה.

### 10.1 המכונה עצמה

```
Provider: Google Cloud Platform (GCP)
Machine type: e2-micro (Always Free tier)
CPU: 2 vCPU shared (0.25 vCPU guaranteed, ~1 vCPU burst)
RAM: 1024 MB total (~950 MB usable after kernel)
Disk: 30 GB SSD (Standard persistent disk)
OS: Debian 12 (bookworm)
Location: us-east1-c (Virginia)
Public IP: static, one address
Cost: $0/month (במסלול Always Free — 1 מכונת e2-micro חינם לצמיתות)
```

**מה חשוב לשים לב:**

> **"‎0.25 vCPU guaranteed"** אומר שברירת המחדל היא רבע ליבה. יש ‎burst עד ~‎1 vCPU כשיש חלון פנוי בשרת המשותף. בפועל אנחנו מקבלים ‎~2 vCPU רוב הזמן, אבל **לא בזמנים עמוסים של GCP** — כל סבב לפעמים לוקח 25% יותר זמן ללא סיבה נראית לעין. זה נורמלי.

> **‏‎1 GB RAM זה מעט מאוד** למודל של ‎‎11.2M פרמטרים ‎+ 4 זרמי HLS פעילים ‎+ OSNet. הוספתי swap של 2 GB לקובץ חלופי כדי לספוג peaks. ‏‎swap הוא איטי אבל טוב יותר מ-OOM kill.

> **הסיבה שבחרנו us-east1-c ולא אזור קרוב לטורקיה:** ‏‎Always Free מוגבל למקומות מסוימים ‏(`us-west1`, `us-central1`, `us-east1`). לזרמים מטורקיה יש RTT של ~150ms, אבל זה לא חשוב לנו — אנחנו לא מנתחים תנועה בזמן אמת, רק דוגמים מדי 40 שניות.

### 10.2 מבנה התיקיות ב-VM

בהתחברות ל-VM (‎`gcloud compute ssh turkey-collector`), זה מה שרואים:

```
/opt/turkey-footfall/         ← הריפו נמצא כאן, קלון מ-main
├── src/
│   ├── app/                  ← כל קבצי הפייתון
│   ├── tools/                ← כלי CLI
│   ├── data/
│   │   ├── reid.db           ← SQLite של OSNet embeddings
│   │   ├── osnet_x0_25_msmt17.onnx   ← מודל re-ID
│   │   ├── confidence_boost.json     ← ‎learned per-cam gates
│   │   ├── blacklist_auto.json       ← polygons של auto-blacklist
│   │   ├── reviews.json              ← לא רלוונטי ב-VM (עולה מהמחשב שלך)
│   │   ├── adapters/                 ← ‎head-only artifacts
│   │   │   ├── current.json          ← מצביע לראש שפעיל
│   │   │   ├── history.jsonl         ← יומן קידומים/דחיות
│   │   │   └── head_run7.pt          ← קובץ הראש עצמו
│   │   └── training_pull/            ← ריק ב-VM (רק ב-Actions)
│   ├── web/
│   │   ├── snapshots/                ← המאגר החי — נקודת האמת
│   │   │   ├── review_frames/        ← פריימים לתיוג (‎LRU 500)
│   │   │   ├── live_samples/         ← crops לחיפוש (‎LRU 1000)
│   │   │   ├── entities/             ← ‎per-entity crops (‎LRU 400)
│   │   │   └── anomalies/            ← צילומים של אנומליות (‎24h TTL)
│   │   └── firebase-config.js        ← מפתחות ציבוריים של Firebase Web
│   ├── .venv/                        ← ‎Python virtualenv (‎~2 GB)
│   └── deploy/gcp-vm/
│       ├── install.sh                ← סקריפט התקנה
│       ├── collector.service         ← יחידת systemd של הקולקטור
│       └── digest.service, digest.timer   ← יחידות של דוח יומי
├── yolov8s.pt                        ← משקולות הבסיס (‎~22 MB)

/etc/turkey-footfall/            ← קונפיגורציה מוגנת (‎root only)
├── serviceAccount.json          ← מפתח Firebase Admin SDK (0400)
└── digest.env                   ← ‎GMAIL_USER + APP_PASSWORD (0600)

/etc/systemd/system/
├── collector.service            ← ‎symlinked to /opt/turkey-footfall
├── digest.service               ← ‎symlinked
└── digest.timer                 ← ‎symlinked

/var/log/journal/                ← יומני systemd של הקולקטור וה-digest

/var/swap                        ← ‎2 GB swap file
```

**מה חשוב לזכור:**

> **‏‎`/etc/turkey-footfall/`** מוגן ‎`root:root`. הסודות שם. אף אחד מלבד ‎root לא יכול לקרוא. הקולקטור רץ כ-root כדי לקרוא את המפתח, זו סיבה טובה למה הוא רץ כ-root ולא כמשתמש רגיל.

> **‏‎`.venv/` הוא 2 GB.** רוב זה ‎torch ‎(‎~1.2 GB) ו-`ultralytics` (~200 MB). אם תרצה לחסוך זיכרון דיסק, הרץ `pip install --no-cache-dir` — הסקריפט כבר עושה זה. אין דרך למחוק torch — הוא נדרש.

### 10.3 install.sh — התקנה שלב-אחר-שלב

הסקריפט ב-`src/deploy/gcp-vm/install.sh` בנוי להיות **‎idempotent** — אפשר להריץ אותו שוב ושוב ללא נזק. הרצה שנייה = git pull + restart. שלב-אחר-שלב:

```bash
#!/usr/bin/env bash
set -euo pipefail   # יציאה על כל שגיאה
```

**שלב 1 — התקנת חבילות מערכת:**
```bash
apt-get update -qq
apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ca-certificates curl \
    fonts-dejavu-core
```
- **‏‎`git`** — למשוך את הקוד.
- **‏‎`python3-venv`** — כדי ליצור סביבה מבודדת.
- **‏‎`ffmpeg`** — קידוד/פענוח וידאו (‎backend של OpenCV).
- **‏‎`libgl1`, `libglib2.0-0`, וכו'** — תלויות של ‎`opencv-python-headless`. אפילו ‎headless צריך ‎libGL.
- **‏‎`fonts-dejavu-core`** — נדרש ל-PDF (‎דוח יומי).

**שלב 2 — קלון של הריפו:**
```bash
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  git -C "${INSTALL_DIR}" fetch --depth 1 origin "${REPO_BRANCH}"
  git -C "${INSTALL_DIR}" reset --hard "origin/${REPO_BRANCH}"
else
  git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi
```
**‏‎`--depth 1`** — shallow clone. חוסך ~50 MB של היסטוריה. אין צורך בהיסטוריה ב-VM.

**‏‎`git reset --hard`** — אם יש קונפליקטים מקומיים, מוחקים. סקריפט הפצה, לא סביבת פיתוח.

**שלב 3 — הכנת virtualenv:**
```bash
cd "${INSTALL_DIR}/src"
python3 -m venv .venv
export TMPDIR=/var/tmp   # /tmp על tmpfs (RAM); pip צריך מקום אמיתי
.venv/bin/pip install --no-cache-dir -r requirements.txt
```

**‏‎`TMPDIR=/var/tmp`** — קריטי. ‏‎`/tmp` על ‎Debian ‎12 הוא ‎tmpfs (‎‎‎RAM). התקנה של ‎torch דורשת פריקת ‎800 MB זמניים — יקרוס את ה-RAM.

**‏‎`--no-cache-dir`** — לא שומר את קבצי ההתקנה. חוסך עוד ‎800 MB.

**שלב 4 — הבאת מפתח Firebase:**
```bash
mkdir -p "${CFG_DIR}"
gcloud secrets versions access latest --secret="${SECRET_NAME}" > "${SA_PATH}"
chown root:root "${SA_PATH}"
chmod 0400 "${SA_PATH}"
```
- **‏‎`Secret Manager`** של GCP — הדרך הבטוחה לאחסן את המפתח. לא בגיט, לא ב-`.env`.
- **‏‎`chmod 0400`** — קריאה בלבד לבעלים (‎root). אף אחד אחר לא יכול לקרוא.

**שלב 5 — זיהוי אוטומטי של Storage bucket:**
```bash
PROJECT_ID=$(python3 -c "import json; print(json.load(open('${SA_PATH}'))['project_id'])")
STORAGE_BUCKET=""
for candidate in "${PROJECT_ID}.firebasestorage.app" "${PROJECT_ID}.appspot.com"; do
  if gcloud storage buckets describe "gs://${candidate}" >/dev/null 2>&1; then
    STORAGE_BUCKET="${candidate}"; break
  fi
done
```
פרויקטים שנוצרו ‎לפני אוקטובר ‎2024 קיבלו ‎bucket בפורמט `<project>.appspot.com`; חדשים יותר קיבלו `<project>.firebasestorage.app`. הסקריפט בודק את שניהם ובוחר את הקיים.

**שלב 6 — התקנת יחידות systemd:**
```bash
sed -e "s|__STORAGE_BUCKET__|${STORAGE_BUCKET}|g" \
    -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
    -e "s|__SA_PATH__|${SA_PATH}|g" \
    "${UNIT_SRC}" > "${UNIT_DEST}"
systemctl daemon-reload
systemctl enable --now collector.service
```
- **‏‎`sed`** ממיר את הפלייסהולדרים ב-`collector.service` לערכים הממשיים.
- **‏‎`daemon-reload`** — אומר ל-systemd לקרוא את הקבצים החדשים.
- **‏‎`enable --now`** — מפעיל מיד ‎+ מגדיר להפעלה אוטומטית ב-boot.

### 10.4 collector.service — פירוק שורה-שורה

הקובץ ב-`src/deploy/gcp-vm/collector.service`:

```systemd
[Unit]
Description=Turkey Business Activity — footfall collector
After=network-online.target
Wants=network-online.target
```

**‏‎`After=network-online.target`** — לא להתחיל עד שיש רשת. חיוני — הקולקטור נכשל מיד אם אין רשת.

**‏‎`Wants=`** ‎(ולא `Requires=`) — עדיפות רכה. אם הרשת לא עולה, הקולקטור בכל זאת ינסה.

```systemd
[Service]
Type=simple
WorkingDirectory=/opt/turkey-footfall/src
Environment=PYTHONUNBUFFERED=1
```

**‏‎`PYTHONUNBUFFERED=1`** — כל `print()` נכתב מיד ל-journal. בלי זה, פייתון מבפר ולוגים מגיעים באיחור של דקות.

```systemd
Environment=OMP_NUM_THREADS=2
Environment=MALLOC_ARENA_MAX=2
```
כפי שהוסבר בסעיף ‎4.5 — קריטי לזיכרון.

```systemd
Environment=FIREBASE_CREDENTIALS=/etc/turkey-footfall/serviceAccount.json
Environment=FIREBASE_STORAGE_BUCKET=turkey-footfall.firebasestorage.app
Environment=REID_MODEL=/opt/turkey-footfall/src/data/osnet_x0_25_msmt17.onnx
```

**‏‎`REID_MODEL`** — אם הקובץ קיים, יש OSNet. אם לא, נופלים ל-HSV histogram. הקולקטור לא נופל בלי הקובץ הזה.

```systemd
ExecStart=/opt/turkey-footfall/src/.venv/bin/python -m app.collector \
    --interval 40 --imgsz 512 --burst 2 --burst-stride 13
```

הפרמטרים המוסברים בסעיף 4.

```systemd
Restart=always
RestartSec=15
```

**‏‎`Restart=always`** — אם התהליך נפל מכל סיבה, ‎systemd מפעיל שוב אחרי ‎15 שניות. **חשוב:** ‏‎לא ‎`on-failure` — גם אם התהליך יצא ב-0 (‎‎`sys.exit()`), הוא מופעל שוב. הרעיון: הקולקטור לעולם לא צריך להיעצר לבד.

```systemd
MemoryHigh=760M
MemoryMax=900M
```

הגבלות ‎cgroup:
- **‏‎`MemoryHigh=760M`** — הגעה לזה גורמת לקרנל להאט את הקצאת הזיכרון (‎`memory pressure`).
- **‏‎`MemoryMax=900M`** — הגעה לזה = OOM kill. שים לב שזה הרבה פחות מ-1024 MB של המכונה — 100 MB buffer לקרנל וסקריפטים אחרים.

```systemd
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**‏‎`journal`** — הכל הולך ל-`journalctl`. אפשר לצפות בזמן אמת עם `journalctl -u collector -f`.

**‏‎`multi-user.target`** — הפעלה ב-boot רגיל (לא recovery mode).

### 10.5 digest.service + digest.timer

הזוג הזה מפעיל את הדוח היומי פעמיים ביום. **‏‎`digest.service`** הוא `Type=oneshot` — רץ פעם אחת, מסתיים, לא רץ שוב עד שה-timer מפעיל אותו.

`digest.timer`:
```systemd
[Timer]
OnCalendar=*-*-* 12:00:00 Asia/Jerusalem
OnCalendar=*-*-* 20:00:00 Asia/Jerusalem
Persistent=true

[Install]
WantedBy=timers.target
```

**‏‎`OnCalendar` × 2** — שתי שעות שיריצו את השירות.

**‏‎`Persistent=true`** — אם ה-VM היה כבוי בשעת ה-fire הצפויה (למשל rebooot לתחזוקה), הריצה תבוצע מיד כשהוא חוזר. חסכה לי כמה דוחות מפוספסים.

**‏‎`Asia/Jerusalem`** — ‏‎systemd מטפל בעצמו בשעון קיץ. לא צריך לשנות שום דבר פעמיים בשנה.

**איך לבדוק שהטיימר חמוש:**
```bash
systemctl list-timers digest.timer --no-pager
# NEXT: Mon 2026-07-14 12:00:00 IDT
# LAST: Mon 2026-07-14 08:00:00 IDT   (16 hours ago)
```

### 10.6 main() של הקולקטור — נקודת הכניסה

```python
# app/collector.py, בערך שורה 1500
def main():
    args = _parse_args()
    firebase = FirebaseStore()   # קורא את FIREBASE_CREDENTIALS
    model = load_model()          # yolov8s + adapter overlay אם קיים
    reid = ReidStore() if REID_MODEL else None
    trackers = _init_trackers()   # לכל מצלמה: rolling window
    profile = HourlyProfile()     # פרופיל שעות פעילות
    presence = PresenceTracker()  # לזיהוי loitering
    alerts = AlertSink() if not args.no_alerts else None
    
    print("Restoring analysis state from Firestore...")
    _restore_state(firebase, trackers, profile, ...)
    firebase.write_grid_config(...)
    
    print(f"Collector started. {len(GRID_SLOTS)} slot(s):")
    # מדפיס תחזית מכסת Firestore
    # ...
    
    while True:  # הלולאה הראשית
        ...
```

**שלבי אתחול (‎לפני הלולאה הראשית):**

1. **קריאת ארגומנטים** ‎(`--interval`, ‎`--imgsz`, וכו').
2. **‏‎`FirebaseStore()`** — טוען את ה-`serviceAccount.json`, מאמת מול Google.
3. **‏‎`load_model()`** — טוען את `yolov8s.pt`. אם יש `data/adapters/current.json`, מכסה את משקולות ה-Detect head במשקולות מהראש המקודם.
4. **‏‎`ReidStore()`** — פותח את `data/reid.db` (SQLite), טוען את OSNet.
5. **‏‎`_restore_state`** — קורא את מצב האנומליות מ-Firestore ‏(‎`config/analysis_state`). אם היה restart, לא מאבדים את החלון הגלגול.
6. **‏‎`write_grid_config`** — כותב את מצב 4 המצלמות ל-`config/grid` כדי שהדשבורד ידע איזו מצלמה בכל slot.

**חשוב לשים לב:**

> **‏‎`_restore_state`** מבטיח שאחרי restart, החישוב הרובוסטי ‎(‎median + MAD) של אנומליות ממשיך מאיפה שנעצר. בלי זה, כל restart היה גורם ל"אנומליות שווא" ב-‎12 הדגימות הראשונות עד שהחלון הגלגול מתמלא שוב.

> **הודעת המזל שהקולקטור מדפיס בסטארט:** ‎`~19,008 Firestore writes/day projected (free tier ~20,000).`. זה הצ'ק שלנו — מתחת ל-‎20K. אם השורה מראה מספר גבוה יותר, הצוואר בקבוק לא רחוק.

### 10.7 הלולאה הראשית — סבב-אחר-סבב

הלולאה בערך משורה 1677 של `collector.py`:

```python
_REVIEW_RELOAD_EVERY_ROUNDS = 10
_STATIC_LEARN_EVERY_ROUNDS = 90
_ADAPTER_CHECK_EVERY_ROUNDS = 30
_round_counter = 0

while True:
    round_start = time.time()
    _round_counter += 1
    
    # (1) טעינה מחדש של overrides — כל 10 סבבים
    if _round_counter % _REVIEW_RELOAD_EVERY_ROUNDS == 0:
        reload_review_overrides()   # קריאה מחדש של confidence_boost.json, blacklist_auto.json
    
    # (2) בדיקת ראש מקודם חדש — כל 30 סבבים (‎~20 דקות)
    if _round_counter % _ADAPTER_CHECK_EVERY_ROUNDS == 0:
        fetched = adapters.refresh_from_storage(firebase.storage)
        if fetched:
            n = adapters.apply_current(model)
            print(f"  * adapter: hot-loaded {fetched} ({n} head tensors)")
    
    # (3) למידת mishaps סטטיים — כל 90 סבבים (‎~1 שעה)
    if _round_counter % _STATIC_LEARN_EVERY_ROUNDS == 0:
        added = learn_from_positions(...)   # אם המודל מסמן אותו pixel range שוב ושוב → auto-blacklist
    
    # (4) הליבה — לכל אחד מ-4 המצלמות
    for slot in GRID_SLOTS:
        picker = pickers[slot["slot_id"]]
        cam_id = picker.current_cam()
        ok = sample_slot(model, slot, cam_id, firebase, reid=reid, ...)
        changed = picker.record_result(ok)
        if changed is not None:
            firebase.write_grid_config(...)   # ‎fallback: primary נפל, עברנו ל-fallback
    
    # (5) העלאת pool ל-Storage
    stats = pool_sync.sync_up(firebase, snapshots_root, reid_db_path)
    
    # (6) שמירת פרופילי שעות — כל 15 דקות
    if time.time() - last_profile_save >= profile_save_s:
        _persist_profiles()
        last_profile_save = time.time()
    
    # (7) גיזום re-ID — כל 6 שעות
    if time.time() - last_reid_prune >= reid_prune_s:
        reid.prune_stale(older_than_hours=48)
        last_reid_prune = time.time()
    
    # (8) המתנה עד לסבב הבא
    elapsed = time.time() - round_start
    if elapsed < args.interval:
        time.sleep(args.interval - elapsed)
```

**דברים חשובים שכדאי לשים לב:**

> **הפעולות התזמניות (‎`_round_counter % N == 0`) הן עדינות** — הן לא נכנסות ל-critical path של דגימת המצלמות. הן רצות **לפני** ה-`for slot in GRID_SLOTS`, אז אם משהו נכשל בהן — הקולקטור עדיין דוגם.

> **‏‎`sync_up`** ב-(5) הוא ‎incremental. אם לא היה שינוי מקומי, הוא רק משווה dict — משנייה עולה מעט. רק כשיש קבצים חדשים הוא מעלה.

> **‏‎`_persist_profiles`** שומר את חלון האנומליות ל-Firestore. אם ה-VM ייעצר או ריסטרט, הריסטור ב-`_restore_state` יקרא את זה חזרה.

> **‏‎`reid.prune_stale(48h)`** מוחק entities שלא נראו 48 שעות. חשוב — בלי זה מסד ה-SQLite יגדל בלי גבול.

### 10.8 sample_slot — צלילה למצלמה בודדת

זה החלק הכי דחוס. מתחיל בערך שורה 989:

```python
def sample_slot(model, slot, cam_id, firebase, reid, conf, ...):
    cam = CAMERAS[cam_id]
    
    # (1) הבאת burst
    frames = grab_burst(cam["url"], n=burst, stride=burst_stride)
    if not frames:
        return False   # ‎picker יזכור, אחרי X כישלונות יעבור ל-fallback
    
    # (2) לילה?
    luma = mean_gray(frames[0])
    night = is_night(luma, now_utc)  # שעון-מבוסס
    
    # (3) YOLO על כל פריים ב-burst
    gates = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
    if night:
        gates = night_adjusted_conf(gates)  # ‎+0.08 לכל class
    
    counts, boxes, frame, burst_dbg = detect_burst(
        model, frames, per_class_conf=gates, burst_stride=burst_stride)
    
    # (4) speed estimation
    speeds = burst_dbg.pop("speeds", None)
    
    # (5) re-ID
    if reid:
        results = reid.update_from_frame(cam_id, frame, boxes)
        new_ids = [r.entity_id for r in results if r.is_new]
        seen_again = [r.entity_id for r in results if not r.is_new]
    
    # (6) אנומליות סצנה (‎`extreme_load`, `camera_obstructed`, `camera_dark`)
    scene = check_scene_anomalies(cam_id, counts, boxes, frame.shape, luma)
    
    # (7) loitering ‎+ returning
    for r in reid_results:
        if r.sightings >= 3:
            entity_gallery.save_sighting(...)   # לגלריה
        if presence.observe(...):
            _handle_loiter(...)   # ‎emit event
        if r.is_returning():
            _handle_returning(...)  # ‎emit event
    
    # (8) כתיבה ל-Firestore
    record = {"ts": ts, "cam_id": cam_id, "cam_name": cam["name"],
              "person": counts["person"], "vehicles": counts["vehicles"],
              "counts": counts, "ok": ok,
              "new_entities": len(new_ids), "seen_entities": len(seen_again),
              "is_anomaly": bool(scene)}
    if speeds:
        record["speeds"] = summarize_speeds(speeds)
    firebase.write(slot["slot_id"], record)
    
    return True
```

**מה חשוב שיהיה ברור:**

> **הסדר של השלבים משנה!** ‏‎(3) YOLO **חייב לבוא לפני** (4) speed ולפני (5) re-ID. ‏‎speed צריך תיבות; re-ID צריך תיבות; שניהם דורשים את הפלט של YOLO.

> **‏‎`night_adjusted_conf`** ב-(3) הוא הגנה מיצירתית: בלילה יש יותר רעש (‎מצלמות משפרות ISO, יש הרבה sensor noise). מרים את כל הרפים ב-0.08 כדי לקבל פחות FPs. הפסד: כמה true positives נופלים. איזון נכון.

> **‏‎`(8) firebase.write`** הוא כתיבה **אחת** ל-Firestore לכל סבב. עם 4 מצלמות = 4 writes/round. אם round הוא 40s → 8,640 writes/day = בטוח מתחת ל-20K.

### 10.9 מודל הזיכרון — cgroups, malloc, swap

זה החלק שהפיל אותנו פעמיים לפני שהתייצב.

**cgroup limits מ-`collector.service`:**
- **‏‎`MemoryHigh=760M`** — כשהתהליך חוצה, קרנל מפעיל ‎memory pressure. ההקצאות הופכות איטיות, יותר עבודה עוברת ל-swap.
- **‏‎`MemoryMax=900M`** — כשהתהליך חוצה, OOM kill. ‏‎systemd מפעיל שוב אחרי 15 שניות.

**מדוע דווקא 900MB ולא ‎‎1000MB?** ‏‎למה שנשאר ‎`~50MB` בערך. הקרנל של Debian לוקח קצת (‎‎‎`vm.min_free_kbytes ≈ 40MB`), יש `journald` שרץ, יש ‎`gcloud agent` — הם צריכים לחיות.

**malloc arenas — הבעיה השקטה:**

‏‎glibc יוצר ‎arena אחד לכל thread. כל arena לוקח 32-64MB של virtual memory (RSS מעל 20MB כשמתמלאים). ‏‎torch יוצר ~8 threads → 8 arenas → ‎160MB בזבוז מרגע ההפעלה.

**התיקון:** ‏‎`MALLOC_ARENA_MAX=2` — מגביל ל-2 arenas. RSS יורד ב-100MB לפחות. **זה מה שהציל אותנו מ-oom-kill loop.**

**‏‎swap file:**

הוספתי ב-VM:
```bash
sudo fallocate -l 2G /var/swap
sudo chmod 600 /var/swap
sudo mkswap /var/swap
sudo swapon /var/swap
# הוספתי גם ל-/etc/fstab לצורך persist
```

**‏‎`2GB swap` על ‎1GB RAM זה יחס גדול** — הרעיון: לספוג peaks נדירים. הקולקטור **לא** צריך להשתמש ב-swap בשגרה (swap איטי פי 100 מ-RAM). אם רואים ‎swap הרבה בשימוש (`free -h`), הבעיה עמוקה יותר.

**חשוב לזכור:**

> **‏‎`journalctl -u collector | grep "oom-killed"`** — אם רואים את זה, ‎OOM חזר. או שהוספתי מצלמה חמישית ולא שיניתי פרמטרים, או שהראש המקודם גדול מהצפוי.

> **‏‎`free -h`** ב-SSH נותן תמונת מצב מיידית. אם ‎`used > 950MB` באופן קבוע — במצוקה. אם ‎`Swap used` > ‎0 — הקולקטור נגע ב-swap.

### 10.10 הרשת — HLS decoding + iter_frames

מצלמות tvkur לא עובדות עם `cv2.VideoCapture(url)` הרגיל. הסיבה: הן דורשות ‎`Referer` ו-`Origin` headers, ו-OpenCV לא מאפשר להעביר headers.

**הפתרון: `iter_frames` ב-`app/detect_core.py`:**
```python
def iter_frames(stream_url, max_frames=1):
    # HTTP GET של ה-master playlist עם headers מותאמים
    playlist = requests.get(stream_url, headers=BROWSER_HEADERS).text
    # פרסינג של m3u8 → רשימת segments
    segments = _parse_m3u8(playlist)
    # הורדת segments כ-bytes
    for seg_url in segments[-3:]:  # 3 האחרונים
        seg_bytes = requests.get(seg_url, headers=BROWSER_HEADERS).content
        # פענוח מקומי עם ffmpeg → פריימים
        frames = _decode_h264_bytes(seg_bytes)
        for f in frames:
            yield f
```

**זרימת עבודה:**

1. **‏‎`GET` על ה-playlist** — `.m3u8` הוא קובץ טקסט שמפרט segments. תפוגה ~30 שניות.
2. **‏‎‏parsing** — שלושת ה-segments האחרונים = 30 השניות הכי עדכניות.
3. **‏‎`GET` על segments** — כל אחד ~1MB, מכיל ~5 שניות של HD בקידוד H.264.
4. **פענוח מקומי** — ‎`ffmpeg` מקבל את ה-bytes, מחזיר frames כ-numpy arrays.
5. **`yield`** — generator, חוסך זיכרון.

**מה חשוב לזכור:**

> **‏‎`BROWSER_HEADERS`** ‏(‎ב-‎`detect_core.py`) מגדירים ‎`User-Agent`, ‎`Referer=https://tvkur.com/`, ‎`Origin=https://tvkur.com`. בלי זה, tvkur מחזיר 403. הם שינו את המדיניות שלהם פעמיים בזמן החיים של הפרויקט — אם הזרם נפל, בדוק אם צריך לעדכן.

> **הטעימות של resolution:** ‏‎`_pick_variant` בוחר את הגרסה הכי קרובה ל-‎`imgsz` אבל לא נמוך יותר. אם הפריים המקורי הוא 1080p וה-imgsz שלנו 512 — נבחר את הגרסה של 720p ‏(‎‎`~1.5x`) כדי לא לפתוח 1080p בכלל. חוסך ‎H.264 decode של ‎~50%.

### 10.11 Firestore — מבנה ומכסה

**הקולקציות שהקולקטור כותב אליהן:**

| קולקציה | מה יש שם | קצב כתיבה |
|---|---|---|
| `footfall` | דגימות היסטוריות של ספירות ‎+ עוצמות ‎+ אנומליות ‎(‎‎`is_anomaly`) | 4 מצלמות/סבב = ~‎8,640/day |
| `latest` | דוגמה **אחת בלבד** לכל מצלמה — הכי חדשה. `set` (‎לא `add`) | 4 מצלמות/סבב = ~‎8,640/day |
| `events` | אירועים תפעוליים (loiter, returning, obstructed, וכו') | ‎5-30/day בממוצע |
| `reid_stats` | סטטיסטיקות re-ID ‏(entities, sightings) | ‎4 מצלמות / **5 סבבים** = ‎‎1,728/day |
| `config` | ‎`grid` (מצלמות פעילות), ‎`analysis_state` (‎checkpoint) | ‎‎~4/day |

**סה"כ:** ‎~19,000 writes/day.

**‏‎`Firestore Spark plan` (‎חינם):** ‎‎20,000 writes/day, 50,000 reads/day, 1 GiB storage.

**איך שומרים מרחוק מהרף:**

- **‏‎`REID_STATS_EVERY_ROUNDS = 5`** — כותבים סטטיסטיקות re-ID רק כל 5 סבבים. חוסך ‎6,912 writes/day. **זה מה שמכניס אותנו מתחת ל-20K.**
- **‏‎`footfall`** משתמש ב-‎`add()` (מוסיף) ולא ב-`set()`, כי כל דגימה חדשה = מסמך חדש. ‏‎TTL של 24 שעות (‎דרך שדה `expire_at`) מגביל את סה"כ הדוקומנטים לכ-‎‎‎34,560 (‎‎4 cams × 8,640/day).
- **‏‎`latest`** משתמש ב-`set()` על מסמך קבוע — 1 מסמך למצלמה. **תמיד** יש בדיוק 4 מסמכים בקולקציה הזו.
- **‏‎`events`** נדיר. ‎5-30 ביום.

**מה חשוב שיהיה ברור:**

> **‏‎`is_anomaly=True`** ב-`footfall` הוא **לא** אירוע ב-`events`. שני מאגרים שונים לקטגוריות שונות. הדשבורד קורא מ-`events` להצגה, ומ-`footfall` לגרף.

> **‏‎`Firestore quotas` הן פר-דקה** ‏(‎‎‎`50,000 reads/day`, אבל גם ‎`10,000 reads/minute`). אם הדשבורד עולה על ‎10K reads בבת אחת — יש throttle. הדשבורד שלנו קורא רק מה שצריך (‎onSnapshot listeners עם where clauses), אז זה לא קרה.

### 10.12 Firebase Storage — מבנה הבאקט

באקט: `turkey-footfall.firebasestorage.app`.

**מבנה:**

```
gs://turkey-footfall.firebasestorage.app/
├── snapshots/                          ← 24h TTL (lifecycle rule)
│   ├── anomalies/{slot_id}/{ts}.jpg
│   ├── anomalies/{slot_id}/{ts}_annotated.jpg
│   ├── returning/{slot_id}/eid{N}_seen{K}_{ts}.jpg
│   ├── returning/{slot_id}/eid{N}_seen{K}_{ts}_full.jpg
│   └── events/loiter/{slot_id}/loiter_eid{N}_{ts}.jpg
├── review_sync/                        ← אין TTL — מאגר קבוע
│   ├── manifest.json                   ← אינדקס של כל הקבצים ב-review_sync
│   ├── review_frames/{cam_id}/{ts_us}.jpg + .json
│   ├── live_samples/{cam_id}/{ts_us}.jpg
│   ├── entities/{cam_id}/{entity_id}/{ts_us}.jpg
│   └── reid.db                         ← ‎compact snapshot של SQLite
└── training/                           ← מאגר הענן של האימון
    ├── reviews.json                    ← ‎uploads מהדשבורד המקומי
    ├── snapshots/review_frames/...     ← פריימים מתויגים
    ├── adapter_current.json            ← מצביע לראש הפעיל
    ├── history.jsonl                   ← יומן קידומים/דחיות
    └── adapters/head_run{N}.pt         ← קבצי הראש עצמם
```

**חוקי lifecycle:**

- **‏‎`snapshots/`** נמחק אחרי ‎24 שעות. חוסך storage. הדוחות/הדשבורד מציגים אותם עד לתפוגה.
- **‏‎`review_sync/`** נשאר לצמיתות. LRU-מנוהל על ידי הקולקטור עצמו (`pool_sync.py`).
- **‏‎`training/`** נשאר לצמיתות. גדל עם הזמן.

**מה חשוב לשים לב:**

> **‏‎`manifest.json` הוא הקובץ הכי חשוב.** הוא dict של כל הקבצים ב-`review_sync/` עם `mtime` ו-`url` של כל אחד. הדשבורד המקומי מוריד אותו וסונכרן משם. אם ה-manifest נשבר — הדשבורד לא רואה כלום.

> **‏‎`cache_control="no-store"`** על ה-manifest ועל `reid.db`. בלי זה, ‎GCS CDN מגיש גרסה מיושנת עד שעה — הבעיה שגילינו חי.

> **הבאקט כולו public-read.** לא ‎`private`. בגלל שהקבצים מתאימים לצפייה מהדשבורד המקומי דרך HTTP רגיל. הדשבורד לא צריך מפתח.

### 10.13 Hot-swap של Detect head — איך זה בפועל עובד

זה מהלך עדין. כל ‎30 סבבים (‎‎‎~20 דקות):

```python
# ב-collector.py, סעיף ‎10.7 (2)
fetched = adapters.refresh_from_storage(firebase.storage)
if fetched:
    n = adapters.apply_current(model)
    print(f"  * adapter: hot-loaded {fetched} ({n} head tensors)")
```

**מה קורה בפועל:**

1. **‏‎`refresh_from_storage`** מוריד את `training/adapter_current.json` ‏(‎‎‎`~200 bytes`).
2. משווה עם `data/adapters/current.json` המקומי. אם השם זהה — מחזיר `None`.
3. אם שונה — מוריד את קובץ הראש `training/adapters/head_run{N}.pt` (~‎5MB).
4. שומר מקומית ל-`data/adapters/head_run{N}.pt`.
5. מעדכן את `data/adapters/current.json` המקומי.

**אחר כך `apply_current`:**

1. קורא את `data/adapters/current.json`.
2. טוען את הקובץ .pt (רק tensors של הראש, ~‎4-6 MB).
3. **‏‎`model.model.load_state_dict(head_tensors, strict=False)`** — מחליף את הראש **‎in-place** במודל הפעיל.
4. **אין restart, אין reload של השאר, אין spike ב-RAM.**

**חשוב שיהיה ברור:**

> **הפעלה של ‎`load_state_dict(strict=False)` באמצע לולאה חיה** — כן, זה בטוח. הקולקטור לא נמצא ב-`.predict()` בזמן הזה (הוא בין סבבים). ה-tensors הישנים משוחררים (‎‎`refcount=0`), החדשים מוקצים. **הצריכה תישאר זהה** — אותו גודל מודל.

> **‏‎`ADAPTERS_DISABLE=1`** — משתנה סביבה שאם מוגדר, מתעלמים מכל ראש מקודם. הצלה חירום אם ראש חדש מקרים דיוק (‎‎rollback רך).

### 10.14 יומנים ודיבוג — SSH ‎+ journalctl

**התחברות ל-VM:**
```bash
gcloud compute ssh turkey-collector --zone=us-east1-c
```

**יומן הקולקטור בזמן אמת:**
```bash
sudo journalctl -u collector -f
```

**יומן ה-digest (‎דוחות):**
```bash
sudo journalctl -u digest -n 20
```

**סטטוס:**
```bash
sudo systemctl status collector.service
# יראה: active (running), memory usage, uptime, last log lines
```

**‏‎restart ידני:**
```bash
sudo systemctl restart collector
```

**‏‎disable זמני (למשל לתחזוקה):**
```bash
sudo systemctl stop collector
# ...לעשות מה שצריך...
sudo systemctl start collector
```

**חיפוש OOM kills בעבר:**
```bash
sudo journalctl -u collector --since "1 week ago" | grep -i "oom\|killed"
```

**זיכרון בזמן אמת:**
```bash
free -h
# יראה: total 1.0G, used, available, swap
```

**דיסק:**
```bash
df -h /
# יראה: 30GB total, used (בטוח ‎<10GB)
```

### 10.15 Billing kill-switch

יש cloud function שכתבתי לתסריט חירום. הרעיון: אם GCP billing עולה מעל 0.10$ — ניצול חורג — הפונקציה **מפסיקה את הבילינג ומכבה כל שירות**. שיריון נגד תאונות.

**איך זה עובד:**

1. GCP Billing שולח alerts ל-Pub/Sub topic על שינויי חיוב.
2. הפונקציה `disable_billing` מאזינה ל-topic.
3. אם הודעה על ‎`cost > threshold`, קוראת ל-Cloud Billing API עם `unlink_project_from_billing_account`.
4. GCP מכבה מיד את כל השירותים במסלול משלמים.
5. VM נעצר (אבל לא נמחק).

**איך להפעיל מחדש:**

בעצמך, ידנית. הפונקציה לא יכולה להחזיר לבד — הזה בכוונה.

**קובץ:** `src/deploy/gcp-billing-killswitch/`. פרטים מלאים ב-`README.md` שם.

**מה חשוב לזכור:**

> **הכפתור הזה לא נלחץ במהלך הפרויקט.** אבל טוב שהוא שם — הפרויקט רץ במסלול ‎Always Free, ובסה"כ נשלם רק אם חורגים ממנו במקרה של קונפיגורציה שגויה.

### 10.16 שגרות ריקוברי

**מקרים ידועים:**

| מקרה | סימפטום | טיפול אוטומטי | טיפול ידני |
|---|---|---|---|
| **OOM kill** | ‎`journalctl` מראה ‎`Killed`, RSS ~900MB | ‎`Restart=always` ‎+ 15s | להוריד `--imgsz` או להסיר מצלמה |
| **‏‎Stream drop** | ‎`grab_frame` מחזיר `None` | ‎`picker` מדלג ל-fallback אחרי X כישלונות | לבדוק ‎`tvkur.com` — לפעמים down |
| **‏‎Firestore quota exceeded** | ‎‎`write_error 429` | לא — הקולקטור לא ידע לעצור | ‎`stop collector` עד למחר |
| **‏‎Reboot של GCP** | ‎VM כבוי לכמה דקות | ‎VM חוזר, ‎`enable=on-boot` מפעיל את הקולקטור | לא נדרש |
| **‏‎Adapter רע** | דיוק נופל מיד אחרי hot-load | לא | ‎`export ADAPTERS_DISABLE=1` ‎+ restart, או ‎`rollback` דרך `promote_adapter --rollback` |
| **‏‎`reid.db` corrupt** | ‎`SQLite disk image is malformed` | לא | ‎`rm data/reid.db && restart` — יתחיל מחדש |
| **‏‎disk full** (יומן ‎`No space left on device`) | קולקטור נכשל בכתיבה | לא | ‎`sudo journalctl --vacuum-size=100M` |

**‏‎Rules of thumb לתקינות:**

> **אם `journalctl -u collector -f` מראה סבב תוך ‎40-60 שניות ורק הודעות `*` פה ושם** — הכל בסדר. אם כל ריצה לוקחת ‎‎‎90+ שניות, יש בעיה (רשת איטית או ‎swap).

> **אם ‎`free -h`** מראה ‎`used > 850MB` באופן קבוע — קרוב ל-cgroup limit. יעצור בקרוב.

> **אם ‎`df -h /`** מראה ‎`>‎75%` בשימוש — הזמן לפינוי (`journalctl --vacuum-size=100M`).

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
