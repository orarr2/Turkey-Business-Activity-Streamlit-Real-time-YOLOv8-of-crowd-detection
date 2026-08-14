<div dir="rtl">

# מדריך תא-אחר-תא - המחברת התאומה (turkey_business_activity_yolov8s.ipynb)

מסמך זה מלווה את **המחברת התאומה** של הפרויקט, תא-אחר-תא. התאומה היא מראה של המחברת הראשית עם שינוי אחד קריטי: `MODEL_WEIGHTS = 'yolov8s.pt'`, בדיוק המשקולות המקובעות בקולקטור שרץ 24/7 על ה-VM מסוג `e2-micro` ב-GCP. המחברת הראשית רצה עם `yolo26m` (מודל דור 2026, גרסה medium) - מודל חזק שמשמש כהפניית דיוק (ground truth) מקומית; התאומה רצה עם המודל החלש יותר בדיוק כדי שהתוצאות המקומיות ניתנות להשוואה ישירה למספרים המצטברים בדשבורד הענן. הקובץ הזה עבר שינוי-שם היום מהשם ההיסטורי `_yolov8n` לאחר שהמשקולות ב-VM שודרגו מ-nano ל-small.

כשתא כאן זהה לתא במחברת הראשית מציינים זאת מפורשות, יחד עם ההשלכה המעשית של השימוש במודל החלש (חוסר-דיוק מדוד, שנצבע בפרק הקליברציה).

הדשבורד שהתאומה מרימה בפרק 7 הוא אותו דשבורד של הראשית, כולל עשר שכבות הניתוח החי (‏paths, pose, gestures, body, faces, line, loiter, parking, plates, heat). ההסבר המלא מנגנון-אחר-מנגנון, עם הספים והנימוקים לכל שכבה: ‏PROJECT_GUIDE_HE פרק 5 ("10 שכבות הניתוח החי").

</div>

---

<div dir="rtl">

## תוכן עניינים

### מבוא (תאים 0-3)

1. [תא 0 - כותרת + סיכום מלמעלה למטה](#cell-0)
2. [תא 1 - מציאות רשת](#cell-1)
3. [תא 2 - סולם ה-fallback של המדינות](#cell-2)
4. [תא 3 - שני זמני-ריצה: המחברת מול ה-VM](#cell-3)

### חלק 0 - Setup (תאים 4-6)

5. [תא 4 - כותרת החלק](#cell-4)
6. [תא 5 - בדיקת תלויות](#cell-5)
7. [תא 6 - יבואים + טעינת מודל (‏VM parity)](#cell-6)

### חלק 1 - בחירת מצלמה (תאים 7-13)

8. [תא 7 - סבר-בורר מצלמות](#cell-7)
9. [תא 8 - קטלוג המצלמות עם קישורים](#cell-8)
10. [תא 9 - הבורר עצמו + auto-follow ל-VM](#cell-9)
11. [תא 10 - כותרת: מצלמות שנבחרו](#cell-10)
12. [תא 11 - checkpoint של הבחירה](#cell-11)
13. [תא 12 - הסבר: `resolve_stream` ו-kinds](#cell-12)
14. [תא 13 - טעינת המצלמה הראשונה](#cell-13)

### חלק 2 - בדיקת פריים בודד (תאים 14-15)

15. [תא 14 - כותרת + הסבר](#cell-14)
16. [תא 15 - grab_frame + YOLO + הצגה](#cell-15)

### חלק 3 - סדרת זמן של footfall (תאים 16-17)

17. [תא 16 - כותרת + הסבר](#cell-16)
18. [תא 17 - `footfall_series` והרצה קצרה](#cell-17)

### חלק 4 - אנומליות + פרופיל שעה (תאים 18-19)

19. [תא 18 - כותרת + הסבר](#cell-18)
20. [תא 19 - robust-z (‏median + MAD) + גרפים](#cell-19)

### חלק 5 - Dwell-time / עצירות ממושכות (תאים 20-22)

21. [תא 20 - כותרת + מוטיבציה](#cell-20)
22. [תא 21 - `dwell_analysis` עם ByteTrack](#cell-21)
23. [תא 22 - סימון עצירות ממושכות + Linger rate](#cell-22)

### חלק 5b - Re-identification (תאים 23-28)

24. [תא 23 - כותרת + הסבר האלגוריתם](#cell-23)
25. [תא 24 - הכנת מאגר Re-ID](#cell-24)
26. [תא 25 - לולאת דגימה + עדכון re-ID](#cell-25)
27. [תא 26 - roll-up: יישויות ייחודיות + regulars](#cell-26)
28. [תא 27 - גרף עקומת המבקרים החוזרים](#cell-27)
29. [תא 28 - הערת איכות + מסלול פרודקשן](#cell-28)

### חלק 6 - ציון "האם כדאי לפתוח כאן עסק" (תאים 29-30)

30. [תא 29 - כותרת + נוסחת הציון](#cell-29)
31. [תא 30 - `business_score` והדפסה](#cell-30)

### חלק 7 - השוואה לדשבורד הענן החי (תאים 31-32)

32. [תא 31 - כותרת + מוטיבציה](#cell-31)
33. [תא 32 - שרת local + local_grid.json + הטמעה ב-iframe](#cell-32)

### חלק 8 - השוואת אתרים מסחריים מרובים (תאים 33-34)

34. [תא 33 - כותרת + הסבר](#cell-33)
35. [תא 34 - דירוג המצלמות שנבחרו](#cell-34)

### חלק 9 - סיכום חי (תאים 35-36)

36. [תא 35 - כותרת + הסבר](#cell-35)
37. [תא 36 - איסוף אנומליות + re-ID + גרף](#cell-36)

### חלק 10 - קליברציית דיוק (תאים 37-40)

38. [תא 37 - כותרת + workflow](#cell-37)
39. [תא 38 - 10a: capture פריימים + ריצת YOLO בשני imgsz](#cell-38)
40. [תא 39 - 10b: תיוג אינטראקטיבי](#cell-39)
41. [תא 40 - 10c: דוח MAE + bias](#cell-40)

### חלק 11 - חיזוי (תאים 41-44)

42. [תא 41 - כותרת + הפילוסופיה: רק המנצח האופרטיבי](#cell-41)
43. [תא 42 - 11a: משיכת ההיסטוריה מ-Firestore ל-cache מקומי](#cell-42)
44. [תא 43 - 11b: resample לרשת 15 דקות + סינון epoch](#cell-43)
45. [תא 44 - 11c-VM: profile x EWMA + band](#cell-44)

### חלק 12 - איך הדשבורד עובד במצב twin (תא 45)

46. [תא 45 - הסבר על tabs + פורט + restart](#cell-45)

</div>

---

<a id="cell-0"></a>
<div dir="rtl">

## מבוא

### תא 0 - markdown - כותרת + סיכום מלמעלה למטה

**מה עושה:** תא הפתיחה של המחברת. מכיל שורת פרולוג ("Twin notebook - runs the same pinned yolov8s weights as the GCP collector"), כותרת ראשית ופסקת רקע: מה יקרה מלמעלה למטה כשמריצים את המחברת מקצה-לקצה, ולמה בכלל להריץ אותה מקומית במקום להסתכל על הדשבורד. הרעיון: לנתח דגימה קצרה של YOLO מ**מצלמה פומברית** אחת (טורקיה / תאילנד / יפן / ארה"ב) ולהשוות את התוצאה **לחי** להיסטוריית 24 השעות שנצברה בענן.

**למה:** תא המבוא מכוון את הקורא. הוא מבהיר שהמחברת אינה סתם demo של YOLO, אלא כלי השוואה: אתה מריץ אצלך דקה של דגימה על מצלמה, ומיד רואה איך היא נראית מול מספרי הענן על אותה יחידת זמן.

**פלטים:** רק תוכן markdown שמוצג ב-Jupyter (אין קוד להריץ). מפורט מה תקבל בכל שלב: setup, קטלוג + בורר, פרקים 1-6 עם ניתוחים, פרק 7 שמראה את הדשבורד עצמו מוטמע, ואחר-כך פרקים לחישוב ציון, דירוג, קליברציה וסיכום.

**שונה מהראשית?** כן. הכותרת מוסיפה שורה בראש ("Twin notebook - runs the same pinned yolov8s weights") שאינה קיימת בראשית, ובתיאור המפורט של setup נכתב במפורש "this twin = the VM's yolov8s". שאר התוכן זהה.

</div>

```markdown
Twin notebook - runs the same pinned yolov8s weights as the GCP collector.

# Business Activity - Live Footfall

Run this end-to-end to analyze **your own** short YOLO sample from a public
street camera (Turkey / Thailand / Japan / USA) AND compare it live to the
**cloud dashboard's 24-hour history** (pushed continuously by the collector
running on a GCP e2-micro).

What you get, top-to-bottom:

- Setup: dependency check, then load the detector (this twin = the VM's yolov8s).
- Camera catalog + picker: pick 4 cameras from ONE country by number.
- Sections 1-6: verify the stream decodes, then footfall, anomalies, dwell-time
  tracking and appearance-based re-identification on your picks.
- Section 7: **the live dashboard**, embedded inline, showing YOUR picked
  cameras (video) alongside the cloud collector's cumulative counts.
- Later sections: "is it worth opening a business here?" score, multi-site
  ranking, accuracy calibration and a final summary.
```

<a id="cell-1"></a>
<div dir="rtl">

### תא 1 - markdown - מציאות רשת

**מה עושה:** מזהיר שהזרמים של איסטנבול (`kamerayayin.ibb.istanbul` והישן `livestream.ibb.gov.tr`) פומביים אך נגישים רק מרשת פתוחה - המחשב שלך, VM או אפליקציה פרוסה. סביבות ארגז-חול חסומות (כולל הסביבה שבה נבנה הריפו) חוסמות את המארחים האלה דרך allowlist. לכן: **הרץ את המחברת מקומית**. מקטעי 1080p כבדים יורדים ב-head בלבד (~2.5 MB ראשונים), כך שגריפת פריים בודד נשארת מהירה גם בקו איטי.

**למה:** מנע ניסיונות ריצה כושלים. הקורא שיפעיל את המחברת בענן (Colab, sandbox) יגלה שכל המצלמות התורכיות מחזירות `None` ולא יבין למה - התא הזה חוסך את הזמן.

**פלטים:** רק תוכן markdown. אין קוד.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
### Network reality

The Istanbul streams (`kamerayayin.ibb.istanbul`, and the older
`livestream.ibb.gov.tr`) are public but reachable only from an **open
network** - your own machine, a VM, or a deployed app. Restricted
sandboxes (incl. the environment that generated this repo) block those
hosts via an allowlist. So **run this notebook locally**, where the
streams resolve. Heavy 1080p segments are downloaded head-only (first
~2.5 MB) so a single frame grab stays fast even on a slow link.
```

<a id="cell-2"></a>
<div dir="rtl">

### תא 2 - markdown - סולם ה-fallback של המדינות

**מה עושה:** מסביר איך הקולקטור ב-VM בוחר מצלמות: אף פעם לא נעול על סט קבוע, אלא מריץ מחלקת `CountryDirector` שתמיד מציגה 4 מצלמות ממדינה אחת ונופלת בסולם עדיפות **טורקיה - תאילנד - יפן - ארה"ב** רק כשהמדינה הפעילה כולה חשוכה. בתוך כל מדינה יש סולם משנה (רשימת המצלמות של אותה מדינה) עם ארבעה כללים:

1. **בריאות פר-מצלמה** - מצלמה שמחטיאה 3 דגימות ברצף נחה 15 דקות; הגריד ממלא מהספסל של אותה מדינה. מצלמה מסוג `tvkur` (קוניה) היא בדיקה בסיכון נמוך - החטאה אחת מכניסה אותה למנוחה.
2. **מנתק-זרם ברמת ה-host** - כשה-host כולו מסרב גישה (‏HTTP 403/429, מה שקורה למצלמות IBB איסטנבול מ-GCP), *כל* המצלמות שלו נחות 20 דקות ובקשת גישוש בודדת קובעת מתי הן חוזרות. זה מונע מהקולקטור להכות ב-CDN חוסם.
3. **התקדמות מדינה** - רק כשהמדינה הפעילה לא יכולה להעמיד ולו מצלמה חיה אחת הגריד עובר למדינה הבאה. מצלמה מתה אחת לעולם לא מזיזה את הגריד ממדינה - היא רק ממולאת.
4. **התאוששות לפני הדוח** - כמה דקות לפני כל דוח יומי הקולקטור מבצע re-probe למדינות עדיפות גבוהה יותר. טורקיה היא הנושא, לכן הגריד קופץ אליה חזרה ברגע שהחסימה משתחררת.

שדות הדוח (כותרת, אזור-זמן ל-baseline של שעה-של-שבוע, שער יום/לילה) עוקבים אחר איזו מדינה - ואיזו מצלמה - חיה כרגע.

**למה:** הכלל הזה שולט על מה תראה במחברת התאומה כשהיא ב-auto-follow. אם ה-VM ברגע נתון על ספסל ארה"ב, גם התאומה שלך תרוץ על אותן 4 מצלמות ארה"ב, לא על המצלמות שקיווית לראות.

**פלטים:** רק תוכן markdown.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
### How the grid chooses cameras: a country fallback ladder

The 24/7 cloud collector (on the VM) never watches a fixed set of cameras. It
runs a **`CountryDirector`** that always shows **4 cameras from one country**
and falls through a priority ladder when a country goes dark:

**Turkey - Thailand - Japan - USA**

Within each country there is a second ladder (the country's own camera list).
The rules:

- **Per-camera health.** A camera that misses 3 samples in a row rests for
  15 min; the grid backfills from deeper in the **same country's** bench. A
  `tvkur` (Konya) camera is a low-risk probe - one miss rests it.
- **Host circuit breaker.** When a whole host refuses access (HTTP 403/429 -
  what happens to Istanbul's IBB cameras from Google Cloud), *all* its cameras
  rest for 20 min and a single probe request decides when they return. This
  stops the collector from hammering a blocking CDN.
- **Country advance.** Only when the active country can field **no** live
  camera does the grid move to the next country. A single dead camera never
  moves the grid off a country - it just backfills.
- **Recovery before the report.** A few minutes before each daily report the
  collector re-probes higher-priority countries. Turkey is the subject, so the
  grid jumps back to it the moment its block lifts.

The report's fields (title, timezone for the hour-of-week baseline, the
day/night gate) follow whichever country - and camera - is live: a Bangkok
street and an Istanbul square cross into night at different UTC hours, and the
US bench alone spans Eastern, Central and Pacific time.
```

<a id="cell-3"></a>
<div dir="rtl">

### תא 3 - markdown - שני זמני-ריצה: המחברת מול ה-VM

**מה עושה:** מציג טבלת השוואה בין המחברת המקומית (החזקה) לקולקטור ה-VM (החלש) עם ארבעה עמודים: מטרה, מודל, גודל קלט, חומרה, עלות. הקטע העיקרי שהתאומה חולקת עליו: הראשית מתוארת כרצה `yolo26m` ב-`imgsz=960`, וה-VM כרץ `yolov8s` ב-`imgsz=640` (עד 2026-08-05 היה `yolov8n`). **התא נשאר כפי שהוא גם בתאומה** ולא הוחלף - הוא נשאר טקסט הסבר על הפער בין המקומי לענן, למרות שהתאומה עצמה כבר לא "המקומית החזקה" אלא בעצם המצב של ה-VM.

מספרים מדודים ב-2026-08-05 על אותם פריימים חיים: `yolov8n@512` מצא 0 אנשים בטקסים ו-0 כלי רכב בסראחנה; `yolov8s@960` מצא 5 ו-7; `yolo26m@960` מצא 6 אנשים ו-16 כלי רכב + אוטובוס. ההערה בסוף מפנה למחברת התאומה (הזאת): `turkey_business_activity_yolov8s.ipynb` (השם ההיסטורי, טרם עודכן ל-`yolov8s`) - הרץ אותה כדי לראות בדיוק מה ה-VM רואה.

**למה:** לתאר את פער-הדיוק הצפוי. מי שמסתכל על המספרים בענן ומצפה למספרים שלמים ומדויקים צריך לדעת שהמודל שמייצר אותם מגיב חלש למצלמות רחוב רחבות; המחברת הראשית + הקליברציה נמדדים את הפער.

**פלטים:** רק תוכן markdown.

**שונה מהראשית?** בתוכן - לא. במעמד - כן: התא מדבר על "This notebook (local, strong)" מול "The VM collector (weak)", בעוד שהתאומה עצמה **היא** המצב החלש. ההערה בסוף עדיין מפנה לתאומה בשם היסטורי (`_yolov8n`) שכבר לא רלוונטי מאז השינוי לשם `_yolov8s`. הקורא מוזמן לפרש: המחברת שאתה מריץ עכשיו = המחברת שההערה מפנה אליה.

</div>

```markdown
### Two runtimes: this notebook (strong) vs the VM collector (weak)

The exact same detection pipeline runs in two places, tuned differently:

| | **This notebook (local)** | **The VM collector (cloud)** |
|---|---|---|
| Purpose | Explore, calibrate, prove accuracy | 24/7 aggregation into Firestore |
| Model | **YOLO26-m** (2026 generation, medium) | **`yolov8s`** (small; nano until 2026-08-05) |
| Input size (`imgsz`) | 960 (recovers small/distant objects) | 640 (fits the free-tier CPU/RAM) |
| Hardware | your machine (can use a GPU) | GCP `e2-micro`, 2 shared vCPU, 1 GB |
| Cost | free, run on demand | must stay inside the Always-Free tier |

**Why weaker on the VM?** The `e2-micro` has ~1 GB RAM and two *shared* vCPUs.
A model like YOLO26-m at `imgsz=960` would blow the memory budget, so the VM
runs the strongest configuration that fits: `yolov8s` at `imgsz=640` (measured
on the live VM 2026-08-05: ~3 s of model time per 40 s round, RSS within the
service's 760M ceiling). The earlier `yolov8n@512` era was an OOM fix from the
5-camera days and undercounted badly - Sarachane peaked at 0 people in seven
straight digests. What the small model still misses vs the YOLO26-m reference
is the accuracy gap the calibration section measures.

**How big is the gap?** Measured 2026-08-05 on identical live frames: the old
`yolov8n@512` found **0** people at Taksim and **0** vehicles at Sarachane;
`yolov8s@960` found 5 and 7; **YOLO26-m@960 found 6 people and 16 vehicles
plus a bus.** So: the YOLO26-m notebook is the accurate reference; the VM is
the cheap, always-on estimator. The calibration section quantifies the gap.

> There are two notebooks. **This one** (`turkey_business_activity.ipynb`, on
> GitHub) is the YOLO26-m reference. A local-only twin
> (`turkey_business_activity_yolov8s.ipynb`) is identical except it loads
> `yolov8n` - run it to see EXACTLY what the VM sees.
```

<a id="cell-4"></a>
<div dir="rtl">

## חלק 0 - Setup

### תא 4 - markdown - כותרת החלק

**מה עושה:** כותרת חלק בלבד (`## 0. Setup`). מסמן את המעבר מהמבוא לחלק ההכנה הטכני.

**שונה מהראשית?** לא.

</div>

```markdown
## 0. Setup
```

<a id="cell-5"></a>
<div dir="rtl">

### תא 5 - code - בדיקת תלויות

**מה עושה:** רץ פעם אחת ומוודא שכל הספריות שהמחברת צריכה מותקנות; מה שחסר, מותקן ב-`pip install -q`; בסוף מדפיס גרסאות מותקנות. הרשימה `REQUIREMENTS` מזווגת שם ייבוא לשם חבילה ב-pip: `cv2` -> `opencv-python-headless`, `PIL` -> `Pillow`, `yt_dlp` -> `yt-dlp`, `firebase_admin` -> `firebase-admin`, וכולי. `ipywidgets` מודבק ב-`>=8` כי הדשבורד ב-Jupyter דורש את זה.

**למה:** אם ההרצה הראשונה נפלה על `ModuleNotFoundError` באמצע פרק 5 זה מעצבן; להתקין הכל בהתחלה זול, בטוח לחזור ולהריץ (הפעלות עוקבות הן רק dump של גרסאות).

**אלטרנטיבות:** `requirements.txt` + `pip install -r` היה אלגנטי יותר בפרויקט אמיתי, אבל במחברת שרצה גם על מכונה חדשה בלי הכנה מקדימה, קוד שמזהה ומתקן חסרים חוסך שלב.

**פלטים:** טבלת סטטוס כמו:

```
import             pip package              status
------------------------------------------------------------------
cv2                opencv-python-headless   OK  v4.10.0.84
numpy              numpy                    OK  v2.1.3
ultralytics        ultralytics              OK  v8.3.44
...
```

אם משהו התקין, מודפסת בסוף הערה: "restart the kernel and re-run from the top" - חובה אחרי התקנת חבילות בזמן ריצת קרנל.

**שונה מהראשית?** לא, זהה.

</div>

```python
# Dependency check: verify every library the notebook needs is installed,
# install any that are missing, then print the installed versions.
# Safe to re-run - subsequent runs are just a version dump.
import importlib, subprocess, sys

# (import_name, pip_name). Pinned only where a min version matters.
REQUIREMENTS = [
    ('cv2',            'opencv-python-headless'),
    ('numpy',          'numpy'),
    ('pandas',         'pandas'),
    ('matplotlib',     'matplotlib'),
    ('PIL',            'Pillow'),
    ('ultralytics',    'ultralytics'),
    ('yt_dlp',         'yt-dlp'),
    ('firebase_admin', 'firebase-admin'),
    ('ipywidgets',     'ipywidgets>=8'),
    ('urllib3',        'urllib3'),
]

def _pip_install(spec: str) -> None:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', spec],
        stdout=sys.stdout, stderr=sys.stderr,
    )

def _version(mod) -> str:
    return getattr(mod, '__version__', getattr(mod, 'VERSION', 'unknown'))

print(f'{"import":18} {"pip package":24} status')
print('-' * 66)
missing = []
for import_name, pip_spec in REQUIREMENTS:
    try:
        m = importlib.import_module(import_name)
        print(f'{import_name:18} {pip_spec:24} OK  v{_version(m)}')
    except ImportError:
        print(f'{import_name:18} {pip_spec:24} MISSING -> installing')
        missing.append((import_name, pip_spec))

for import_name, pip_spec in missing:
    _pip_install(pip_spec)
    importlib.invalidate_caches()
    m = importlib.import_module(import_name)
    print(f'  {import_name:16} installed  v{_version(m)}')

if missing:
    print()
    print('NOTE: some packages were just installed. If the next cell errors '
          'with ModuleNotFoundError, restart the kernel (Kernel -> Restart) '
          'and re-run from the top so Python picks up the new installs.')
```

<a id="cell-6"></a>
<div dir="rtl">

### תא 6 - code - יבואים + טעינת מודל (‏VM parity)

**מה עושה:** תא ההכנה הגדול של המחברת. מבצע ארבעה דברים:

1. **יבואים סטנדרטיים** - `sys`, `time`, `datetime`, `defaultdict`, `Path`, `cv2`, `numpy`, `pandas`, `matplotlib.pyplot`.
2. **איתור עץ `src/`** - הבלוק מנחש איפה נמצאת חבילת `app`: אם `cwd/src/app` קיים אז `_src_dir = cwd/src`, אחרת `_src_dir = cwd`. זה מאפשר להריץ את המחברת גם מתיקיית השורש (הפריסה הרגילה) וגם מתוך `src/` עצמו.
3. **יבואים מ-`app`** - `load_model`, `detect_and_count`, `grab_frame`, `resolve_youtube`, `resolve_stream`, `VEHICLE_NAMES` מ-`detect_core`; `CAMERAS`, `active_cameras`, `GRID_CAMERAS` מ-`cameras`.
4. **טעינת המודל עם VM parity** - `MODEL_WEIGHTS = 'yolov8s.pt'`. `DATA_DIR = _src_dir / 'data'` נוצרת אם חסרה, ו-`model = load_model(str(_src_dir / MODEL_WEIGHTS))` טוענת את המשקולות. מוצגים בסוף: שם המודל, רשימת המצלמות הזמינות, וה-`GRID_CAMERAS` (4 המצלמות של גריד ברירת-המחדל).

**למה:** התא הזה הוא **הלב של התאומה**. הערת ה-`# --- Model: VM parity ---` בקוד מסבירה שהמחברת הזאת מדביקה בדיוק את מה שרץ 24/7 ב-VM: `yolov8s` ב-`imgsz 640`, מקובע ב-`deploy/gcp-vm/collector.service` מ-2026-08-05 (התקופה של `yolov8n@512` הכשילה במניה חסרה). הדפוס `(VM-parity: this IS the VM model, @640)` בפלט מוודא שהמפעיל מזהה מיד את המצב.

**אלטרנטיבות:** אפשר היה לטעון את `yolo26m.pt` ולקבל תוצאות מדויקות יותר - אבל אז המחברת הופכת לעוד עותק של הראשית ומאבדת את היעד שלה: להוציא מספרים שאפשר להשוות אחד-לאחד ל-Firestore ולדשבורד.

**פלטים:** שלוש שורות:

```
model: yolov8s.pt (VM-parity: this IS the VM model, @640)
cameras available: [ ... כל cam_id-ים ]
dashboard grid (4 live cameras): [ ... GRID_CAMERAS ...]
```

**שונה מהראשית?** כן, זה **התא המכריע**. בראשית `MODEL_WEIGHTS = 'yolo26m.pt'` (המודל החזק), כאן `MODEL_WEIGHTS = 'yolov8s.pt'` והערת VM parity ארוכה בקוד. כל התאים שלאחר-מכן מייצרים מספרים שונים בהתאם - נמוכים יותר בממוצע, אבל תואמים לענן.

</div>

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

# --- Model: VM parity ------------------------------------------------------
# This TWIN mirrors EXACTLY what the 24/7 cloud VM runs on its 1 GB e2-micro:
# yolov8s at imgsz 640 (pinned in deploy/gcp-vm/collector.service since
# 2026-08-05; the nano@512 era undercounted badly). The GitHub notebook
# (turkey_business_activity.ipynb) is the strong YOLO26-m reference.
MODEL_WEIGHTS = 'yolov8s.pt'
DATA_DIR = _src_dir / 'data'; DATA_DIR.mkdir(parents=True, exist_ok=True)
model = load_model(str(_src_dir / MODEL_WEIGHTS))
print('model:', MODEL_WEIGHTS, '(VM-parity: this IS the VM model, @640)')
print('cameras available:', list(active_cameras()))
print('dashboard grid (4 live cameras):', GRID_CAMERAS)
```

<a id="cell-7"></a>
<div dir="rtl">

## חלק 1 - בחירת מצלמה

### תא 7 - markdown - הסבר הבורר

**מה עושה:** מבוא לתאים 8-11: התא הבא ידפיס את **הקטלוג המלא של המצלמות** כרשימה ממוספרת אחת, מקובצת לפי מדינה, ואז ישאל אותך **ארבע פעמים** על מספר מצלמה - אחת לכל תא בגריד. הארבע חייבות להיות **שונות** וכולן מ**אותה מדינה** (הגריד מנתח מדינה אחת בכל פעם, בדיוק כמו הקולקטור). אחרי המספר הרביעי מודפסת הבחירה, וכל שאר המחברת מנתחת את ארבע המצלמות האלה. ההערה בסוף: הבחירה נשמרת לכל חיי הקרנל הזה, כך ש-**Run All** זורם ישר אחרי ההקלדה. לשינוי: **Kernel > Restart Kernel** ואז בורר מחדש.

**למה:** להבטיח שהמפעיל יבין שהוא הולך להיות שאול ולא ייבהל מבקשת input באמצע Run All. גם מסביר מדוע 4 מצלמות ולמה מאותה מדינה - תואם התנהגות הקולקטור.

**פלטים:** רק תוכן markdown.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
### Camera catalog + picker - pick 4 by number

The cell below prints the **full camera catalog** as one numbered list,
grouped by country, then asks you **four times** for a camera number - one
per grid slot. The four must be **distinct** and all from the **same
country** (the grid analyses one country at a time, exactly like the VM
collector: 4 cameras from one country, falling through to the next country
when one is blocked). After the fourth number it prints your selection and
the rest of the notebook analyses those four cameras.

> The pick is held for this kernel session, so **Run All** flows straight
> through after you enter the four numbers once. To choose a different grid:
> **Kernel > Restart Kernel** and run the picker again.
```

<a id="cell-8"></a>
<div dir="rtl">

### תא 8 - code - קטלוג המצלמות עם קישורים

**מה עושה:** מייצר טבלה עשירה של HTML עם כל המצלמות בקטלוג, מקובצות לפי מדינה, כשכל שם מצלמה הוא לינק לדף המקור שלה (webcamera24 / IBB / tvkur). השדות בדף בורר החוזר של מבנה הנתונים: `flag`, `display` של המדינה; `name`, `city`, `page`, `url` של המצלמה. כל מדינה יוצרת רשימה `<ol>` עם התכולה, וכל item מנומם רץ עם `value="{מספר גלובלי}"`. בסוף מודפס סכום המצלמות והערה שהבורר משתמש באותם מספרים.

**למה:** מאפשר למפעיל להסתכל, לחפש לפי שם, ולפתוח את דף המקור בלחיצה - קטלוג טהור באתר בלי טקסט text-only כמו בתא הבא.

**פלטים:** בלוק HTML גלול (max-height 360px), עם לינקים ניתנים ללחיצה שנפתחים בטאב חדש (`target="_blank" rel="noopener"`).

**שונה מהראשית?** לא, זהה.

</div>

```python
# Camera catalog with LINKS - auto-generated from app/cameras.py, so it is
# always accurate and self-maintaining. Every camera, grouped by country,
# links to its source page (webcamera24 / IBB / tvkur).
from app.cameras import (CAMERAS as _CAT, COUNTRIES as _CO,
                         COUNTRY_ORDER as _OR, country_pool as _cp)
from IPython.display import display as _disp, HTML as _H

_rows = ['<div style="font-family:Arial,sans-serif;font-size:14px;max-height:'
         '360px;overflow:auto;border:1px solid #ddd;padding:10px 14px">',
         '<b>All available camera streams</b> (click a name to open its source):']
_n = 0
for _c in _OR:
    _m = _CO[_c]
    _rows.append(f'<h4 style="margin:10px 0 2px">{_m["flag"]} {_m["display"]}</h4>'
                 '<ol style="margin:0 0 6px 22px;padding:0">')
    for _cid in _cp(_c):
        _n += 1
        _cam = _CAT.get(_cid, {})
        _url = _cam.get("page") or _cam.get("url") or "#"
        _city = _cam.get("city", "")
        _where = f' <span style="color:#888">({_city})</span>' if _city else ''
        _rows.append(f'<li value="{_n}"><a href="{_url}" target="_blank" '
                     f'rel="noopener">{_cam.get("name", _cid)}</a>{_where}</li>')
    _rows.append('</ol>')
_rows.append(f'<div style="color:#888;margin-top:6px">{_n} cameras total. '
             'The picker below uses these same numbers.</div></div>')
_disp(_H(''.join(_rows)))
```

<a id="cell-9"></a>
<div dir="rtl">

### תא 9 - code - הבורר עצמו + auto-follow ל-VM

**מה עושה:** התא המשמעותי ביותר בפרק 1. שלוש פסקאות עבודה עיקריות:

1. **בניית הקטלוג הממוספר** - `_CAT_IDS` היא רשימה שטוחה של כל ה-`cam_id`-ים לפי סדר `_ORDER` (סולם הענדפויות של המדינות: טורקיה, תאילנד, יפן, ארה"ב). מספר גלובלי 1 = המצלמה הראשונה של טורקיה, וכו'. מודפסת כותרת גדולה + טבלה אנושית עם דגלים ומספרים.
2. **פונקציית `_probe_picks`** - בדיקה חיה של הבחירה: קריאת פריים אחד מכל מצלמה שנבחרה. אם מצלמה מחזירה `None` - היא מסומנת DEAD ומודפסת אזהרה. זה נוסף ב-2026-07-18 כדי לתפוס גריד מת בזמן בחירה, לא לגלות MISS אחרי חמישה פרקים של ניתוחים ריקים. המצלמות של IBB איסטנבול חסומות מ-IP-ים לא-טורקיים ברמת HLS גם אם דף האינטרנט משחק.
3. **VM-grid auto-follow (רק בתאומה)** - כשהדגל `FORCE_MANUAL_PICK = False` (ברירת המחדל) והבחירה עוד לא הוחלה: הקוד מנסה לקרוא מ-Firestore את המסמך `config/grid`, לחלץ את שדות `country` ו-`slots[].active_cam`, ולעקוב **אחר מה שה-VM מפעיל ברגע זה**. אם ההתחברות ל-Firebase הצליחה והנתונים תקינים: `SELECTED_CAMS` = אותן 4 מצלמות מהגריד של ה-VM, `SELECTED_COUNTRY` = אותה מדינה. חיפוש credentials: `FIREBASE_CREDENTIALS` בסביבה, אחרת קובץ בעל תבנית `*adminsdk*.json` בתיקייה הנוכחית או ההורה.

לבסוף, אם אין auto-follow או המשתמש הגדיר `FORCE_MANUAL_PICK = True`: לולאה `while len(_picks) < 4` שקוראת מספר מ-`input()`, מוודאת שהמספר בטווח, שהמצלמה לא כבר נבחרה, ושהמדינה שלה תואמת לקודמות. אחרי 4 מוצלחות: `SELECTED_CAMS_APPLIED = True` וקורא ל-`_probe_picks`.

**למה:** התא הזה הוא שער הכניסה של המחברת. ההוספה של VM auto-follow ייחודית לתאומה: המטרה שלה היא לשקף את ה-VM, לכן ברירת המחדל היא לעקוב אחריו אוטומטית. אם המפעיל רוצה לבחור ידנית - הוא משנה את הדגל בקוד.

**אלטרנטיבות:** ipywidgets Selectionmenu היה נחמד יותר ויזואלית, אבל `input()` בטוח יותר בסביבות כמו JupyterLab desktop / VS Code Notebooks שלפעמים לא מרנדרים widgets נכון. הבחירה של lax on `_KEEP` (שמירת בחירה קיימת בין הרצות) חוסכת רה-בחירה בכל Run All.

**פלטים:** בזמן ריצה - טבלה עשירה של קטלוג, שאלות `input()`, שורות `ok N/4:`, בסוף:

```
APPLIED: turkey -> ['ibb_taksim', 'ibb_eminonu', 'ibb_sarachane', 'tvkur_hukumet']
probing your picks (one frame each, ~5-20s per dead cam)...
   LIVE  Taksim Square
   LIVE  Eminonu
   ...
```

או במקרה של auto-follow:

```
FOLLOWING THE VM GRID - this notebook mirrors the live collector.
APPLIED: turkey -> ['ibb_taksim', ...]
(set FORCE_MANUAL_PICK = True in this cell to pick manually)
```

**שונה מהראשית?** כן, מהותית. הראשית לא כוללת את הבלוק של `VM-grid auto-follow` - היא תמיד מבקשת בחירה ידנית. התאומה כברירת מחדל **עוקבת אחר ה-VM** (זה תפקידה), ורק דגל `FORCE_MANUAL_PICK` מחזיר את הבורר הידני. זו ההבחנה הפונקציונלית העיקרית בין השתיים בחלק 1.

</div>

```python
# CAMERA PICKER - one big catalog, pick 4 cameras by NUMBER.
# Prints every camera with a number, then asks you 4 times for a camera
# number. The 4 must be DISTINCT and all from the SAME country (the grid runs
# one country at a time, exactly like the VM collector). The pick is held for
# this kernel session; Kernel > Restart Kernel to choose a different grid.
from app.cameras import (CAMERAS as _CATALOG, COUNTRIES as _COUNTRIES,
                         COUNTRY_ORDER as _ORDER, country_pool as _country_pool)

MAX_CAMS = 4

# One numbered catalog across ALL countries, in the collector's priority
# order. _CAT_IDS[i] is the cam_id shown as number i+1.
_CAT_IDS = []
print("=" * 66)
print("CAMERA CATALOG  -  pick 4 cameras by number (all from ONE country)")
print("=" * 66)
for _c in _ORDER:
    _m = _COUNTRIES[_c]
    print(f'\n{_m["flag"]}  {_m["display"].upper()}')
    for _cid in _country_pool(_c):
        _CAT_IDS.append(_cid)
        _cam = _CATALOG.get(_cid, {})
        _city = _cam.get("city", "")
        _where = f'  ({_city})' if _city else ''
        print(f'  {len(_CAT_IDS):2d}.  {_cam.get("name", _cid)}{_where}')
_N = len(_CAT_IDS)
print("-" * 66)


def _country_of(cid):
    return _CATALOG.get(cid, {}).get("country")


# Keep a valid, already-applied pick across re-runs so Run All flows straight
# through after the one-time selection. A fresh kernel starts unapplied.
def _probe_picks(_cids):
    # Live probe (2026-07-18): one frame per pick, HERE, so a dead
    # grid is caught at apply time - not five sections of MISS later.
    # (Turkey IBB streams, for example, currently refuse non-Turkey
    # IPs at the raw-HLS level even when the web PAGE plays.)
    print('probing your picks (one frame each, ~5-20s per dead cam)...')
    try:
        from app.detect_core import resolve_stream, grab_frame
        _dead = []
        for _cid in SELECTED_CAMS:
            _fr = None
            try:
                _fr = grab_frame(resolve_stream(_CATALOG[_cid]))
            except Exception:
                pass
            _st = 'LIVE' if _fr is not None else 'DEAD'
            if _fr is None:
                _dead.append(_cid)
            print(f'   {_st:4s}  {_CATALOG[_cid].get("name", _cid)}')
        if _dead:
            print(f'WARNING: {len(_dead)} of {len(SELECTED_CAMS)} picks '
                  'are not delivering frames from THIS machine. Re-run '
                  'this cell (Kernel > Restart) and pick live cameras, '
                  'or expect MISS lines in every analysis section.')
    except Exception as _pe:
        print(f'(probe skipped: {type(_pe).__name__}: {_pe})')


# --- VM-grid auto-follow (twin only): this notebook exists to MIRROR the
# collector, so by default it analyzes exactly the cameras the VM is on
# right now (config/grid). Manual picking stays available via the flag.
FORCE_MANUAL_PICK = False

_auto = None
if not FORCE_MANUAL_PICK and not globals().get('SELECTED_CAMS_APPLIED'):
    try:
        import os as _os
        from pathlib import Path as _Path
        import firebase_admin as _fba
        from firebase_admin import credentials as _fbc, firestore as _fbf
        _key = _os.environ.get('FIREBASE_CREDENTIALS')
        if not _key:
            _hits = (sorted(_Path('.').glob('*adminsdk*.json'))
                     or sorted(_Path('..').glob('*adminsdk*.json')))
            _key = str(_hits[0]) if _hits else None
        if _key and not _fba._apps:
            _fba.initialize_app(_fbc.Certificate(_key))
        if _fba._apps:
            _g = (_fbf.client().collection('config').document('grid')
                  .get().to_dict() or {})
            _cams = [s.get('active_cam') for s in _g.get('slots', [])
                     if s.get('active_cam') in _CATALOG]
            if _cams and _g.get('country') in _COUNTRIES:
                _auto = (_g['country'], _cams[:MAX_CAMS])
    except Exception as _e:
        print(f'(VM-grid auto-follow unavailable: {type(_e).__name__}: {_e} '
              f'- falling back to the manual picker)')

if _auto:
    SELECTED_COUNTRY, SELECTED_CAMS = _auto[0], list(_auto[1])
    SELECTED_CAMS_APPLIED = True
    print('FOLLOWING THE VM GRID - this notebook mirrors the live collector.')
    print(f'APPLIED: {SELECTED_COUNTRY} -> {SELECTED_CAMS}')
    print('(set FORCE_MANUAL_PICK = True in this cell to pick manually)')
    _probe_picks(SELECTED_CAMS)

_prev = globals().get('SELECTED_CAMS')
_prev_country = globals().get('SELECTED_COUNTRY')
_KEEP = (bool(globals().get('SELECTED_CAMS_APPLIED'))
         and isinstance(_prev, list) and len(_prev) == MAX_CAMS
         and _prev_country in _COUNTRIES
         and all(c in _country_pool(_prev_country) for c in _prev))

if _KEEP:
    print(f'APPLIED (kept): {_prev_country} -> {_prev}')
    print('Selection held from earlier this session. '
          'Kernel > Restart Kernel to change it.')
else:
    try:
        _picks = []
        _country = None
        while len(_picks) < MAX_CAMS:
            _raw = input(f'Camera {len(_picks) + 1} of {MAX_CAMS}  '
                         f'(number 1-{_N}): ').strip()
            if not _raw.isdigit() or not (1 <= int(_raw) <= _N):
                print(f'  x  "{_raw}" is not a number between 1 and {_N}. '
                      'Try again.')
                continue
            _cid = _CAT_IDS[int(_raw) - 1]
            if _cid in _picks:
                print(f'  x  #{_raw} ({_CATALOG[_cid]["name"]}) already picked. '
                      'Choose another.')
                continue
            _cc = _country_of(_cid)
            if _country is None:
                _country = _cc
            elif _cc != _country:
                print(f'  x  #{_raw} is in {_cc}, but you started with '
                      f'{_country}. All 4 must be ONE country. Try again.')
                continue
            _picks.append(_cid)
            print(f'  ok  {len(_picks)}/{MAX_CAMS}: '
                  f'{_CATALOG[_cid]["name"]} ({_country})')
        SELECTED_CAMS = list(_picks)
        SELECTED_COUNTRY = _country
        SELECTED_CAMS_APPLIED = True
        print("=" * 66)
        print(f'APPLIED: {SELECTED_COUNTRY} -> {SELECTED_CAMS}')
        print('Run the cells below (or Run All).')
        _probe_picks(SELECTED_CAMS)
    except (EOFError, KeyboardInterrupt):
        SELECTED_CAMS_APPLIED = False
        print('\nPicker cancelled - nothing selected. Re-run this cell.')
    except Exception as _e:
        # Headless execution (nbconvert/papermill) has no stdin.
        if type(_e).__name__ == 'StdinNotImplementedError':
            SELECTED_CAMS_APPLIED = False
            print('This picker needs an interactive kernel (it calls '
                  'input()). Run it in Jupyter, not headless.')
        else:
            raise
```

<a id="cell-10"></a>
<div dir="rtl">

### תא 10 - markdown - כותרת: מצלמות שנבחרו

**מה עושה:** כותרת קצרה שמסבירה שהתא הבא הוא ה-checkpoint היחיד שכל שאר המחברת סומכת עליו. לפני שמריצים את הבורר הוא עוצר בנימוס (צפוי בהרצה טרייה); אחרי Apply הוא רושם את הבחירה הסופית שכל הריצה תשתמש בה.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
### Picked cameras

The single checkpoint the rest of the notebook depends on. Before you Apply
in the picker above it stops politely (expected on a fresh run); after
Apply it records the final selection the whole run will use.
```

<a id="cell-11"></a>
<div dir="rtl">

### תא 11 - code - checkpoint של הבחירה

**מה עושה:** מוודא ש-`SELECTED_CAMS_APPLIED` דלוק. אם לא - זורק חריגה מותאמת `_ApplyFirst` שה-Jupyter מציג כטרייסבק ידידותי בן שלוש שורות: "PAUSED: no cameras selected yet (expected on a fresh run). In the picker cell above: enter your 4 camera numbers, then run this cell - or Run All." אם כן: מייצר קופסת HTML ירוקה שמציגה את המדינה, רשימת המצלמות שנבחרו והכיתוב "The rest of the notebook will analyse THESE 4 cameras from <country>".

**למה:** לתת סימן חזותי חד-משמעי שהבחירה הוחלה בהצלחה, ולעצור בבטחה כשמנסים להריץ מלפני הבורר.

**פלטים:** בלוק HTML ירוק עם השדות המלאים, למשל:

```
APPLIED. COUNTRY = turkey | SELECTED_CAMS = ['ibb_taksim', ...]
- ibb_taksim - Taksim Square
- ibb_eminonu - Eminonu
...
The rest of the notebook will analyse THESE 4 cameras from turkey.
```

**שונה מהראשית?** לא, זהה.

</div>

```python
# PICKED CAMERAS - the checkpoint for everything below.
if not globals().get('SELECTED_CAMS_APPLIED'):
    class _ApplyFirst(Exception):
        def _render_traceback_(self):
            return [
                'PAUSED: no cameras selected yet (expected on a fresh run).',
                'In the picker cell above: enter your 4 camera numbers,',
                'then run this cell - or Run All.',
            ]
    raise _ApplyFirst()

from IPython.display import display, HTML
_ctry = globals().get('SELECTED_COUNTRY', 'turkey')
_rows = ''.join(f'<li><b>{c}</b> - {CAMERAS[c]["name"]}</li>'
                for c in SELECTED_CAMS)
display(HTML(
    '<div style="border:3px solid #16a34a;background:#f0fdf4;padding:14px 16px;">'
    f'<b style="font-size:16px;color:#166534;">APPLIED. COUNTRY = {_ctry} | '
    f'SELECTED_CAMS = {SELECTED_CAMS}</b>'
    f'<ul style="margin:8px 0 4px 20px;">{_rows}</ul>'
    f'<b>The rest of the notebook will analyse THESE {len(SELECTED_CAMS)} '
    f'cameras from {_ctry}.</b></div>'))
```

<a id="cell-12"></a>
<div dir="rtl">

### תא 12 - markdown - הסבר: `resolve_stream` ו-kinds

**מה עושה:** כותרת חלק (`## 1. Pick a camera`) עם רקע נרחב יותר: הקטלוג המלא נמצא ב-`app/cameras.py`, מאורגן ל-pools לפי מדינה (`COUNTRIES` / `country_pool`). הקולקטור בענן צועד על סולם מדינות (טורקיה - תאילנד - יפן - ארה"ב): מריץ 4 מצלמות שונות ממדינה אחת ונופל לבאה כשהנוכחית חסומה. הבורר שלמעלה משקף את זה - אתה בוחר 4 מצלמות ממדינה אחת ו-`SELECTED_CAMS[0]` היא זו שהחלק הזה בודק.

הפונקציה `resolve_stream(cam)` הופכת כל ערך בקטלוג ל-URL של HLS שאפשר לפתוח, ללא קשר ל-`kind`, וזוכרת את התוצאה עד לתוקף ה-token:

- `hls` - נעשה בו שימוש ישיר (kamerayayin / tvkur).
- `youtube` - נפתר דרך `yt-dlp` עם client `android` (תאילנד / יפן / ארה"ב).
- `skyline` - ה-playlist עם token של `hd-auth.skylinewebcams.com`, נסחט חי.
- `webcamera24` - נגן tvkur/YouTube המוטמע בדף webcamera24.

מארחים מסוימים נפתרים רק מרשת פתוחה (IBB חסומה מחוץ לטורקיה, skyline/webcamera24 מסובבים tokens). לכן הרץ את זה על המחשב שלך.

**למה:** להסביר למה בורר קטלוג עובד גם על מצלמות מסוגים כל-כך שונים - כי `resolve_stream` מנרמל את כולן ל-URL של HLS ברמת ה-API.

**פלטים:** רק תוכן markdown.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 1. Pick a camera

The full catalog lives in `app/cameras.py`, organised into per-country pools
(`COUNTRIES` / `country_pool`). The cloud collector walks a **country ladder**
(Turkey -> Thailand -> Japan -> USA): it runs four DISTINCT cameras from ONE
country and falls through to the next country when the current one is blocked
or dark. The picker above mirrors that - you chose four cameras from a single
country, and `SELECTED_CAMS[0]` is the one this section inspects.

`resolve_stream(cam)` turns any catalog entry into an openable HLS URL
regardless of `kind`, and caches the result until its token expires:

- `hls` - used directly (kamerayayin / tvkur).
- `youtube` - resolved via yt-dlp using the `android` client (Thailand /
  Japan / USA street cameras are all YouTube-backed).
- `skyline` - the tokenized `hd-auth.skylinewebcams.com` playlist, scraped live.
- `webcamera24` - the embedded tvkur/YouTube player on the webcamera24 page.

Some hosts only resolve from an **open network** (IBB is geo-blocked outside
Turkey; skyline/webcamera24 rotate tokens), so run this on your own machine;
in a restricted sandbox those cameras fail while the YouTube-backed ones work.
```

<a id="cell-13"></a>
<div dir="rtl">

### תא 13 - code - טעינת המצלמה הראשונה

**מה עושה:** שוב מוודא `SELECTED_CAMS_APPLIED` (אותו pattern כמו בתא 11). לאחר-מכן: `CAM_ID = SELECTED_CAMS[0]` - מצלמה #1 מהבחירה של המפעיל. `cam = CAMERAS[CAM_ID]` שולף את המילון של המצלמה, ו-`stream_url = resolve_stream(cam)` מנרמל אותה ל-HLS. מדפיס את השם ואת ה-URL.

**למה:** לקבע את "המצלמה הראשית של הרצה" - כל הפרקים 2-6 (בדיקת פריים, footfall, אנומליות, dwell, re-ID) עובדים עליה. הפרקים 8/10 שרלוונטיים לכל 4 המצלמות מסתמכים גם הם ישירות על `SELECTED_CAMS`.

**פלטים:** שורה אחת:

```
Taksim Square -> https://kamerayayin.ibb.istanbul/.../playlist.m3u8?...
```

**שונה מהראשית?** לא, זהה.

</div>

```python
if not globals().get('SELECTED_CAMS_APPLIED'):
    class _ApplyFirst(Exception):
        def _render_traceback_(self):
            return ['PAUSED: run the picker cell above and enter your 4 camera numbers first.']
    raise _ApplyFirst()
# Read the operator's first choice from the verify cell above so the
# rest of this notebook analyses whatever camera they actually picked
# (instead of a hard-coded default that ignored their choice).
CAM_ID = SELECTED_CAMS[0]
cam = CAMERAS[CAM_ID]
stream_url = resolve_stream(cam)   # handles hls / youtube / skyline / webcamera24
print(cam['name'], '->', stream_url)
```

<a id="cell-14"></a>
<div dir="rtl">

## חלק 2 - בדיקת פריים בודד

### תא 14 - markdown - כותרת + הסבר

**מה עושה:** כותרת החלק ומשפט הסבר קצר: "Confirm the stream decodes and YOLO sees the crowd before collecting anything." לפני שמריצים דגימה ארוכה, בודקים שהזרם באמת פותח, שהמצלמה שולחת ביטים, ושהמודל מזהה עצמים סבירים.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 2. Single-frame check

Confirm the stream decodes and YOLO sees the crowd before collecting anything.
```

<a id="cell-15"></a>
<div dir="rtl">

### תא 15 - code - grab_frame + YOLO + הצגה

**מה עושה:** קורא `grab_frame(stream_url)`. אם הפריים `None` - מודפסת הוראה איך לשנות מצלמה (Kernel > Restart Kernel, בחר מצלמה אחרת או מדינה אחרת - תאילנד/יפן/ארה"ב מגובות YouTube ועובדות מכל מקום). אם קיים: מודפסים `frame.shape` ותוצאת `detect_and_count(model, frame)`.

לאחר-מכן `model.predict(frame, conf=0.35, classes=[0,1,2,3,5,6,7], verbose=False)[0]`:

- `conf=0.35` - סף ביטחון מינימלי לזיהוי.
- `classes=[0,1,2,3,5,6,7]` - מסנן ל-COCO IDs: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 6 train, 7 truck.
- מדפיס את `res.plot()` שממחיש את הזיהויים כתמונה עם bounding boxes וצבעי מחלקה, ומציג עם matplotlib (`figsize=(11, 6)`).

**למה:** בדיקת שפיות ויזואלית. אם מודל nano מדבר על "0 person" בכיכר עסוקה - יש בעיה (זה בדיוק מה שקרה בעידן `yolov8n@512` ב-VM). כמו כן, אישור שהקורא רואה **מה** המודל מזהה, לא רק כמה.

**פלטים:**

```
frame shape: (1080, 1920, 3)
counts: {'person': 5, 'vehicles': 7}
```

+ תמונה עם boxes של כל הזיהויים.

**שונה מהראשית?** לא בקוד. במעשה כן: במחברת הראשית עם `yolo26m@960` תראה יותר boxes ותקבל מספרים גבוהים יותר; כאן, עם `yolov8s@640`, המספרים נמוכים יותר - שוב, בכוונה, לחפיפה מלאה עם ה-VM.

</div>

```python
frame = grab_frame(stream_url)
if frame is None:
    print(f"WARN: {cam['name']} returned no frame (stream down or geo-blocked).")
    print('Pick a different camera: Kernel > Restart Kernel, then in the picker')
    print('choose another camera (or another country - Thailand/Japan/USA are')
    print('YouTube-backed and work from anywhere), enter its 4 numbers,')
    print('and Run All again.')
else:
    print('frame shape:', frame.shape)
    print('counts:', detect_and_count(model, frame))

    res = model.predict(frame, conf=0.35, classes=[0,1,2,3,5,6,7], verbose=False)[0]
    plt.figure(figsize=(11, 6))
    plt.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(cam['name']); plt.show()
```

<a id="cell-16"></a>
<div dir="rtl">

## חלק 3 - סדרת זמן של footfall (דגימה דלילה)

### תא 16 - markdown - כותרת + הסבר

**מה עושה:** מסביר שלמענה על השאלה "כמה / מתי" לא צריך כל פריים - דגימה אחת כל 15-30 שניות מספיקה, ועדינה לשרת. זאת אותה לוגיקה שהקולקטור מריץ ברציפות.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 3. Footfall time series (sparse sampling)

For the **how much / when** question we don't need every frame - one sample every 15-30s is plenty and
is gentle on the server. This is the same logic the collector runs continuously.
```

<a id="cell-17"></a>
<div dir="rtl">

### תא 17 - code - `footfall_series` והרצה קצרה

**מה עושה:** מגדיר `footfall_series(stream_url, cam_name, interval_s=20, duration_min=1.0)`: לולאה שמסתיימת ב-`t_end = time.time() + duration_min*60`. בכל איטרציה:

1. `ts = dt.datetime.now(dt.timezone.utc)` - חותמת זמן UTC.
2. `f = grab_frame(stream_url)` - פריים בודד.
3. `c = detect_and_count(model, f)` אם היה פריים, אחרת `{'person': NaN, 'vehicles': NaN}`.
4. שורה ב-`rows` עם `ts, cam, person, vehicles`.
5. הדפסה של שורה אנושית: `[HH:MM:SS] person=N vehicles=M`.
6. `time.sleep(interval_s)` - המתנה עד הדגימה הבאה.

בסוף מחזיר `pd.DataFrame(rows)`.

הקריאה הישירה: `df = footfall_series(stream_url, cam['name'], interval_s=10, duration_min=1.0)` - דקה של דגימה כל 10 שניות, כלומר 6 דגימות. שומר ל-`data/footfall_{CAM_ID}.csv` ומראה את 5 השורות הראשונות.

**למה:** לייצר מדגם קטן שעליו יעבדו כל הפרקים הבאים (אנומליות, ציון business). "דקה" זה מוזל - למחקר אמיתי כתוב "just leave the collector daemon (`python -m app.collector`) running for genuine 24/7 data".

**אלטרנטיבות:** דגימה כל פריים נותנת רזולוציה גבוהה יותר, אבל: מכפילה את עלות ה-CPU פי כמה, ומחייבת אחסון גדול; לא נצרך למגמות שעתיות ויומיות.

**פלטים:** 6 שורות log כמו `[14:32:15] person=5 vehicles=7`, ואז הראש (5 שורות) של ה-DataFrame:

```
                              ts       cam  person  vehicles
0  2026-08-13 14:32:15.000+00:00  Taksim...      5         7
...
```

+ קובץ `data/footfall_ibb_taksim.csv`.

**שונה מהראשית?** לא, זהה. במעשה: התוצאות של `person`/`vehicles` יהיו נמוכות יותר מהראשית באותה מצלמה + דקה, כי `yolov8s@640` פחות רגיש לאובייקטים קטנים/רחוקים - זה הפער שהקליברציה בפרק 10 מודדת.

</div>

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

# Short live-collection run. Raise duration_min for longer studies, or just leave the
# collector daemon (`python -m app.collector`) running for genuine 24/7 data.
df = footfall_series(stream_url, cam['name'], interval_s=10, duration_min=1.0)
df.to_csv(DATA_DIR / f'footfall_{CAM_ID}.csv', index=False)
df.head()
```

<a id="cell-18"></a>
<div dir="rtl">

## חלק 4 - אנומליות + פרופיל שעה

### תא 18 - markdown - כותרת + הסבר

**מה עושה:** מסביר את הרעיון: **אנומליה = rolling z-score > 2.5** על סדרת ה-footfall - זינוק פתאומי (אירוע/מבצע/הפגנה) או צניחה חריגה (סגירה/מזג-אוויר). פרופיל שעה אומר לך *מתי* חלון-הזמן המסחרי.

**שונה מהראשית?** לא, זהה. (בקוד בפועל התא הבא משתמש ב-z=3.5, לא 2.5 כפי שכתוב במרקדאון - הבדל היסטורי; הראשית זהה גם היא.)

</div>

```markdown
## 4. Anomalies + peak-hour profile

**Anomaly = rolling z-score > 2.5** on the footfall series: a sudden surge (event/promotion/protest) or an
unusual drop (closure/weather). Peak-hour profile tells you *when* the commercial window is.
```

<a id="cell-19"></a>
<div dir="rtl">

### תא 19 - code - robust-z (‏median + MAD) + גרפים

**מה עושה:** מגדיר `flag_anomalies(s, window=12, z=3.5, min_delta=3)`. זו הפונקציה של הקולקטור בשם אחר, מיושמת על סדרת ה-`person` המקומית:

1. `med = s.rolling(window, min_periods=4).median()` - חציון גולל של הסדרה.
2. `mad = (s - med).abs().rolling(window, min_periods=4).median() * 1.4826` - MAD (‏Median Absolute Deviation) מכוון כדי לתת אומדן חסון של סטיית התקן; המקדם 1.4826 הופך את ה-MAD לאומדן consistent של sigma תחת גאוסיאני.
3. `spread = mad.clip(lower=1.0)` - מונע spread=0 כשה-MAD אפס (סדרה יציבה); בסופר של הזיהוי, הספירות שלמות, לכן רף אפס אפשרי כשכל הסביבה זהה.
4. `robust_z = (s - med) / spread` - z חסון.
5. מחזיר `(|robust_z| > z) & (|s - med| >= min_delta)` - שני התנאים ביחד: גם חריגות סטטיסטית וגם שינוי מוחלט משמעותי (לפחות 3 אנשים).

הערה בקוד: "Outliers already inside the window inflate a mean/std baseline and mask the next event; a median/MAD baseline barely moves." - זהו היתרון של MAD על ממוצע/סטיית-תקן.

לאחר-מכן: `df['anomaly'] = flag_anomalies(df['person'])`, ואז שני גרפים בפריסה 1x2:

- **שמאל:** `person` על ציר הזמן עם `marker='o'`; מעליו scatter אדום של הנקודות שסומנו כאנומליה.
- **ימין:** ממוצע `person` פר שעה של היום, כ-bar chart - פרופיל שעת השיא.

**למה:** להדגים בעין המפעיל מדד ואיכות של החוסן. באנומליה של דקה בזרם קצר של 6 דגימות סביר להניח שלא יופיע כלום; זה במכוון, זה שיפוט של המדד.

**פלטים:** גרף כפול. הימני אפור-כמעט-ריק כי כל הדגימות היו באותה שעה של היום.

**שונה מהראשית?** לא בקוד. בפועל: אותם ערכי סף על מספרים נמוכים יותר יגרמו לפחות אנומליות פר יחידת-זמן במחברת התאומה - עוד גילוי-חסר של המודל החלש.

</div>

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

df['ts'] = pd.to_datetime(df['ts'])
df['anomaly'] = flag_anomalies(df['person'])

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].plot(df['ts'], df['person'], marker='o', label='people')
an = df[df['anomaly'] == True]
ax[0].scatter(an['ts'], an['person'], color='red', zorder=5, label='anomaly')
ax[0].set_title('Footfall over time (robust z)'); ax[0].legend()

df['hour'] = df['ts'].dt.hour
df.groupby('hour')['person'].mean().plot(kind='bar', ax=ax[1])
ax[1].set_title('Avg people by hour (peak-hour profile)')
plt.tight_layout(); plt.show()
```

<a id="cell-20"></a>
<div dir="rtl">

## חלק 5 - Dwell-time / עצירות ממושכות (tracking)

### תא 20 - markdown - כותרת + מוטיבציה

**מה עושה:** מסביר את שינוי הפרדיגמה: "כמה זמן אדם או רכב נשאר מול המצלמה?" - זו שאלה שמחייבת **object tracking** (‏IDs יציבים לאורך פריימים), שעובד רק על פריימים **רצופים** - לכן לוקחים כאן **burst צפוף** קצר (כמה fps ל-~60 שניות) במקום דגימה דלילה. `model.track()` של Ultralytics (‏ByteTrack) נותן id לכל אובייקט; מצטברים כמה פריימים כל id נראה וכמה מעט הוא זז.

- **שהייה ארוכה + תנועה נמוכה** = lingering: window-shopping / תור / רכב חונה.
- אחוז גבוה של אנשים lingering הוא סימן חזק לאיכות מסחרית (אנשים עוצרים, לא רק חולפים).

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 5. Dwell-time / prolonged stops (tracking)

"How long does a person or vehicle stay in front of the camera?" needs **object tracking** (stable IDs
across frames), which only works on *consecutive* frames - so here we take a short **dense burst**
(a few fps for ~60s) instead of sparse sampling. Ultralytics `model.track()` (ByteTrack) gives each
object an id; we accumulate how many frames each id is seen and how little it moves.

- **Long dwell + low movement** = lingering: window-shopping / a queue / a parked vehicle.
- High share of *lingering* people is a strong **commercial-quality** signal (people stop, not just pass).
```

<a id="cell-21"></a>
<div dir="rtl">

### תא 21 - code - `dwell_analysis` עם ByteTrack

**מה עושה:** מגדיר `dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.35)`. שלושה בלוקים:

1. **מבני נתונים** - `frames_seen: defaultdict(int)` (סופר פריימים ל-id), `centroids: defaultdict(list)` (רשימת מרכזי bbox ל-id לחישוב תנועה), `track_cls: {}` (מחלקת האובייקט לפי id).
2. **לולאת דגימה** - `n_frames = int(seconds * target_fps)`, `stride = max(1, round(25 / target_fps))`. ההערה בקוד: "source streams run ~25 fps, so skip frames with stride - reading consecutive frames would compress the whole window into ~n/25 seconds and overstate dwell ~8x." לכל פריים ב-`iter_frames(stream_url, max_frames=n_frames, stride=stride)`: `r = model.track(frame, persist=True, conf=conf, classes=[0,1,2,3,5,6,7], tracker='bytetrack.yaml', verbose=False)[0]`. `persist=True` שומר את ה-tracker בין קריאות. אם `r.boxes.id` לא None: לכל box+id+cls, מגדיל `frames_seen[tid]`, מוסיף `(x_center, y_center)` ל-`centroids[tid]`, שומר `track_cls[tid] = cl`.
3. **סיכום ל-DataFrame** - לכל id: `movement = float(np.linalg.norm(pts.max(0) - pts.min(0)))` = המרחק בין הפינות הכי מרוחקות של מסלול המרכז (מדד גס של טווח תנועה). `dwell_s = round(n / target_fps, 1)`. שורה: `track_id, class, dwell_s, movement_px`. מסדר לפי `dwell_s` יורד.

הערה בדוקסטרינג: "iter_frames handles header-required hosts (tvkur, IBB, skylinewebcams) by downloading the latest segments with the right Referer/Origin and decoding locally, since cv2.VideoCapture(url) can't pass headers on Windows." - הסיבה שלא משתמשים ב-`cv2.VideoCapture` ישירות.

הקריאה: `dwell = dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.25)` - 30 שניות של burst ב-3 fps ~ 90 פריימים. `conf` הורד ל-0.25 (מה-0.35 של הפרק הקודם) כדי לתפוס יותר.

**למה:** תרגיל האמונה החזק ביותר של המחברת. אחרי שדגימה דלילה נותנת רק ספירות, כאן רואים id-ים אמיתיים של אנשים ורכבים, כמה זמן הם נשארו, וכמה הם זזו.

**אלטרנטיבות:** DeepSORT היה שם עם association יותר חזק; ByteTrack נבחר כי הוא מובנה ב-Ultralytics ולא דורש embedder נוסף.

**פלטים:** DataFrame של 15 שורות ראשונות:

```
   track_id class     dwell_s  movement_px
0        12 person       28.0         14.2
1         7 car          25.7         42.0
...
```

**שונה מהראשית?** לא, זהה. בפועל: המחברת הראשית עם `yolo26m` תיצור יותר tracks, ותשמור אותם יציבים לאורך יותר פריימים (זיהוי טוב יותר -> פחות אובדן track -> יותר dwell זמן ארוך).

</div>

```python
from app.detect_core import iter_frames, NAME_BY_ID

def dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.35):
    """Dense burst with tracking. Returns per-track dwell seconds + movement.

    iter_frames handles header-required hosts (tvkur, IBB, skylinewebcams) by
    downloading the latest segments with the right Referer/Origin and decoding
    locally, since cv2.VideoCapture(url) can't pass headers on Windows.
    """
    frames_seen = defaultdict(int)
    centroids = defaultdict(list)
    track_cls = {}
    n_frames = int(seconds * target_fps)
    # Sample at ~target_fps: source streams run ~25 fps, so skip frames
    # with `stride` - reading consecutive frames would compress the
    # whole window into ~n/25 seconds and overstate dwell ~8x.
    stride = max(1, round(25 / target_fps))
    for frame in iter_frames(stream_url, max_frames=n_frames, stride=stride):
        r = model.track(frame, persist=True, conf=conf, classes=[0,1,2,3,5,6,7],
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

<a id="cell-22"></a>
<div dir="rtl">

### תא 22 - code - סימון עצירות ממושכות + Linger rate

**מה עושה:** מגדיר שלושה קבועים: `PERSON_DWELL_S=25, VEHICLE_DWELL_S=40, MAX_MOVE_PX=60`. אם `dwell` לא ריק, מסמן `stationary` = כל שורה שבה:

- אדם + `dwell_s >= 25` (25 שניות)
- או לא-אדם + `dwell_s >= 40`
- וגם `movement_px <= 60` (הזיז מעט)

מדפיס את מספר העצירות, מציג את הטבלה, ומחשב `linger_rate = (אנשים ששהו >=25s) / (סה"כ אנשים)`.

**למה:** להוציא מ-DataFrame גולמי של dwell שני מדדים אנושיים: "כמה עצרו" ו-"איזה אחוז מהאנשים שנראו עצר". השני הוא הבסיס של רכיב `linger` בציון business בפרק 6.

**אלטרנטיבות:** מעל 25 ו-40 שניות זה שרירותי; אפשר להתאים לפי מצלמה (רחבה יותר -> יותר זמן שהייה). מסמכי הפרויקט לא מציעים אמצעי כיול אוטומטי כאן; זו החלטת אחרונה של המפעיל.

**פלטים:**

```
Prolonged stops detected: 3
(טבלה של 3 שורות)
Linger rate (people who stayed >= 25s): 33%
```

**שונה מהראשית?** לא, זהה.

</div>

```python
# Flag prolonged stationary objects: long dwell AND little movement.
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

<a id="cell-23"></a>
<div dir="rtl">

## חלק 5b - Re-identification

### תא 23 - markdown - כותרת + הסבר האלגוריתם

**מה עושה:** מסביר את הבעיה: הספירות בפרק 3 אומרות **כמה** אנשים נראים ברגע נתון, אבל סופרות פעמיים כל מי ששהה מול המצלמה. כדי לענות "כמה לקוחות ייחודיים חלפו היום?" או "האם זה אותו רכב משלוחים שראיתי אתמול?" - צריך **re-identification**: זהות מתמשכת שדבוקה לאדם/רכב שורדת חוצה פריימים, bursts וימים.

יישום ב-`app/reid.py`:

1. לכל זיהוי YOLO, לחתוך את ה-bounding box.
2. לבנות היסטוגרמת HSV מוסָוה (‏masked) של הצבע: 8x8x8 bins, פיקסלים עם V<30 מסוננים החוצה (הופך את חוסר-הרגישות ללילות הצהובים של סודיום של הכיכר בקוניה), + יחס-אספקט + שטח מנורמל.
3. L2-normalize -> וקטור מראה 514-ממדי.
4. השוואה לכל יישות מאותה מחלקה שכבר ב-`data/reid.db` בקוסינוס. אם הטוב ביותר >= `threshold` (ברירת מחדל 0.92): עדכן `sightings` ו-`last_seen`; אחרת: רשום יישות חדשה.

זו **חתימה ברמת demo**. עובד טוב באור יום (בגדים בצבעים שונים -> היסטוגרמות שונות בבירור). מפיק התאמות שקר בלילה כשכל הסצנה צבועה צהוב. להחליף `embed_crop()` ב-forward pass של OSNet/torchreid לרמת פרודקשן; רגיסטר SQLite סביבו נשאר כפי שהוא.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 5b. Re-identification - "have I seen this person before?"

The detection counts above tell you *how many* people are visible at any moment, but they
double-count anyone who lingers in front of the camera. To answer questions like *"how many
unique customers walked by today?"* or *"is that the same delivery van I saw yesterday?"*
we need **re-identification**: a persistent identity attached to each person/vehicle that
survives across frames, bursts and days.

The implementation is in `app/reid.py`:

1. For each YOLO detection, crop the bounding box.
2. Build a *masked* HSV color histogram (8x8x8 bins, V<30 pixels ignored - kills the
   sodium-yellow night cast on the Konya square) plus aspect ratio + normalized area.
3. L2-normalize -> 514-dim appearance vector.
4. Compare to every entity of the same class already in `data/reid.db` via cosine
   similarity. If the best match is >= `threshold` (default 0.92) we update its
   `sightings` and `last_seen`; otherwise we register a new entity.

This is a **demo-grade signature**. It works well in daylight (different clothing colors
give clearly different histograms). It produces false matches at night when the whole
scene is yellow-tinted - swap `embed_crop()` for an OSNet/torchreid forward pass for
production-grade re-ID; the SQLite registry around it stays the same.
```

<a id="cell-24"></a>
<div dir="rtl">

### תא 24 - code - הכנת מאגר Re-ID

**מה עושה:** תא ההכנה של פרק 5b. שלושה בלוקים:

1. **checkpoint + יבואים** - וידוא `SELECTED_CAMS_APPLIED`, יבוא `detect_with_boxes, annotate` מ-`detect_core`, `ReidStore` מ-`reid`, ו-`cv2, time, plt`.
2. **בסיס הנתונים** - `REID_DB = str(_src_dir / 'data' / 'reid_notebook.db')`, בונה תיקייה אם חסרה. **פותר בעיה טיפוסית של Windows:** אם המשתמש מריץ תא זה שוב ב-Jupyter, `reid` הקודם עדיין מחזיק חיבור SQLite ל-DB, ו-`os.unlink` נכשל ב-`PermissionError [WinError 32]`. הפתרון: לנסות `reid.close()` (בלוק `try/except NameError` על הפעם הראשונה). לאחר-מכן `Path(REID_DB).unlink(missing_ok=True)` מוחק את הקובץ למען הריצה הריקה מחדש. אם עדיין נכשל ב-`PermissionError`: מדפיס אזהרה שהמאגר ימוזג במקום להישרף.
3. **פתיחת מאגר + בחירת מצלמה** - `reid = ReidStore(REID_DB, threshold=0.92)`. `CAM_ID = SELECTED_CAMS[0]` (בכוונה משתמשים באותה מצלמה מפרק 1-4 כדי שיהיו לרצוף), `cam = CAMERAS[CAM_ID]`, `stream_url = resolve_stream(cam)`. מדפיס "feeding re-ID from ...".

**למה:** להקפיד על "clean-room run" בכל הרצה - מבטיח שהתוצאות של הפרק לא מפירות המסד מקודם. הטיפול ב-PermissionError הוא תזכיר ש-Windows נעלה קבצים חזק יותר מ-Linux.

**פלטים:**

```
reid_notebook.db cleared - fresh demo registry
feeding re-ID from Taksim Square
```

**שונה מהראשית?** לא, זהה.

</div>

```python
if not globals().get('SELECTED_CAMS_APPLIED'):
    class _ApplyFirst(Exception):
        def _render_traceback_(self):
            return ['PAUSED: run the picker cell above and enter your 4 camera numbers first.']
    raise _ApplyFirst()
from app.detect_core import load_model, grab_frame, detect_with_boxes, annotate
from app.reid import ReidStore
import cv2, time
import matplotlib.pyplot as plt

REID_DB = str(_src_dir / 'data' / 'reid_notebook.db')
Path(REID_DB).parent.mkdir(parents=True, exist_ok=True)

# If we're re-running the notebook (the kernel is alive), the previous ReidStore is
# still holding a SQLite connection to REID_DB. Close it before we try to delete
# the file, otherwise Windows returns PermissionError [WinError 32].
try:
    reid.close()         # noqa: F821  (reid is defined by a prior run of this cell)
except NameError:
    pass

# Fresh registry for the demo so re-runs are reproducible. If something else
# still holds the file (orphan kernel, antivirus scan), we keep the existing
# rows instead of crashing - re-identification just continues with what's there.
try:
    Path(REID_DB).unlink(missing_ok=True)
    print('reid_notebook.db cleared - fresh demo registry')
except PermissionError:
    print('reid_notebook.db is locked by another process - keeping existing rows. '
          'New entities will be merged into the existing registry; this is fine '
          'for the demo, just not a clean-room run.')

reid = ReidStore(REID_DB, threshold=0.92)

# Use the model we already loaded above; lower conf so we catch the small/distant
# people the Konya wide-angle camera shows.
# CAM_ID inherits from the verify cell (SELECTED_CAMS[0]) so re-ID runs on
# the same camera as the earlier sections, not a hard-coded default.
CAM_ID = SELECTED_CAMS[0]
cam = CAMERAS[CAM_ID]
stream_url = resolve_stream(cam)   # hls / youtube / skyline / webcamera24
print('feeding re-ID from', cam['name'])
```

<a id="cell-25"></a>
<div dir="rtl">

### תא 25 - code - לולאת דגימה + עדכון re-ID

**מה עושה:** לולאה קצרה של 8 דגימות כל 5 שניות. פרמטרים: `N_SAMPLES, INTERVAL_S, CONF = 8, 5, 0.25`. לכל דגימה:

1. `f = grab_frame(stream_url)`; אם `None`: מדפיס `[NN] miss` וממתין.
2. `counts, boxes = detect_with_boxes(model, f, conf=CONF)` - שני dicts: `counts={'person', 'vehicles'}` ו-`boxes` = רשימת bounding boxes עם מחלקות.
3. `results = reid.update_from_frame(CAM_ID, f, boxes)` - בלב האלגוריתם: לכל box, מחשב embedding, מחפש ב-DB, מעדכן או רושם.
4. `new = sum(r.is_new for r in results)`, `seen_again = len(results) - new`.
5. שורה ב-`rows` עם `sample, person, vehicles, detections, new_ids, seen_again`.
6. הדפסת שורה אנושית.

בסוף `reid_df = pd.DataFrame(rows)`.

**למה:** להדגים בזמן אמת איך המאגר מתמלא. בהרצה ראשונה `new_ids` גבוה, `seen_again` נמוך; ככל שהזמן עובר `seen_again` עולה. זהו הצורה הבסיסית שהקולקטור מריץ ב-24/7, רק לזמן קצר.

**אלטרנטיבות:** דגימה עם `iter_frames` הייתה יעילה יותר (יותר פריימים לזמן), אבל אין צורך לזמן קצר.

**פלטים:** 8 שורות log וכשה-DataFrame מוצג בסוף:

```
[00] person=5 vehicles=7 -> new=12 seen_again=0
[01] person=6 vehicles=7 -> new=3  seen_again=9
...
```

**שונה מהראשית?** לא, זהה. במעשה: המודל החלש שולף פחות boxes -> פחות ניסיונות re-ID -> `total_unique` נמוך יותר; זה שוב הפער היחסי בין הגרסאות.

</div>

```python
# Sample N frames every `interval_s` seconds, run YOLO on each, push every detection
# through the re-ID registry. Short loop here so the notebook completes; the collector
# daemon does the real long-running version.
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
    print(f'[{i:02d}] person={counts["person"]} vehicles={counts["vehicles"]} '
          f'-> new={new} seen_again={seen_again}')
    time.sleep(INTERVAL_S)

reid_df = pd.DataFrame(rows)
reid_df
```

<a id="cell-26"></a>
<div dir="rtl">

### תא 26 - code - roll-up: יישויות ייחודיות + regulars

**מה עושה:** תא סיכום קצר:

1. `stats = reid.stats(CAM_ID)` - dict של סטטיסטיקות במחסן.
2. הדפסת סה"כ יישויות ייחודיות (`total_unique`), סה"כ תצפיות (`total_sightings`), ופר מחלקה (person / car / bicycle וכו') עם `unique`, `total_sightings`, `regulars(>=3)` (מספר יישויות שנצפו לפחות 3 פעמים).
3. `reid.top_regulars(CAM_ID, n=10)` - 10 היישויות שנצפו הכי הרבה, עם `entity_id, cls, sightings, first_seen, last_seen`.

**למה:** בלוק המספרים הזה עונה על השאלה "כמה לקוחות ייחודיים ראינו, וכמה מהם היו רגיל". הרגילים הם הבסיס לזיהוי דפוסי חזרה ("אותו רכב משלוחים ב-11:00 כל יום").

**פלטים:**

```
Total unique entities (this camera): 45
Total sightings: 67
  person      unique=32  sightings=48  regulars(>=3)=3
  car         unique=13  sightings=19  regulars(>=3)=1
Top returning entities:
  #  17  person    sightings=5  first=2026-08-13T14:32Z  last=2026-08-13T14:37Z
  ...
```

**שונה מהראשית?** לא, זהה.

</div>

```python
# Roll-up: how many unique entities did we see? how many came back >=3 times?
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

<a id="cell-27"></a>
<div dir="rtl">

### תא 27 - code - גרף עקומת המבקרים החוזרים

**מה עושה:** אם `len(reid_df) >= 3`: מחשב `returning_rate = seen_again / detections` (עם `.replace(0, NaN)` כדי לא לחלק באפס), ובונה גרף כפול:

- **שמאל:** `new_ids` (עיגולים) ו-`seen_again` (ריבועים) על ציר מספר הדגימה.
- **ימין:** `returning_rate` (`0..1`, ירוק) על ציר מספר הדגימה.

אחרת: "Not enough samples for the returning-visitor plot."

**למה:** הגרף השמאלי מראה איך הפוקוס עובר מ-"IDs חדשים" בהתחלה ל-"seen again" בסוף - עקומת ההתקדמות הטבעית של מאגר. הגרף הימני נותן מדד תמצית: אחוז המבקרים החוזרים לאורך זמן.

**פלטים:** גרף כפול (יופיע אם מספיק דגימות).

**שונה מהראשית?** לא, זהה.

</div>

```python
# Visual: returning-visitor curve - what fraction of detections are 'seen again' over time?
if len(reid_df) >= 3:
    reid_df = reid_df.copy()
    reid_df['returning_rate'] = (reid_df['seen_again'] /
                                 reid_df['detections'].replace(0, np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(reid_df['sample'], reid_df['new_ids'], marker='o', label='new IDs')
    ax[0].plot(reid_df['sample'], reid_df['seen_again'], marker='s', label='seen again')
    ax[0].set_title('Re-ID activity per sample')
    ax[0].set_xlabel('sample #'); ax[0].set_ylabel('count'); ax[0].legend()

    ax[1].plot(reid_df['sample'], reid_df['returning_rate'].fillna(0), marker='o',
               color='#36d399')
    ax[1].set_title('Returning-visitor rate (seen_again / detections)')
    ax[1].set_xlabel('sample #'); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()
else:
    print('Not enough samples for the returning-visitor plot.')
```

<a id="cell-28"></a>
<div dir="rtl">

### תא 28 - markdown - הערת איכות + מסלול פרודקשן

**מה עושה:** הערה חשובה על איכות re-ID תלויה בסצנה. בכיכר הממשלה בקוניה בלילה כל הסצנה אחידה בצהוב-סודיום. re-ID מבוסס היסטוגרמת-צבע ימזג over-merge שם. כדי לאמת את הרעיון, הפנה את המצלמה ל-Grand Bazaar / Spice Bazaar באור יום (בגדים בצבעים שונים) או קבע `threshold=0.97` לסלקטיביות גבוהה.

מסלול פרודקשן:

```
pip install torchreid
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='', device='cpu')
def embed_crop(crop, cls): return extractor([crop])[0].cpu().numpy()
```

ואז שמור על שאר `app/reid.py` בדיוק כפי שהוא. ה-embedding של OSNet ב-2,048 ממדים שורד שינויי תאורה, שינויי pose וחסימה חלקית טוב יותר בהרבה מהיסטוגרמת צבע.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
IMPORTANT - re-ID quality depends on the scene.
#
At Konya Hukumet Meydani at night the whole scene is uniform sodium yellow.
Color-histogram re-ID will over-merge IDs there. To validate the *concept*, point
the camera at the daylight Grand Bazaar / Spice Bazaar (different clothing colors)
or set `threshold=0.97` to be very conservative about matches.
#
Production path:
  pip install torchreid
  from torchreid.utils import FeatureExtractor
  extractor = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='', device='cpu')
  def embed_crop(crop, cls): return extractor([crop])[0].cpu().numpy()
Then keep the rest of app/reid.py exactly as-is. The 2,048-dim OSNet embedding
survives lighting changes, pose changes, and partial occlusion much better than
a color histogram.
```

<a id="cell-29"></a>
<div dir="rtl">

## חלק 6 - ציון "האם כדאי לפתוח כאן עסק"

### תא 29 - markdown - כותרת + נוסחת הציון

**מה עושה:** מסביר את שלושת האותות שמעורבבים לציון אחד 0-100. כוונן את המשקולות לפי סוג העסק (בית קפה רוצה `linger` גבוה; דוכן רוצה `throughput` גבוה):

- **Volume** - חציון footfall (ביקוש גולמי).
- **Linger** - אחוז אנשים שעצרים (engagement / conversion potential).
- **Consistency** - מקדם שונות (‏CV) נמוך (תנועה יציבה עדיפה על ספייקים).

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 6. "Is it worth opening a business here?" - a simple score

Combine three signals into one 0-100 score. Tune the weights to your business type (a cafe wants high
*linger*; a kiosk wants high *throughput*).

- **Volume** - median footfall (raw demand).
- **Linger** - share of people who stop (engagement / conversion potential).
- **Consistency** - low coefficient of variation (steady traffic beats spiky).
```

<a id="cell-30"></a>
<div dir="rtl">

### תא 30 - code - `business_score` והדפסה

**מה עושה:** מגדיר `business_score(footfall_df, dwell_df, w=(0.5, 0.3, 0.2))`:

1. `people = footfall_df['person'].dropna()`.
2. `volume = float(people.median())`.
3. `cv = float(people.std() / people.mean())` אם הממוצע > 0, אחרת 1.0.
4. `consistency = max(0.0, 1 - cv)`.
5. `is_p = dwell_df['class'] == 'person'`.
6. `linger = (אנשים ששהו >=25s) / (סה"כ אנשים)` אם `dwell_df` לא ריק, אחרת 0.
7. `vol_norm = min(1.0, volume / 40.0)` - נורמליזציה: 40 אנשים לפריים נחשב "מאוד עסוק"; לכייל פר FOV של מצלמה.
8. `score = 100 * (w[0]*vol_norm + w[1]*linger + w[2]*consistency)`.
9. מחזיר dict של 4 שדות: `volume_median, linger_rate, consistency, score_0_100`.

הקריאה: `print(cam['name']); business_score(df, dwell)`.

**למה:** לתת מספר יחיד ניתן להשוואה בין מיקומים. `w=(0.5, 0.3, 0.2)` נותן משקל גדול ל-volume (חצי מהציון), פחות ל-linger, ומעט ל-consistency - מתאים לחנות פיזית. משתמש יכול להעביר משקולות משלו.

**אלטרנטיבות:** מודל מפוקח (‏learned) על נתונים היסטוריים של מיקומים שהצליחו/נכשלו היה מדויק יותר; אבל דורש labels ואין להם כאן. הגישה החוקית: קומבינציה של אותות עם משקולות שאפשר להצדיק.

**פלטים:**

```
Taksim Square
{'volume_median': 5.0, 'linger_rate': 0.33, 'consistency': 0.85, 'score_0_100': 42.7}
```

**שונה מהראשית?** לא, זהה. במעשה: הציון בתאומה יהיה נמוך יותר משמעותית, כי `volume` תלוי במניית אנשים (חלשה) והוא הרכיב הכבד.

</div>

```python
def business_score(footfall_df, dwell_df, w=(0.5, 0.3, 0.2)):
    people = footfall_df['person'].dropna()
    volume = float(people.median()) if len(people) else 0.0
    cv = float(people.std() / people.mean()) if people.mean() else 1.0
    consistency = max(0.0, 1 - cv)
    is_p = dwell_df['class'] == 'person'
    linger = float((is_p & (dwell_df['dwell_s'] >= 25)).sum() / max(1, is_p.sum())) if len(dwell_df) else 0.0
    vol_norm = min(1.0, volume / 40.0)  # ~40 people/frame treated as 'very busy'; tune per camera FOV
    score = 100 * (w[0]*vol_norm + w[1]*linger + w[2]*consistency)
    return {'volume_median': round(volume,1), 'linger_rate': round(linger,2),
            'consistency': round(consistency,2), 'score_0_100': round(score,1)}

print(cam['name'])
business_score(df, dwell)
```

<a id="cell-31"></a>
<div dir="rtl">

## חלק 7 - השוואה לדשבורד הענן החי

### תא 31 - markdown - כותרת + מוטיבציה

**מה עושה:** מסביר את המעבר: שאר המחברת היה ניתוח **מקומי** - דקה של דגימה על מצלמה אחת. הקולקטור בענן רץ ברציפות ב-VM של GCP וצובר 4 מצלמות x 24 שעות ל-Firestore, וה-HTML דשבורד למטה נרשם לזה. השוואה בין השניים עונה על שאלות אמיתיות:

- האם הרגע שדגמתי מייצג את כל היום? (‏הדקה שלי מול הגרף של 24 שעות)
- האם אני על שיא, שפל או ממוצע?
- האם הופעלה אנומליה ב-24 השעות האחרונות שהחמצתי בגלל דגימה עכשיו?

שום דבר כאן לא כותב ל-Firestore - זה דף HTML פשוט שקורא ממנה.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 7. Compare with the live cloud dashboard

The rest of this notebook was your **local** analysis - a minute of sampling on
one camera. The cloud collector has been running non-stop on a GCP VM,
accumulating 4 cameras × 24 hours into Firestore, and the HTML dashboard below
subscribes to that. Comparing the two answers real questions:

- Is the moment I sampled representative of the whole day? (my minute vs the 24h chart)
- Am I hitting a peak, a valley, or the average?
- Did any anomaly fire in the last 24 hours that I missed by sampling now?

Nothing here writes to Firestore - it's a plain HTML page that reads from it.
```

<a id="cell-32"></a>
<div dir="rtl">

### תא 32 - code - שרת local + local_grid.json + הטמעה ב-iframe

**מה עושה:** התא הארוך והמהותי ביותר בחלק זה. שלושה בלוקים:

1. **בניית `local_grid.json`** - יוצר קובץ שאומר לדשבורד "הצג את המצלמות שבחר המשתמש, לא את גריד ה-VM". פונקציית `_local_grid_slot(i, cam_id)` מנרמלת כל מצלמה מהקטלוג ל-`slot` תואם-דשבורד: `slot_id`, `display_area`, `placeholder_name`, ואחד מ-`placeholder_hls / placeholder_embed / placeholder_page`. חוקים לפי `kind`:
   - `youtube` -> `placeholder_embed = _yt_embed(video_id)` (מזרם `www.youtube.com/embed/...?autoplay=1&mute=1&playsinline=1&enablejsapi=1`).
   - `webcamera24` -> קודם קורא ל-`resolve_webcamera24` שגורף את דף המקור, שם מחפש `_TVKUR_ID_RE` (אם tvkur -> proxy `/tvkur/{id}/master.m3u8`), אחרת `_YOUTUBE_RE`.
   - `hls` (‏default) -> tvkur ישיר עובר דרך proxy; IBB ישיר (בדרך כלל geo-blocked).
   
   אם המצלמות נבחרו: כותב JSON ל-`web/local_grid.json`. אם לא: מוחק את הקובץ (כדי שהדשבורד יחזור לגריד ה-VM).

2. **שרת HTTP מקומי** - חיפוש פורט חופשי בטווח 8000-8020 (מונע דריסה על שרת קיים). אם קיים שרת מהרצה קודמת (‏`_main._dash_server`): שימוש חוזר. אחרת: `ThreadingHTTPServer` עם `DashboardHandler` שמגיש מ-`WEB_DIR`. פועל ב-thread daemon.

3. **טעינת דפדפן + הטמעה ב-iframe** - `dash_url = f'http://localhost:{DASHBOARD_PORT}/?mode=twin'`. אם `_main._dash_browser_opened` False: `webbrowser.open(dash_url, new=2)` בפעם הראשונה. מציג לינק ב-HTML + IFrame בגודל 100%x640 בתא Jupyter.

`?mode=twin` אומר ל-`web/app.js` איזה סט של פאנלים לרנדר.

**למה:** הזה הרגע ה-"ווא": אתה רואה **בו-זמנית**, בתוך ה-Jupyter, את המצלמות שבחרת + הספירות מהענן. הפורט האוטומטי מונע התנגשות עם dev servers אחרים.

**אלטרנטיבות:** נגד "פשוט לפתוח דפדפן חיצוני" - IFrame מאפשר scroll טבעי בתוך ה-notebook ומחזיק את כל ה-scoring באותו סשן.

**פלטים:**

```
Dashboard grid -> YOUR 4 picked cameras (turkey): ['Taksim Square', ...]
Dashboard server started at http://localhost:8000/?mode=twin
```

+ קישור + iframe מוטמע.

**שונה מהראשית?** מהותית. הראשית פותחת עם `?mode=main` (או ללא query), התאומה עם `?mode=twin`. מצב twin נותן לדשבורד ה-JS לרנדר טאבים אחרים: Analysis + Search + **Reinforcement learning** (טאב שסקירה בפרק 12 של המדריך הזה). מצב main מציג יותר טאבים ואת כפתור "Send Report From VM" שלא רלוונטי כאן.

</div>

```python
# Serve web/ on http://localhost:8000 AND point the dashboard at the cameras
# YOU picked. Section 7 writes web/local_grid.json = your 4 SELECTED_CAMS (each
# resolved to a YouTube embed / tvkur proxy / direct HLS), so the local
# dashboard shows EXACTLY your picks - not the cloud VM's grid. The live COUNTS
# still come from Firestore (the 24/7 VM), so the picked tiles show video with
# "-" counts unless the VM happens to run the same camera; the numeric analysis
# for your picks lives in Sections 1-6 above. Delete web/local_grid.json to see
# the pure cloud dashboard.
import sys, json, threading, http.server, webbrowser
from app.dashboard_server import DashboardHandler, WEB_DIR, port_is_free
from app.detect_core import _YOUTUBE_RE, _TVKUR_ID_RE, resolve_webcamera24

def _yt_embed(vid):
    return f'https://www.youtube.com/embed/{vid}?autoplay=1&mute=1&playsinline=1&enablejsapi=1'

def _local_grid_slot(i, cam_id):
    cam = CAMERAS[cam_id]
    kind = cam.get('kind', 'hls')
    where = ', '.join(x for x in [cam.get('city', ''),
                                  (cam.get('country', '') or '').title()] if x)
    slot = {'slot_id': f'local_{i}', 'display_area': where,
            'placeholder_name': cam.get('name', cam_id),
            'placeholder_hls': None, 'placeholder_embed': None,
            'placeholder_page': cam.get('page') or cam.get('url')}
    try:
        if kind == 'youtube':
            m = _YOUTUBE_RE.search(cam['url'])
            if m:
                slot['placeholder_embed'] = _yt_embed(m.group(1))
        elif kind == 'webcamera24':
            master = resolve_webcamera24(cam.get('page', cam['url']))
            t = _TVKUR_ID_RE.search(master)
            if t:
                slot['placeholder_hls'] = f'/tvkur/{t.group(1)}/master.m3u8'
            else:
                y = _YOUTUBE_RE.search(master)
                if y:
                    slot['placeholder_embed'] = _yt_embed(y.group(1))
        else:  # 'hls': tvkur direct -> proxy; IBB -> direct (often geo-blocked)
            m = _TVKUR_ID_RE.search(cam.get('url', ''))
            slot['placeholder_hls'] = (f'/tvkur/{m.group(1)}/master.m3u8'
                                       if m else cam.get('url'))
    except Exception as e:
        print(f'  ! {cam_id}: could not resolve a stream for the tile '
              f'({type(e).__name__}); the tile will show its page link.')
    return slot

_lg = WEB_DIR / 'local_grid.json'
if globals().get('SELECTED_CAMS_APPLIED') and globals().get('SELECTED_CAMS'):
    _grid = {'country': globals().get('SELECTED_COUNTRY', ''),
             'slots': [_local_grid_slot(i, c) for i, c in enumerate(SELECTED_CAMS)]}
    _lg.write_text(json.dumps(_grid, indent=1), encoding='utf-8')
    print(f'Dashboard grid -> YOUR {len(SELECTED_CAMS)} picked cameras '
          f'({globals().get("SELECTED_COUNTRY", "")}): '
          f'{[s["placeholder_name"] for s in _grid["slots"]]}')
else:
    if _lg.exists():
        _lg.unlink()
    print('No cameras picked yet - the dashboard will show the cloud VM grid. '
          'Run the picker first to see YOUR cameras here.')

# Free-port auto-scan (8000..8020). Skips any port already listening so we
# never override an existing dashboard on localhost (a previous notebook
# run, a colleague's server, etc.). First run: 8000. Repeat runs: same
# port if still free, otherwise the next free one.
_main = sys.modules['__main__']
existing_srv = getattr(_main, '_dash_server', None)
existing_port = getattr(_main, '_dash_port', None)
if existing_srv is not None and existing_srv != 'external' and existing_port:
    DASHBOARD_PORT = existing_port
    print(f'Reusing existing dashboard server on port {DASHBOARD_PORT}.')
else:
    DASHBOARD_PORT = None
    for _candidate in range(8000, 8021):
        if port_is_free(_candidate):
            DASHBOARD_PORT = _candidate
            break
    if DASHBOARD_PORT is None:
        raise RuntimeError('No free port in 8000-8020 - close another dashboard first.')
    factory = lambda *a, **k: DashboardHandler(*a, directory=str(WEB_DIR), **k)
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads      = True
    srv = http.server.ThreadingHTTPServer(('', DASHBOARD_PORT), factory)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _main._dash_server = srv
    _main._dash_port = DASHBOARD_PORT
    if DASHBOARD_PORT != 8000:
        print(f'Port 8000 was busy; bound the dashboard to {DASHBOARD_PORT} instead.')
    print(f'Dashboard server started at http://localhost:{DASHBOARD_PORT}/?mode=twin')

# `?mode=twin` tells web/app.js which set of dashboard panels to render:
# see the "Dual-mode dashboard" comment block near the top of src/web/app.js.
dash_url = f'http://localhost:{DASHBOARD_PORT}/?mode=twin'
if not getattr(_main, '_dash_browser_opened', False):
    try:    webbrowser.open(dash_url, new=2)
    except Exception: pass
    _main._dash_browser_opened = True

from IPython.display import display, HTML, IFrame
display(HTML(f'<p><b>Live dashboard</b> (your picked cameras; counts from the '
             f'cloud collector): <a href="{dash_url}" target="_blank">{dash_url}</a></p>'))
display(IFrame(dash_url, width='100%', height=640))
```

<a id="cell-33"></a>
<div dir="rtl">

## חלק 8 - השוואת אתרים מסחריים מרובים

### תא 33 - markdown - כותרת + הסבר

**מה עושה:** משפט קצר: לולל את דגימת ה-footfall מעל כמה מצלמות כדי לדרג מיקומים לפי פעילות - הקלט להחלטת בחירת אתר.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 8. Compare multiple commercial sites

Loop the footfall sampler over several cameras to rank locations by activity - the input to a
site-selection decision.
```

<a id="cell-34"></a>
<div dir="rtl">

### תא 34 - code - דירוג המצלמות שנבחרו

**מה עושה:** checkpoint (`SELECTED_CAMS_APPLIED`), אחר-כך לולאה על כל cam ב-`SELECTED_CAMS`:

1. `c = CAMERAS.get(cid)`; אם לא בקטלוג או ללא URL: skip.
2. `url = resolve_stream(c)`; אם resolve נכשל: skip עם הודעה.
3. `grab_frame(url)`; אם None (‏geo-blocked / down): skip.
4. אחרת: `sdf = footfall_series(url, c['name'], interval_s=10, duration_min=0.5)` - חצי דקה של דגימה (~3 דגימות). שומר `{'site': c['name'], 'median_people': sdf['person'].median(), 'max_people': sdf['person'].max()}`.

בסוף: אם `summary` לא ריק - מציג `pd.DataFrame(summary).sort_values('median_people', ascending=False)`. אחרת: "No camera in SELECTED_CAMS produced usable frames".

**למה:** להוכיח את הכלי לצורך "השוואת מיקומים". הדגימה חצי-דקה קצרה מדי לניתוח מהיר-רק (חכמה של פרודקשן) אבל מספיקה להבחין בין `median_people=5` ל-`median_people=25`.

**אלטרנטיבות:** אפשר להריץ במקביל (‏multiprocessing) - חצי-דקה x 4 מצלמות = 2 דקות ברצף. הפרויקט בחר בבטחון וב-fail-tolerance במקום מקבילות.

**פלטים:** DataFrame ממוין:

```
              site  median_people  max_people
0   Taksim Square              15          22
1    Eminonu Sq.              12          18
2   Sarachane                   8          14
...
```

**שונה מהראשית?** לא, זהה. במעשה: הדירוג עצמו סביר להישאר יציב בין הראשית לתאומה - העדיפויות היחסיות אמינות כי הפגיעות של המודל החלש די אחידה מעל מצלמות.

</div>

```python
if not globals().get('SELECTED_CAMS_APPLIED'):
    class _ApplyFirst(Exception):
        def _render_traceback_(self):
            return ['PAUSED: run the picker cell above and enter your 4 camera numbers first.']
    raise _ApplyFirst()
# Rank the cameras YOU picked in verify7cams above (SELECTED_CAMS) by
# activity. This is a LOCAL analysis - it uses your dynamic pick, not
# the fixed VM grid. Cameras that can't resolve or return no frames are
# skipped, so if some picks are down you still get a partial ranking.
summary = []
for cid in SELECTED_CAMS:
    c = CAMERAS.get(cid)
    if not c or not c.get('url'):
        print(f'{cid}: skipped (not in catalog or no url)')
        continue
    try:
        url = resolve_stream(c)
    except Exception as e:
        print(f'{cid}: resolve failed ({e})')
        continue
    # one quick decode check before spending 30s on this camera
    if grab_frame(url) is None:
        print(f'{cid}: no frame from stream (geo-blocked / down). Skipping.')
        continue
    sdf = footfall_series(url, c['name'], interval_s=10, duration_min=0.5)
    summary.append({'site': c['name'],
                    'median_people': sdf['person'].median(),
                    'max_people': sdf['person'].max()})

if summary:
    pd.DataFrame(summary).sort_values('median_people', ascending=False)
else:
    print('No camera in SELECTED_CAMS produced usable frames - nothing to rank.')
```

<a id="cell-35"></a>
<div dir="rtl">

## חלק 9 - סיכום חי

### תא 35 - markdown - כותרת + הסבר

**מה עושה:** מסביר שהתא הבא מושך את כל מה שהמחברת ראתה בהרצה זו לבלוק אחד: האנומליות שסומנו מעל כל המצלמות שנדגמו, סך re-ID, וויזואליזציה קטנה שמציירת את כל האנומליות על ציר-זמן אחד. הרצה מחדש מרעננת את זה מאפס - אין חותמות זמן ישנות של סשן של מישהו אחר שמזליגות.

**שונה מהראשית?** לא, זהה.

</div>

```markdown
## 9. Live summary - what did we find?

Pulls everything the notebook saw on this run into a single block: the anomalies
flagged across every sampled camera, the re-ID totals, and a tiny visualisation
plotting all anomalies on the same timeline. Re-running the notebook regenerates
this from scratch - no stale timestamps from someone else's session leak through.
```

<a id="cell-36"></a>
<div dir="rtl">

### תא 36 - code - איסוף אנומליות + re-ID + גרף

**מה עושה:** תא סיכום ב-`try/except` שסופג כל exception וידפיס אותו במקום להפיל את המחברת. חמישה בלוקים:

1. **כותרת** - "Notebook run finished at ...", "Live camera for this run: ...".
2. **איסוף אנומליות** - מסתכל אם `df` קיים ומכיל עמודת `anomaly`; אם כן, מסנן שורות עם `anomaly == True`, מוסיף עמודת `cam`. מדפיס `Anomalies flagged (rolling z > 2.5): N` וטבלה. אם 0: "Too few samples for the z-score window to trip, or the scene was steady."
3. **re-ID roll-up** - אם `reid` קיים: מדפיס שוב את סטטיסטיקות המצלמה עם כותרת "Re-identification - {name}", 5 top returning.
4. **גרף always-on** - אם `df` קיים ולא ריק: פותח figure, plots `person` ו-`vehicles`, אם יש אנומליות מוסיף `scatter` אדום עם `marker='X'`, `s=160`, `zorder=5`; כותרת "This run: {name} ({N samples}, {M anomalies})", legend + grid.
5. **קישורים לפרודקשן** - הדפסת הפקודות: `collector: python -m app.collector --interval 20 --country turkey`, `dashboard: python serve.py`, `open: http://localhost:8000`.

הבלוק בנוי ב-try/except כדי שאם קודם משהו נכשל (למשל `reid` לא נטען) - עדיין נראה משהו.

**למה:** תא סיכום נדרש בכל מחברת מקצועית - מאפשר לקורא לחזור בסוף ולראות מה קרה. חשוב במיוחד למפעיל שרץ Run All: אל תרצה לגלול לאחור לפרק 4 לזכור אם נמצאה אנומליה.

**פלטים:** בלוק טקסטואלי ארוך + גרף בסוף.

**שונה מהראשית?** לא, זהה.

</div>

```python
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    print(f"Notebook run finished at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Live camera for this run: {cam['name']}")
    print("=" * 78)

    # ---- aggregate anomalies into one DataFrame ----
    anom_frames = []
    if "df" in dir() and isinstance(df, pd.DataFrame) and not df.empty and "anomaly" in df.columns:
        a = df[df["anomaly"] == True].copy()
        if not a.empty:
            a["cam"] = cam["name"]
            anom_frames.append(a[["ts", "cam", "person", "vehicles"]])
    anom = pd.concat(anom_frames, ignore_index=True) if anom_frames else \
           pd.DataFrame(columns=["ts", "cam", "person", "vehicles"])

    if len(anom):
        print(f"\nAnomalies flagged (rolling z > 2.5): {len(anom)}")
        print(anom.to_string(index=False))
    else:
        print("\nAnomalies flagged: 0")
        print("(Too few samples for the z-score window to trip, or the scene was steady.)")

    # ---- re-ID rollup ----
    if "reid" in dir():
        stats = reid.stats(CAM_ID)
        print("\n" + "-" * 78)
        print(f"Re-identification - {cam['name']}")
        print(f"  total unique entities   : {stats['total_unique']}")
        print(f"  total sightings         : {stats['total_sightings']}")
        for cls, s in stats["per_class"].items():
            print(f"    {cls:10s}  unique={s['unique']}  "
                  f"sightings={s['total_sightings']}  "
                  f"regulars(>=3)={s['regulars']}")
        regulars = reid.top_regulars(CAM_ID, n=5)
        if regulars:
            print("  top returning:")
            for r in regulars:
                print(f"    #{r['entity_id']:>4}  {r['cls']:8s}  "
                      f"sightings={r['sightings']}  last_seen={r['last_seen']}")

    # ---- always-on visual: footfall over this run + anomalies overlaid ----
    # Build the plot only when we have data; do NOT call ax.legend() on an empty
    # axes (that produces the "No artists with labels found" warning).
    if "df" in dir() and isinstance(df, pd.DataFrame) and not df.empty:
        ts = pd.to_datetime(df["ts"])
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(ts, df["person"],   marker="o", color="#4f8cff", label="people")
        ax.plot(ts, df["vehicles"], marker="s", color="#f0a35e", label="vehicles", alpha=0.85)
        if len(anom):
            ax.scatter(pd.to_datetime(anom["ts"]), anom["person"],
                       s=160, color="#ef4444", marker="X", zorder=5, label="anomaly")
        ax.set_title(f"This run: {cam['name']}  ({len(df)} samples, {len(anom)} anomalies)")
        ax.set_ylabel("count per frame"); ax.set_xlabel("timestamp (UTC)")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    print("\n" + "=" * 78)
    print("For the persistent shared HTML dashboard (Firestore-backed):")
    print("  collector  : run the cell in Section 7 of this notebook,")
    print("               or `python -m app.collector --interval 20 --country turkey`")
    print("  dashboard  : python serve.py        (from the project root)")
    print("  open       : http://localhost:8000  (opens automatically)")

except Exception as e:
    print(f"summary cell stopped early: {type(e).__name__}: {e}")
```

<a id="cell-37"></a>
<div dir="rtl">

## חלק 10 - קליברציית דיוק

### תא 37 - markdown - כותרת + workflow

**מה עושה:** מסביר את המשמעות: הדשבורד אמין רק ככל ש-YOLO מדויק על **המצלמות האלו**. הפרק מודד את זה: לוכד פריימים מ-4 המצלמות החיות של הגריד, מריץ את הזיהוי בשני גדלי קלט (‏640 = ברירת מחדל ישנה, 960 = ברירת מחדל הקולקטור הנוכחי), אחר-כך אתה סופר people/vehicles בעצמך ומקבל MAE + bias פר מצלמה ופר גודל.

Workflow (הכל מקומי, ~10 דקות של תיוג):
1. **10a** לוכד פריימים + תחזיות ל-`data/calibration/`;
2. **10b** מציג כל פריים - אתה מקליד `true people,vehicles`;
3. **10c** מדפיס טבלת דיוק והמלצת `conf/imgsz`.

הזן את התוצאה חזרה: `imgsz` המנצח הולך ל-`--imgsz` של הקולקטור, ומצלמה עם bias שיטתית מקבלת `"conf"` override ב-`app/cameras.py` (‏bias < 0 -> הורד conf, bias > 0 -> העלה).

**שונה מהראשית?** לא, זהה בטקסט. **המשמעות מהותית שונה:** בתאומה, ה-"ground truth" של הקליברציה (מה שאתה סופר בעיניים) הוא בדיוק מה שהמודל של הגרסה החזקה `yolo26m` היה מוצא. `imgsz=960` בהיפותזה לא ממש מציל כאן - `yolov8s@960` עדיין חלש יותר מ-`yolo26m@960`. הפרק שימושי לתאומה בעיקר להראות **כמה** ה-VM חסר, לא לתקן את זה.

</div>

```markdown
## 10. Accuracy calibration - how good are the counts, really?

The dashboard is only as trustworthy as YOLO is on THESE cameras. This section
measures it: capture frames from the 4 live grid cameras, run the detector at
two input sizes (640 = old default, 960 = the collector's current default),
then count people/vehicles yourself and get MAE + bias per camera and per size.

Workflow (all local, ~10 minutes of counting):
1. **10a** captures frames + predictions into `data/calibration/`;
2. **10b** shows each frame - you type the true `people,vehicles`;
3. **10c** prints the accuracy table and a conf/imgsz recommendation.

Feed the result back into the pipeline: the winning `imgsz` goes to the
collector's `--imgsz`, and a camera with a systematic bias gets a `"conf"`
override in `app/cameras.py` (bias < 0 -> lower conf, bias > 0 -> raise it).
```

<a id="cell-38"></a>
<div dir="rtl">

### תא 38 - code - 10a: capture פריימים + ריצת YOLO בשני imgsz

**מה עושה:** checkpoint, ואז:

1. **קבועים** - `CALIB_DIR = DATA_DIR / 'calibration'` (נוצרת אם חסרה), `FRAMES_PER_CAM = 6` (4 מצלמות x 6 פריימים = 24 לתיוג), `IMG_SIZES = (640, 960)` (הישן מול הנוכחי), `CALIB_CONF = 0.30` (מסונכרן עם `--conf` של הקולקטור).
2. **לולאה על כל מצלמה** ב-`SELECTED_CAMS`: מנסה `resolve_stream`, אם נכשל -> skip. אחרת לולאה של `FRAMES_PER_CAM` פריימים:
   - `frames = grab_burst(url, n=1)` - פריים בודד; אם `None` -> MISS + continue.
   - שמירת הפריים הגולמי ל-`{stem}.jpg` (`stem = f'{cam_id}_{k:02d}'`).
   - לכל `size` ב-`IMG_SIZES`: `counts, _ = detect_with_boxes(model, frame, conf=CALIB_CONF, imgsz=size)`. שדות `entry['person_640'], entry['vehicles_640'], entry['person_960'], entry['vehicles_960']`.
   - שמירת annotation `{stem}_annotated.jpg` עם `annotate(model, frame, conf=CALIB_CONF, imgsz=max(IMG_SIZES))` (על ההגדרה החזקה יותר).
   - `samples.append(entry)`, `got += 1`, `time.sleep(2)` להתקדמות הזרם.
3. **שמירה** - `predictions.json` = רשימת כל ה-entries.

**למה:** להפריד את שלב הלכידה (עולה זמן, לא דורש interactivity) מהתיוג (זמן קצר של אינטראקציה). כשמפצלים כך אפשר להריץ 10a בשקט ב-background ואז לפתוח את 10b כשמוכן.

**אלטרנטיבות:** לגלגל את כל שלושה השלבים בתא אחד היה נחמד יותר, אבל אז המפעיל שסגר את החלון וחוזר מחר צריך לרוץ את הלכידה מחדש. הפרדה חוסכת עבודה.

**פלטים:**

```
ibb_taksim: captured 6 frames
ibb_eminonu: captured 6 frames
...
24 frames -> C:\Users\...\data\calibration
```

+ 24 קבצי `.jpg` גולמיים, 24 קבצי `_annotated.jpg`, ו-`predictions.json`.

**שונה מהראשית?** לא, זהה.

</div>

```python
if not globals().get('SELECTED_CAMS_APPLIED'):
    class _ApplyFirst(Exception):
        def _render_traceback_(self):
            return ['PAUSED: run the picker cell above and enter your 4 camera numbers first.']
    raise _ApplyFirst()
# --- 10a. Capture calibration frames + predictions (run once, ~1-2 min) ---
import json as _json
from app.detect_core import grab_burst, detect_with_boxes, annotate

CALIB_DIR = DATA_DIR / 'calibration'; CALIB_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_PER_CAM = 6          # 4 cams x 6 frames = 24 to label (aim for 20-30)
IMG_SIZES = (640, 960)      # old default vs the collector's current default
CALIB_CONF = 0.30           # keep in sync with the collector's --conf

samples = []
for cam_id in (SELECTED_CAMS if globals().get('SELECTED_CAMS')
               else GRID_CAMERAS):
    cam = CAMERAS[cam_id]
    try:
        url = resolve_stream(cam)
    except Exception as e:
        print(f'{cam_id}: resolve failed ({e}) - skipping'); continue
    got = 0
    for k in range(FRAMES_PER_CAM):
        frames = grab_burst(url, n=1)
        if not frames:
            print(f'{cam_id}: frame {k} MISS'); continue
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
        samples.append(entry); got += 1
        time.sleep(2)   # let the live stream move on a little between captures
    print(f'{cam_id}: captured {got} frames')

(CALIB_DIR / 'predictions.json').write_text(_json.dumps(samples, indent=2))
print(f'{len(samples)} frames -> {CALIB_DIR}')
```

<a id="cell-39"></a>
<div dir="rtl">

### תא 39 - code - 10b: תיוג אינטראקטיבי

**מה עושה:** תא אינטראקטיבי מוגן ב-flag: `RUN_LABELING = globals().get('RUN_LABELING', False)`. ברירת המחדל False - כדי שלא יעצור Run All עם `input()`. כשמוכן לתייג: לקבוע `RUN_LABELING = True` ולרוץ שוב.

כשמופעל:
1. טוען `samples` מ-`predictions.json`.
2. לכל דגימה:
   - קורא `_annotated.jpg`, ממיר לתצוגת RGB, מציג ב-matplotlib עם כותרת `"{stem} | model@960: person=X vehicles=Y"`.
   - מקבל `input(f"{stem} true 'people,vehicles' (Enter=skip, q=stop): ")`.
   - `q` = יציאה מהלולאה. Enter ריק = skip. אחרת מנסה לפרסר `p_true, v_true = int/int`. אם ולידציה נכשלת: skip.
   - מוסיף ל-`labeled` את כל שדות `s` + `person_true, vehicles_true`.
3. שומר `labeled.json`.

**למה:** התיוג הידני הוא החלק הבלתי-ניתן-לאוטומציה של קליברציה - **אתה** משמש כ-ground truth. הצפייה בפריים המסומן (ולא בפריים הגולמי) עוזרת: אתה רואה מה המודל *חשב* שהוא רואה, ומקליד את מה שאתה **באמת** רואה.

**אלטרנטיבות:** widget של ipywidgets היה נחמד יותר, אבל `input()` בטוח על פני כל סביבות ה-Jupyter. סימוני mouse-click בתמונה היו יעילים יותר אבל דורשים JS/canvas והרבה יותר קוד.

**פלטים:**

```
Labeling skipped (RUN_LABELING is False).
When you want to label: set RUN_LABELING = True in this cell
and run it again - it will show each frame and ask for the
true people,vehicles counts.
```

או במצב אמיתי: 24 תמונות + 24 קלטים + `labeled 24 frames -> .../labeled.json`.

**שונה מהראשית?** לא, זהה.

</div>

```python
# INTERACTIVE cell - it asks YOU to type counts, so it must never run
# under Run All (it would stall the whole run waiting for keyboard
# input). Set RUN_LABELING = True and re-run this cell when you are
# ready to label; leave it False for hands-off runs.
RUN_LABELING = globals().get('RUN_LABELING', False)
if not RUN_LABELING:
    print('Labeling skipped (RUN_LABELING is False).')
    print('When you want to label: set RUN_LABELING = True in this cell')
    print('and run it again - it will show each frame and ask for the')
    print('true people,vehicles counts.')
else:
    # --- 10b. Label: look at each frame, type the true counts ---
    # The annotated image shows what the model saw at imgsz=960. Count what YOU
    # see: people, then vehicles (cars+buses+trucks+motorbikes+bicycles), and type
    # `people,vehicles` (e.g. `7,3`). Enter = skip frame, q = stop early.
    import json as _json

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
            print('  could not parse - skipped'); continue
        labeled.append({**s, 'person_true': p_true, 'vehicles_true': v_true})

    (CALIB_DIR / 'labeled.json').write_text(_json.dumps(labeled, indent=2))
    print(f'labeled {len(labeled)} frames -> {CALIB_DIR / "labeled.json"}')
```

<a id="cell-40"></a>
<div dir="rtl">

### תא 40 - code - 10c: דוח MAE + bias

**מה עושה:** טוען `labeled.json`. אם ריק: זורק `_LabelFirst` בהודעה ידידותית ("PAUSED: no labeled frames yet - set RUN_LABELING = True in 10b, label a few frames, then run this cell again").

בונה `cal` = DataFrame מכל השורות. אז שתי טבלאות:

1. **`overall`** - לכל `(size, metric)`: `err = cal[f'{metric}_{size}'] - cal[f'{metric}_true']`. שומר `MAE = |err|.mean()`, `bias = err.mean()` (שלילי = under-count), `n = |cal|`.
2. **`per_cam`** - לכל `cam_id`, לכל `(metric)` ב-`imgsz=max(IMG_SIZES)` (960): אותם MAE + bias פר מצלמה.

מדפיס את השתיים כטבלאות. אחר-כך בלוק הסבר איך לקרוא:
- `bias < 0` -> under-count שיטתי: הורד את conf של המצלמה (`"conf": 0.25` ב-`app/cameras.py`). `bias > 0` -> over-count: העלה.
- אם `MAE@960 < MAE@640` (טיפוסי לצילומים רחבים) - שמור `--imgsz 960`; אחרת חזור ל-640.
- הרץ שוב אחרי כל החלפת מצלמה או שינוי משקולות.

**למה:** התא הזה הוא **הנקודה** של פרק 10. הוא הופך תיוג לפלט אופרטיבי - שינויים ספציפיים ל-`app/cameras.py` ו-`--imgsz` של הקולקטור.

**פלטים:**

```
=== overall (all cameras) ===
 imgsz  metric   MAE  bias  n
   640  person  4.2  -3.1  24
   640 vehicles 5.5  -2.8  24
   960  person  1.8  -0.4  24
   960 vehicles 2.1  -0.9  24

=== per camera @ imgsz=960 ===
      cam  metric  MAE  bias  n
ibb_tak..  person  0.5   0.2  6
ibb_tak.. vehicles 1.0  -0.5  6
...
```

**שונה מהראשית?** לא בקוד. בפועל: התאומה תראה MAE גבוה יותר וב-bias שלילי חזק יותר, בפירוש כי `yolov8s` מגלה פחות. זו הוכחה מספרית לפער בין המחברת הראשית לבין ה-VM.

</div>

```python
# --- 10c. Accuracy report: MAE + bias per input size and per camera ---
import json as _json

rows = _json.loads((CALIB_DIR / 'labeled.json').read_text())
if not rows:
    class _LabelFirst(Exception):
        def _render_traceback_(self):
            return ['PAUSED: no labeled frames yet - set RUN_LABELING = True '
                    'in 10b, label a few frames, then run this cell again.']
    raise _LabelFirst()
cal = pd.DataFrame(rows)

overall = []
for size in IMG_SIZES:
    for metric in ('person', 'vehicles'):
        err = cal[f'{metric}_{size}'] - cal[f'{metric}_true']
        overall.append({'imgsz': size, 'metric': metric,
                        'MAE': round(err.abs().mean(), 2),
                        'bias': round(err.mean(), 2),   # negative = undercount
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
print(f'=== per camera @ imgsz={best} ===')
print(pd.DataFrame(per_cam).to_string(index=False))

print("""
How to read this:
- bias < 0 -> systematic undercount: lower that camera's conf (add e.g.
  "conf": 0.25 to its entry in app/cameras.py). bias > 0 -> overcount: raise it.
- If MAE@960 < MAE@640 (typical for these wide shots), keep the collector's
  default --imgsz 960; otherwise fall back to 640.
- Re-run this section after any camera swap or weights change.""")
```

<a id="cell-41"></a>
<div dir="rtl">

## חלק 11 - חיזוי

### תא 41 - markdown - כותרת + הפילוסופיה: רק המנצח האופרטיבי

**מה עושה:** תא ההקדמה החשוב ביותר בתאומה - כותב מפורש: **"The VM-faithful version of the forecasting chapter"**. מסביר את חלוקת התפקידים:

- **המחברת המקומית** (הראשית) היא המעבדה: מריצה סולם של מודלים (persistence, seasonal-naive, profile, ridge, GRU) ומראה איזה מהם מרוויח את הזמן שלו.
- **המחברת הזאת** (התאומה) שומרת רק את **המנצח האופרטיבי** מקטגוריית הבוחר - חיזוי profile של שעה-של-יום/שבוע עם תיקון EWMA לרמה ורצועת אי-ודאות MAD. numpy טהור, ללא תלויות חדשות, אלפיות שנייה של חישוב: בדיוק מה שה-e2-micro של 1 GB יכול להרשות לעצמו להריץ ליד הקולקטור.

שני תאים של סנטירת (זהים למחברת המקומית - ה-CSV cache משותף), ואז המחזה עצמו:

- **עקומה צפויה** - לכל מצלמה עם מספיק היסטוריה: איך 12 השעות הבאות אמורות להיראות, שעה שעה.
- **אנומליות סטייה** - `|actual - expected|` ביחידות MAD מסמן גם spikes וגם שקט חריג (כיכר מתה בשעת שיא). rolling-z של חלק 4 עיוור למקרה השקט מעצם התכנון; הזה משלים אותו.

Firestore זוכרת רק כמה ימים אחורה, לכן תא המשיכה למטה הוא גם הארכיבר של הפרויקט: הרץ אותו כל כמה ימים ו-`data/footfall_history.csv` צובר את ההיסטוריה המלאה.

**שונה מהראשית?** מהותית. הראשית מציגה **סולם שלם של מודלים**, מבצעת back-tests, ומדפיסה טבלה שמשווה persistence / seasonal-naive / profile / ridge / GRU. התאומה מפילה את כל זה ומשאירה **רק** את ה-profile - כי זה מה שהיה נבחר בפועל אחרי backtests, וזה מה שיכול לרוץ על VM זעיר. הפילוסופיה: המחברת הראשית מוכיחה WHY, התאומה מריצה WHAT.

</div>

```markdown
## 11. Forecasting - expected activity + deviation anomalies (operational)

The VM-faithful version of the forecasting chapter. The local notebook
(`turkey_business_activity.ipynb`) is the lab: it backtests a ladder of
models (persistence, seasonal-naive, profile, ridge, GRU) and shows which
one earns its keep. THIS notebook keeps only the operational winner-class
forecaster - an hour-of-day/week median profile with an EWMA level
correction and a MAD uncertainty band. Pure numpy, no new dependencies,
milliseconds of compute: exactly what the 1 GB e2-micro could afford to
run next to the collector.

Two cells of plumbing (identical to the local notebook - the CSV cache is
shared), then the forecaster:

* **Expected curve** - for every camera with enough history: what the
  next 12h should look like, hour by hour.
* **Deviation anomalies** - |actual - expected| in MAD units flags both
  spikes AND unusual quiet (a dead square at rush hour). Section 4's
  rolling-z is blind to the quiet case by construction; this detector is
  the missing half.

Firestore only remembers a few days back, so the fetch cell below is also
the project's archivist: run it every couple of days and
`data/footfall_history.csv` accumulates the full history.
```

<a id="cell-42"></a>
<div dir="rtl">

### תא 42 - code - 11a: משיכת ההיסטוריה מ-Firestore ל-cache מקומי

**מה עושה:** תא ארוך שמושך את היסטוריית ה-footfall של הקולקטור ל-cache מקומי ב-`data/footfall_history.csv`, incrementally. שלושה בלוקים עיקריים:

1. **קבועים + חשבונאות מכסות** - `READ_BUDGET = 30_000`, `PAGE = 5_000`. הערה בקוד: "Firestore quota math (free tier = 50k document reads/day): the whole collection is currently ~14k docs, so the first backfill costs ~14k reads - fine. Later runs fetch only docs newer than the cache's max ts (a few hundred). READ_BUDGET hard-caps a single run; if it truncates, the next run continues from where this one stopped."
2. **`_find_credentials()`** - מחפש קובץ SA json: קודם משתני סביבה (`FIREBASE_CREDENTIALS`, `GOOGLE_APPLICATION_CREDENTIALS`), אחר-כך קבצים בתבנית `*firebase-adminsdk*.json` ב-`_src_dir.parent`, `_src_dir`, `cwd`.
3. **`fetch_footfall_delta()`** - הליבה:
   - טוען `cached = pd.read_csv(CACHE)` אם קיים; `since = cached['ts'].max()` (מחרוזות ISO מסתדרות נכון).
   - אם אין credentials אבל יש cache -> משתמש בקיים בלבד.
   - אחרת יוזם `firebase_admin`, בונה query `db.collection('footfall').where('ts', '>', since)` אם `since` קיים.
   - `n_new = q.count().get()[0][0].value` - aggregation יעיל (~1 read לכל 1000 docs).
   - מדפיס מצב cache + מספר docs חדשים.
   - לולאת דפים: `while fetched < min(n_new, READ_BUDGET)`, כל דף `PAGE=5000` docs, `order_by('ts').limit(...)`. לכל doc מפריט את `ts, cam_id, person, vehicles, ok, night`.
   - ממזג עם `cached`, מנקה: `dropna(subset=['ts','cam_id']).drop_duplicates(...).sort_values('ts')`.
   - שומר ל-CSV, מדפיס `cache now X rows, YYYY-MM-DD -> YYYY-MM-DD, N cameras -> {name}`.

לבסוף: `hist = fetch_footfall_delta()`. `hist.groupby('cam_id').agg(rows=..., ok=..., first=..., last=...).sort_values('rows', ascending=False).head(12)` - סיכום למצלמה של 12 עם הכי הרבה שורות.

**למה:** Firestore ב-Spark plan מגבילה 50K reads/day. הפרויקט מרים archive מקומי ב-CSV שגדל incrementally - הרץ פעם ביומיים וקבל היסטוריה מלאה מבלי לפוצץ את המכסה.

**אלטרנטיבות:** BigQuery / GCS export של Firestore היה נותן ארכיון עמיד יותר, אבל: עולה כסף וזמן; CSV מקומי מספיק לניתוח היסטורי.

**פלטים:**

```
cache: 12500 rows (through 2026-08-13T05:12)
new docs on the server: 340
cache now 12840 rows, 2026-07-20 -> 2026-08-13, 8 cameras -> footfall_history.csv
```

+ טבלה של 12 מצלמות.

**שונה מהראשית?** לא, זהה מילולית. הערה בהקדמה של פרק 11: "identical to the local notebook - the CSV cache is shared". שני המחברות טוענות מאותו CSV.

</div>

```python
# 11a. Fetch the collector's footfall history -> local CSV cache (incremental).
#
# Firestore quota math (free tier = 50k document reads/day): the whole
# collection is currently ~14k docs, so the first backfill costs ~14k reads
# - fine. Later runs fetch only docs newer than the cache's max ts (a few
# hundred). READ_BUDGET hard-caps a single run; if it truncates, the next
# run continues from where this one stopped.
import glob

CACHE = DATA_DIR / 'footfall_history.csv'
READ_BUDGET = 30_000
PAGE = 5_000

def _find_credentials():
    """SA json discovery: env vars first, then the project layout."""
    import os
    for env in ('FIREBASE_CREDENTIALS', 'GOOGLE_APPLICATION_CREDENTIALS'):
        p = os.environ.get(env)
        if p and Path(p).is_file():
            return p
    for base in (_src_dir.parent, _src_dir, Path.cwd()):
        hits = sorted(glob.glob(str(base / '*firebase-adminsdk*.json')))
        if hits:
            return hits[0]
    return None

def fetch_footfall_delta():
    import firebase_admin
    from firebase_admin import credentials as fb_credentials, firestore

    cached = None
    since = None
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        if len(cached):
            since = cached['ts'].max()          # ISO strings compare correctly

    cred_path = _find_credentials()
    if cred_path is None:
        if cached is not None:
            print('no Firebase credentials found - working from the existing '
                  f'cache ({len(cached)} rows).')
            return cached
        raise RuntimeError('no Firebase credentials and no cache yet: put the '
                           'service-account json at the project root, or set '
                           'FIREBASE_CREDENTIALS.')
    if not firebase_admin._apps:
        firebase_admin.initialize_app(fb_credentials.Certificate(cred_path))
    db = firestore.client()

    q = db.collection('footfall')
    if since:
        q = q.where('ts', '>', since)
    n_new = int(q.count().get()[0][0].value)      # aggregation: ~1 read/1000 docs
    print(f'cache: {0 if cached is None else len(cached)} rows'
          + (f' (through {since[:19]})' if since else ' (empty)'))
    print(f'new docs on the server: {n_new}'
          + (f' - fetching first {READ_BUDGET} (budget)' if n_new > READ_BUDGET else ''))

    rows, cursor, fetched = [], since, 0
    while fetched < min(n_new, READ_BUDGET):
        page_q = db.collection('footfall')
        if cursor:
            page_q = page_q.where('ts', '>', cursor)
        page = list(page_q.order_by('ts').limit(min(PAGE, READ_BUDGET - fetched)).stream())
        if not page:
            break
        for doc in page:
            d = doc.to_dict()
            rows.append({'ts': d.get('ts'), 'cam_id': d.get('cam_id'),
                         'person': d.get('person'), 'vehicles': d.get('vehicles'),
                         'ok': d.get('ok'), 'night': d.get('night')})
        cursor = rows[-1]['ts']
        fetched += len(page)
        print(f'  fetched {fetched}...', end='\r')

    fresh = pd.DataFrame(rows)
    df = (pd.concat([cached, fresh], ignore_index=True)
          if cached is not None and len(fresh) else
          (fresh if len(fresh) else cached))
    df = (df.dropna(subset=['ts', 'cam_id'])
            .drop_duplicates(subset=['ts', 'cam_id'])
            .sort_values('ts', ignore_index=True))
    df.to_csv(CACHE, index=False)
    print(f'cache now {len(df)} rows, '
          f'{df.ts.min()[:10]} -> {df.ts.max()[:10]}, '
          f'{df.cam_id.nunique()} cameras -> {CACHE.name}')
    return df

hist = fetch_footfall_delta()
hist.groupby('cam_id').agg(rows=('ts', 'size'), ok=('ok', 'sum'),
                           first=('ts', 'min'), last=('ts', 'max')) \
    .sort_values('rows', ascending=False).head(12)
```

<a id="cell-43"></a>
<div dir="rtl">

### תא 43 - code - 11b: resample לרשת 15 דקות + סינון epoch

**מה עושה:** תא שלב-ההכנה של החיזוי. הופך את הדגימות הגולמיות (כל 40-90 שניות עם jitter וחורים) לרשת נקייה של bins 15 דקות פר מצלמה. חמישה בלוקים:

1. **קבועים** - `BIN_MIN = 15`, `BINS_PER_DAY = 96`, `MIN_HOURS = 36` (מצלמה צריכה >=36 שעות של bins נצפים כדי לשחק).
2. **סינון + resample** - `ok = hist[hist.ok == 1]` (רק דגימות מוצלחות), `ts` -> `datetime UTC`. `bins = ok.groupby('cam_id').resample('15min').median()` על person/vehicles. `bins['n']` = size = מספר דגימות פר bin. ההערה: "median, not mean, so one hallucinated burst can't drag a bin".
3. **פיצ'רים של זמן מקומי** - `bins['local'] = bins['ts']`; לכל מצלמה מנסה `tz_convert(cam_tzinfo(cam))` (Bangkok evening != Istanbul evening); אם נכשל: נשאר UTC. יוצר `hod` (‏hour-of-day bin), `dow` (‏day-of-week), `how = dow * 96 + hod` (‏hour-of-week bin).
4. **‏Era detection** - הבלוק החשוב. האוסף מחזיק שאריות של epoch ישן יותר (הגריד המקורי של יוני) מופרד מה-era הנוכחי בחור-רב-שבועות (הקולקטור סייר במדינות אחרות וישנים נוקו). ערבוב epochs חוצה חור כזה שובר כל split מבוסס-זמן (train יהיה עולם אחד, test אחר), לכן כל המידול משתמש רק ב-**era האחרון**: כל מה שאחרי החור האחרון > `ERA_GAP_H = 72` שעות.
   - `_obs_times = pd.Series(bins.loc[bins.n > 0, 'ts'].sort_values().unique())`.
   - `_breaks = _obs_times[_obs_times.diff() > pd.Timedelta(hours=72)]`.
   - אם קיים breaks: `era_start = _breaks.iloc[-1]`; מוריד את כל bins לפני זה, מדפיס `era cut: modeling on data from ... (dropped N bins of an older epoch across a >72h hole)`.
5. **סינון מצלמות זכאיות + מפת כיסוי** - `span = obs.groupby('cam_id').agg(bins_obs, first, last)`, `span_days`. `ELIGIBLE = sorted(...)` = מצלמות עם >= `MIN_HOURS * 4` bins. אחר-כך `cov = pivot_table` (‏cam x day, values=n bins/day) - יוצר heatmap ירוק (‏YlGn) שמראה כיסוי לפי יום.

**למה:** לגרוב את הזרם הגולמי לצורה שהמודל יכול לעבוד עליה. חיתוך epoch הוא הכי מעניין: הפרויקט למד מהמהלכים של הגריד וקידק ש-CI/CD מודל מקומי לא יערבב יקיצות של הקולקטור עם השבתות ארוכות.

**אלטרנטיבות:** interpolation על הפערים היה פשוט יותר אבל שקרי - הצפצוף המדומה יסתיר את החור האמיתי. הבחירה: NaN במקום אינטרפולציה, era-cut במקום שילוב.

**פלטים:**

```
era cut: modeling on data from 2026-08-05 12:00 on (dropped 2380 bins of an older epoch across a >72h hole)
eligible cameras (>= 36h observed): ['ibb_eminonu', 'ibb_taksim', 'ibb_sarachane', 'tvkur_hukumet']
                  bins_obs               first                last  span_days
cam_id
ibb_taksim            432 2026-08-05 12:15 2026-08-13 05:00        7.7
...
```

+ heatmap כיסוי.

**שונה מהראשית?** לא, זהה. שוב, ההערה: "identical to the local notebook - the CSV cache is shared".

</div>

```python
# 11b. Resample to a clean 15-minute grid per camera + eligibility filter.
#
# Raw samples arrive every 40-90s with jitter, gaps and camera swaps; every
# model below works on 15-min bins (median count per bin - median, not
# mean, so one hallucinated burst can't drag a bin). Bins with no samples
# stay NaN - models must handle absence honestly, not invent zeros.
from app.collector import cam_tzinfo

BIN_MIN = 15
BINS_PER_DAY = 24 * 60 // BIN_MIN                       # 96
MIN_HOURS = 36            # a camera needs >= 36h of observed bins to play

ok = hist[hist.ok == 1].copy()
ok['ts'] = pd.to_datetime(ok['ts'], utc=True, format='ISO8601')
g = ok.set_index('ts').sort_index().groupby('cam_id')
bins = g[['person', 'vehicles']].resample(f'{BIN_MIN}min').median()
bins['n'] = g['person'].resample(f'{BIN_MIN}min').size()
bins = bins.reset_index()

# Local-time features (a Bangkok evening is not an Istanbul evening).
bins['local'] = bins['ts']
for cam in bins.cam_id.unique():
    try:
        tz = cam_tzinfo(cam)
        m = bins.cam_id == cam
        bins.loc[m, 'local'] = bins.loc[m, 'ts'].dt.tz_convert(tz)
    except Exception:
        pass                                     # legacy cam ids: keep UTC
bins['hod'] = bins.local.dt.hour * (60 // BIN_MIN) + bins.local.dt.minute // BIN_MIN
bins['dow'] = bins.local.dt.dayofweek
bins['how'] = bins.dow * BINS_PER_DAY + bins.hod          # hour-of-week bin

# Era detection: the collection holds remnants of an OLDER epoch (June's
# original grid) separated from the current era by a multi-week hole - the
# collector toured other countries and old docs were purged. Mixing epochs
# across such a hole breaks every time-based split (train would be one
# world, test another), so all modeling below uses only the LAST
# CONTIGUOUS ERA: everything after the most recent gap > ERA_GAP_H hours.
ERA_GAP_H = 72
_obs_times = pd.Series(bins.loc[bins.n > 0, 'ts'].sort_values().unique())
_breaks = _obs_times[_obs_times.diff() > pd.Timedelta(hours=ERA_GAP_H)]
if len(_breaks):
    era_start = _breaks.iloc[-1]
    dropped = int((bins.ts < era_start).sum())
    bins = bins[bins.ts >= era_start].copy()
    print(f'era cut: modeling on data from {era_start:%Y-%m-%d %H:%M} on '
          f'(dropped {dropped} bins of an older epoch across a '
          f'>{ERA_GAP_H}h hole)')

obs = bins[bins.n > 0]
span = obs.groupby('cam_id').agg(bins_obs=('n', 'size'),
                                 first=('ts', 'min'), last=('ts', 'max'))
span['span_days'] = (span['last'] - span['first']).dt.total_seconds() / 86400
ELIGIBLE = sorted(span[span.bins_obs >= MIN_HOURS * 60 // BIN_MIN].index)
print(f'eligible cameras (>= {MIN_HOURS}h observed): {ELIGIBLE}')
print(span.loc[ELIGIBLE].round(1))

# Coverage picture: which camera has data when - the grid's country
# switches and stream outages are visible as holes.
cov = (obs[obs.cam_id.isin(ELIGIBLE)]
       .assign(day=lambda d: d.ts.dt.floor('D'))
       .pivot_table(index='cam_id', columns='day', values='n', aggfunc='size')
       .fillna(0))
fig, ax = plt.subplots(figsize=(11, 0.45 * max(2, len(cov))))
im = ax.imshow(cov.values, aspect='auto', cmap='YlGn', vmin=0)
ax.set_yticks(range(len(cov)), cov.index)
ax.set_xticks(range(len(cov.columns)), [f'{c:%d.%m}' for c in cov.columns],
              rotation=90, fontsize=7)
ax.set_title('coverage: observed 15-min bins per day (empty = stream dark / other country)')
fig.colorbar(im, label='bins/day')
plt.tight_layout(); plt.show()
```

<a id="cell-44"></a>
<div dir="rtl">

### תא 44 - code - 11c-VM: profile x EWMA + band

**מה עושה:** הליבה של פרק החיזוי - **המנצח האופרטיבי**. numpy בלבד. חמישה בלוקים:

1. **קבועים** - `Z_FLAG=3.5, MIN_DELTA=3.0, EWMA_A=0.25, HOLD_FRAC=0.25`.
2. **הכנה** - `work = bins[bins.cam_id.isin(ELIGIBLE)]`. `work['y'] = groupby('cam_id')['person'].transform(ffill(limit=2))` - מילוי קדימה עד 2 bins (לא יותר, כי גרירת ערך ישן על חור ארוך שקרית). `t_split = work.ts.min() + (work.ts.max()-work.ts.min()) * (1 - 0.25)` - 25% אחרונים = holdout. `span_days_all`; `key = 'how' if >= 7 days else 'hod'` - אם מספיק היסטוריה: פרופיל שעה-של-שבוע (168 סלוטים); אחרת: פרופיל שעה-של-יום בלבד (24 סלוטים).
3. **בניית הפרופיל + MAD** - `prof = work[work.ts <= t_split].groupby(['cam_id', key]).y.median()` - החציון פר (מצלמה, שעה של יום/שבוע). `res['exp0']` = לכל שורה, ה-`prof.get((c, s), NaN)`. `mad = ((res.y - res.exp0).abs().groupby(res.cam_id).median() * 1.4826).clip(lower=1.0)` - MAD כללי לכל מצלמה. מדפיס `profile key: how (hour-of-week; 12.3 days of history)`.
4. **`expected_series(dfc, cam)`** - הפונקציה שממש מייצרת חיזוי מתוקן:
   - `e0 = np.array([prof.get((cam, s), NaN) for s in dfc[key]])` - הפרופיל הגולמי.
   - לוללת ה-EWMA: `lvl = 1.0`. לכל שורה: `out.append(e * lvl if e is finite else NaN)`; אם `a` ו-`e` finite ו-`e >= 1`: `lvl = (1-A)*lvl + A * clip(a/e, 0.4, 2.5)`. הרעיון: הפרופיל מספר על **צורה** יחסית, ה-`lvl` מתקן את **הרמה** בזמן אמת. `clip(0.4, 2.5)` מונע אם או חד של דגימה יחידה מקרעת את הרמה.
5. **‏Mini-eval + גרפים + summary עתידי** - שלושה חלקים:
   - **Backtest קטן על holdout** - לכל אופק `h` ב-`(4, 48)` (‏1h ו-12h קדימה): לכל מצלמה, `tgt = dfc.y.shift(-h)`, `tst = dfc.ts > t_split`. משווה 3 מודלים: `persistence` (‏y חצי בעבר), `seasnaive24` (‏y.shift(96-h) = יום קודם), `profile`. `mae = (pred[tst] - tgt[tst]).abs().mean()`. מדפיס טבלה של MAE פר אופק.
   - **גרפים actual vs expected** - `show` = 4 מצלמות הכי טריות. לכל אחת: 36 שעות אחרונות, `expected_series`, `band = 3.5 * mad[cam]`. `fill_between(np.clip(exp-band, 0, None), exp+band)` - רצועה סביב הצפוי. `plot exp` (‏dashed), `plot actual`. scatter אדום על spikes (`resid > 0`), scatter סגול עם `marker='v'` על שקט חריג (`resid < 0`). הצבע הסגול הוא **המקרה הסטטי-שקט** שחלק 4 לא יכול לתפוס.
   - **‏Next 12h summary** - `future = date_range(work.ts.max()+15min, periods=48, freq='15min')`. לכל cam: המרה לזמן מקומי, `slots = weekday*96 + hour*4 + minute//15` (או רק hour*4+minute//15 ב-hod), `exp_f = np.array([prof.get((cam, s), NaN) for s in slots])`. אם finite: `i = nanargmax(exp_f)`, שומר `peak_local, peak_people, now_expected`. מדפיס DataFrame - השורה שדוח יומי עתידי יכול לצטט ("tomorrow's expected peak at Taksim ~14:00").

**למה:** התא הזה הוא הבשורה של פרק 11 בתאומה. זו הוכחה שמודל פשוט מאוד (חציון + EWMA + MAD) יכול לגלות אנומליות בשני הכיוונים ולהוציא תחזית 12 שעות שאפשר להכניס לדוח יומי. numpy = מסוגל לרוץ ליד הקולקטור על 1 GB RAM.

**אלטרנטיבות:** ARIMA/Prophet/GRU נבחנו במחברת הראשית ובוזבזו כי הם דורשים torch/statsmodels ופיצוץ הזיכרון. הפרופיל פשוט + EWMA לרוב תואם אותם על אופק קצר, ולא נופל כשמצלמה נופלת יום.

**פלטים:**

```
profile key: how (hour-of-week; 8.4 days of history)
             persistence  seasnaive24  profile
h
15min                2.3          2.1     1.9
720min               4.7          3.8     3.1
```

+ 4 גרפים של actual vs expected + טבלת peak time פר מצלמה.

**שונה מהראשית?** מהותית! המחברת הראשית מריצה **סולם מלא** של מודלים (persistence, seasonal-naive, profile, ridge, GRU) עם backtest מלא, table של השוואות ורעיונות ל-hybrid. התאומה שומרת רק את הצורה שיכולה לרוץ על VM (numpy pure) - זה מה שהקולקטור **יכול לעשות** בפועל אם ההוספה תופעל. הפרק הפך מ-lab notebook לפרוטוטיפ operational.

</div>

```python
# 11c-VM. The operational forecaster: profile x EWMA level + MAD band.
#
# Numpy only - this is the piece that could run ON the e2-micro next to
# the collector (milliseconds per camera, no torch, no sklearn). The local
# notebook's ladder is the evidence for WHY this simple shape is the right
# production choice: on short history the seasonal profile is brutally
# hard to beat, and it degrades gracefully when a stream disappears for a
# day (the band just widens - no retraining, no crash).
Z_FLAG, MIN_DELTA, EWMA_A = 3.5, 3.0, 0.25
HOLD_FRAC = 0.25

work = bins[bins.cam_id.isin(ELIGIBLE)].copy()
work['y'] = work.groupby('cam_id')['person'].transform(lambda s: s.ffill(limit=2))
t_split = work.ts.min() + (work.ts.max() - work.ts.min()) * (1 - HOLD_FRAC)
span_days_all = (work.ts.max() - work.ts.min()).total_seconds() / 86400
key = 'how' if span_days_all >= 7 else 'hod'
prof = work[work.ts <= t_split].groupby(['cam_id', key]).y.median()
res = work[work.ts <= t_split].copy()
res['exp0'] = [prof.get((c, s), np.nan) for c, s in zip(res.cam_id, res[key])]
mad = ((res.y - res.exp0).abs().groupby(res.cam_id).median() * 1.4826).clip(lower=1.0)
print(f'profile key: {key} ({"hour-of-week" if key == "how" else "hour-of-day"}; '
      f'{span_days_all:.1f} days of history)')

def expected_series(dfc, cam):
    e0 = np.array([prof.get((cam, s), np.nan) for s in dfc[key]])
    lvl, out = 1.0, []
    for a, e in zip(dfc.y, e0):
        out.append(e * lvl if np.isfinite(e) else np.nan)
        if np.isfinite(a) and np.isfinite(e) and e >= 1:
            lvl = (1 - EWMA_A) * lvl + EWMA_A * np.clip(a / e, 0.4, 2.5)
    return np.array(out)

# Mini-eval on the holdout tail: does the profile beat the two dumb
# baselines? (The full ladder lives in the local notebook.)
rows = []
for h in (4, 48):                                        # 1h and 12h ahead
    maes = {'persistence': [], 'seasnaive24': [], 'profile': []}
    for cam, dfc in work.groupby('cam_id'):
        dfc = dfc.sort_values('ts').reset_index(drop=True)
        tgt = dfc.y.shift(-h)
        tst = dfc.ts > t_split
        t_slot = dfc[key].shift(-h)
        pf = np.array([prof.get((cam, s), np.nan) for s in t_slot])
        for name, pred in (('persistence', dfc.y),
                           ('seasnaive24', dfc.y.shift(96 - h)),
                           ('profile', pd.Series(pf))):
            e = (pred[tst] - tgt[tst]).abs().dropna()
            if len(e):
                maes[name].append(e.mean())
    if any(maes.values()):
        rows.append({'h': f'{h * 15}min',
                     **{k: np.round(np.mean(v), 3)
                        for k, v in maes.items() if v}})
    else:
        print(f'{h * 15}min ahead: not yet scoreable '
              '(test window too short) - appears as the cache grows')
print(pd.DataFrame(rows).set_index('h'))

# Expected band + deviation anomalies for the freshest cameras. Purple =
# unusual QUIET - the case section 4's rolling-z cannot see by design.
show = [c for c in ELIGIBLE
        if work[(work.cam_id == c) & (work.ts > t_split)].n.sum() > 0][:4]
fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.6 * len(show)),
                         sharex=True, squeeze=False)
for ax, cam in zip(axes[:, 0], show):
    dfc = work[work.cam_id == cam].sort_values('ts')
    dfc = dfc[dfc.ts > work.ts.max() - pd.Timedelta(hours=36)]
    exp = expected_series(dfc, cam)
    band = Z_FLAG * mad[cam]
    ax.fill_between(dfc.ts, np.clip(exp - band, 0, None), exp + band, alpha=0.18)
    ax.plot(dfc.ts, exp, lw=1, ls='--', label='expected')
    ax.plot(dfc.ts, dfc.y, lw=1.2, label='actual')
    resid = dfc.y - exp
    flag = resid.abs() >= np.maximum(band, MIN_DELTA)
    ax.scatter(dfc.ts[flag & (resid > 0)], dfc.y[flag & (resid > 0)],
               color='red', zorder=5, s=18)
    ax.scatter(dfc.ts[flag & (resid < 0)], dfc.y[flag & (resid < 0)],
               color='purple', zorder=5, s=18, marker='v')
    ax.set_ylabel(cam, fontsize=8); ax.legend(fontsize=7, loc='upper left')
axes[0, 0].set_title('actual vs expected (last 36h) - red = spike, purple = unusual quiet')
plt.tight_layout(); plt.show()

# Next 12h, per camera: expected curve + peak hour - the line a future
# daily digest could quote ("tomorrow's expected peak at Taksim ~14:00").
future = pd.date_range(work.ts.max() + pd.Timedelta(minutes=15),
                       periods=48, freq='15min', tz='UTC')
from app.collector import cam_tzinfo as _tz
summary = []
for cam in show:
    loc = future.tz_convert(_tz(cam))
    slots = (loc.dayofweek * 96 + loc.hour * 4 + loc.minute // 15) \
        if key == 'how' else (loc.hour * 4 + loc.minute // 15)
    exp_f = np.array([prof.get((cam, s), np.nan) for s in slots])
    if np.isfinite(exp_f).any():
        i = int(np.nanargmax(exp_f))
        summary.append({'cam': cam, 'peak_local': f'{loc[i]:%H:%M}',
                        'peak_people': round(float(exp_f[i]), 1),
                        'now_expected': round(float(exp_f[0]), 1)})
print(pd.DataFrame(summary).set_index('cam'))
```

<a id="cell-45"></a>
<div dir="rtl">

## חלק 12 - איך הדשבורד עובד במצב twin

### תא 45 - markdown - הסבר על tabs + פורט + restart

**מה עושה:** תא ההסבר הסופי של המחברת התאומה, שמתאר איך הדשבורד עובד ב-`?mode=twin`. הדשבורד מוגש על ידי אותו שרת HTTP קטן (`app.dashboard_server`) כמו במחברת הראשית, אבל התאומה פותחת אותו ב-**twin mode** (`?mode=twin`). Twin mode הוא סביבת הסקירה / התיוג של המפעיל עבור נתוני הפרודקשן האמיתיים של ה-VM (הקולקטור על ה-VM של GCP כותב ל-Firestore, וה-ReID DB + review pool שלו הם מה שהדשבורד הזה מציג).

**Tabs (twin-mode):**

- **Analysis** - גריד 2x2 + KPIs + אנומליה + אירועים אופרטיביים, בדיוק כמו במחברת הראשית. מאחר שהתאומה משקפת את גריד ה-VM לטורקיה, האריחים כאן הם המצלמות של ה-VM עצמו.
- **Search** - חיפוש דמיון תמונות + browse לפי מחלקה/זמן על pools של review + live-samples של הקולקטור.
- **Reinforcement learning** *(רק ב-twin)* - שולחן העבודה של התיוג. כל verdict שאתה שומר עושה hot-reload ל-confidence של הקולקטור פר-מצלמה בתוך ~7 דקות וגם מזין את המייצא-של-תיוג לאימון (‏`tools/export_labels.py`). Review ממוקם מעל Learning-proof כדי שפאנל הפעולה יהיה למעלה וגרף התוצאה אחריו.

**לא נראה במצב twin:** Send Report From VM (‏digest פעמיים ביום + מייל לפי דרישה מחווטים במקום אחר, לא מהכפתור הזה), Live Analysis 🔬 לכל אריח (ה-VM לא מריץ את השכבות החיות האלו), Window analysis, ותפריטי class/time בסטריפ של model-view. כמו כן, אין טאב Snapshots - זו סביבת הסקירה, לא אוסף screenshots.

**פורט** - אותו auto-scan של פורט חופשי (‏8000-8020). אם 8000 עסוק המחברת הזאת בוחרת את הפורט החופשי הבא ומדפיסה את ה-URL.

**‏Restart** - טען מחדש את הדף (‏Ctrl+F5) אחרי שינוי כל קובץ `src/web/*.html` או `.js`.

**למה:** להבטיח שהמפעיל מבין ש-`?mode=twin` הוא לא סתם דגל cosmetic - הוא מפעיל טאב שלם (‏Reinforcement learning) שלא קיים במצב main, ומסתיר תכונות שלא רלוונטיות לסביבת הסקירה.

**שונה מהראשית?** התא **הזה** קיים רק בתאומה. במחברת הראשית פרק 12 לא קיים כלל - הדשבורד מוצג בפרק 7 עם `?mode=main` וזה סוף הסיפור. התאומה מוסיפה תא הסבר נפרד כדי לתעד את mode ה-twin, שהיא הגדולה שלה: להיות סביבת סקירה + תיוג אמיתית של הקולקטור.

</div>

```markdown
## How this dashboard works

The dashboard below is served by the same tiny local HTTP server
(`app.dashboard_server`) as the main notebook, but this notebook opens it
in **twin mode** (`?mode=twin`). Twin mode is the operator's review /
tagging environment for the VM's actual production data (the collector on
the GCP VM writes to Firestore, and its ReID DB + review pool are what
this dashboard shows).

**Tabs (twin-mode)**:

- **Analysis** - 2x2 grid + KPIs + anomaly + operational-events, exactly
  as in main. Since the twin mirrors the VM's Turkey grid, the tiles here
  are the VM's own cameras.
- **Search** - image-similarity + class/time browse over the collector's
  review + live-samples pools.
- **Reinforcement learning** *(twin only)* - the tagging workbench. Every
  verdict you save hot-reloads the collector's per-camera confidence
  within ~7 minutes AND feeds the training exporter
  (`tools/export_labels.py`). Review is placed above Learning-proof so
  the action panel is on top and the outcome chart follows.

**Not visible in twin mode**: Send Report From VM (twice-daily digest
+ on-demand email are wired elsewhere, not from this button), per-tile
Live Analysis 🔬 (the VM does not run these live layers), Window
analysis, and the class/time dropdowns in the model-view strip. Also, no
Snapshots tab - this is the review environment, not a screenshot
collector.

**Port** - same free-port auto-scan (8000-8020). If 8000 is busy this
notebook picks the next free one and prints the URL.

**Restart** - reload the page (Ctrl+F5) after changing any
`src/web/*.html` or `.js` file.
```


</content>
