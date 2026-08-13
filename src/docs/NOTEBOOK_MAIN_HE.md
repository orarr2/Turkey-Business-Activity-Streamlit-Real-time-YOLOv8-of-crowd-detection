<div dir="rtl">

# מדריך תא-אחר-תא - המחברת הראשית (turkey_business_activity.ipynb)

מסמך זה הוא מלווה מלא של המחברת הראשית של הפרויקט: `turkey_business_activity.ipynb`. הוא סוקר את **כל 50 התאים** של המחברת (0..49), אחד אחרי השני, בסדר הריצה, ולכל תא הוא עונה על ארבע שאלות: מה התא עושה, למה הוא קיים בכלל, אילו אלטרנטיבות היו על השולחן, ומה הפלט הצפוי כשאתה מריץ אותו.

המחברת הראשית היא **מחברת ההפניה החזקה** של הפרויקט: היא טוענת את `yolo26m.pt` (דור 2026, גודל medium) ב-`imgsz=960`, מריצה את כל שכבות הניתוח על 4 מצלמות שאתה בוחר בעצמך, ובמקביל מציגה את הדשבורד החי שאותו מזין ה-VM Collector שרץ 24/7 ב-GCP (`yolov8s@640` על `e2-micro` של 1GB RAM). כך אתה יכול להשוות דגימה מקומית שלך למה שהתרחש באותה מצלמה במשך 24 השעות האחרונות.

אם אתה קורא את זה בפעם הראשונה - המסלול המומלץ הוא ללחוץ **Run All** אחרי שהזנת 4 מספרי מצלמות בבורר (תא 10), ואז לחזור לתאים המעניינים כאן כדי להבין מה קרה. המסמך משלים את [`PROJECT_GUIDE_HE.md`](PROJECT_GUIDE_HE.md); שם המקום להבין את המערכת כמכלול, **כאן** הזום הוא על תא אחד בכל פעם.

</div>

---

<div dir="rtl">

## תוכן העניינים

### חלק א - הקדמה ותפקיד המחברת

1. [תא 0 - כותרת ומבוא](#cell-0)
2. [תא 1 - מציאות הרשת](#cell-1)
3. [תא 2 - סולם המדינות של הגריד](#cell-2)
4. [תא 3 - שני זמני ריצה: מחברת חזקה מול VM חלש](#cell-3)

### פרק 0 - התקנה

5. [תא 4 - כותרת פרק Setup](#cell-4)
6. [תא 5 - בדיקת תלויות והתקנת חבילות חסרות](#cell-5)
7. [תא 6 - טעינת המודל וקטלוג המצלמות](#cell-6)

### פרק 1 - בחירת מצלמה

8. [תא 7 - כותרת: קטלוג מצלמות + בורר](#cell-7)
9. [תא 8 - כותרת: קטלוג עם קישורים](#cell-8)
10. [תא 9 - קטלוג HTML אינטראקטיבי](#cell-9)
11. [תא 10 - בורר המצלמות (4 מספרים ממדינה אחת)](#cell-10)
12. [תא 11 - כותרת: המצלמות שנבחרו](#cell-11)
13. [תא 12 - checkpoint של הבחירה](#cell-12)
14. [תא 13 - כותרת: 1. Pick a camera](#cell-13)
15. [תא 14 - שליפת המצלמה הראשונה מהבחירה](#cell-14)

### פרק 2 - בדיקת פריים בודד

16. [תא 15 - כותרת: 2. Single-frame check](#cell-15)
17. [תא 16 - פריים בודד + זיהוי YOLO + פלוט](#cell-16)

### פרק 3 - סדרת פוטפול לאורך זמן

18. [תא 17 - כותרת: 3. Footfall time series](#cell-17)
19. [תא 18 - איסוף סדרת פוטפול חיה + שמירה ל-CSV](#cell-18)

### פרק 4 - אנומליות ופרופיל שעות שיא

20. [תא 19 - כותרת: 4. Anomalies + peak-hour](#cell-19)
21. [תא 20 - זיהוי אנומליות חסין (median/MAD) + פרופיל שעתי](#cell-20)

### פרק 5 - Dwell-time ועצירות ממושכות

22. [תא 21 - כותרת: 5. Dwell-time / prolonged stops](#cell-21)
23. [תא 22 - burst צפוף עם ByteTrack וחישוב dwell](#cell-22)
24. [תא 23 - סימון עצירות ממושכות + linger rate](#cell-23)

### פרק 5ב - Re-identification

25. [תא 24 - כותרת: 5b. Re-identification](#cell-24)
26. [תא 25 - אתחול מאגר ה-re-ID + טעינת המצלמה](#cell-25)
27. [תא 26 - לולאת דגימה + עדכון ה-registry](#cell-26)
28. [תא 27 - סיכום ישויות ייחודיות ומבקרים חוזרים](#cell-27)
29. [תא 28 - גרפים של re-ID לאורך הריצה](#cell-28)
30. [תא 29 - הערות איכות + מסלול פרודקשן OSNet](#cell-29)

### פרק 6 - ציון "האם להקים עסק כאן"

31. [תא 30 - כותרת: 6. Is it worth opening a business here](#cell-30)
32. [תא 31 - חישוב ציון עסקי משוקלל](#cell-31)

### פרק 7 - השוואה עם הדשבורד החי

33. [תא 32 - כותרת: 7. Compare with the live cloud dashboard](#cell-32)
34. [תא 33 - הפעלת הדשבורד המקומי עם המצלמות שבחרת](#cell-33)

### פרק 8 - השוואה בין מספר אתרים

35. [תא 34 - כותרת: 8. Compare multiple commercial sites](#cell-34)
36. [תא 35 - דירוג המצלמות שבחרת לפי פעילות](#cell-35)

### פרק 9 - סיכום חי של הריצה

37. [תא 36 - כותרת: 9. Live summary](#cell-36)
38. [תא 37 - איסוף כל הממצאים לסיכום אחד + פלוט](#cell-37)

### פרק 10 - Calibration של דיוק

39. [תא 38 - כותרת: 10. Accuracy calibration](#cell-38)
40. [תא 39 - 10a: לכידת פריימי calibration](#cell-39)
41. [תא 40 - 10b: תיוג אינטראקטיבי של פריימים](#cell-40)
42. [תא 41 - 10c: דוח MAE + bias](#cell-41)

### פרק 11 - חיזוי (Forecasting)

43. [תא 42 - כותרת: 11. Forecasting](#cell-42)
44. [תא 43 - 11a: משיכת היסטוריה מ-Firestore](#cell-43)
45. [תא 44 - 11b: Resample לרשת 15 דקות + חיתוך עידן](#cell-44)
46. [תא 45 - 11b-x: פיצ'רים חיצוניים (מזג אוויר + חגים)](#cell-45)
47. [תא 46 - 11c: baseline ladder + Ridge](#cell-46)
48. [תא 47 - 11c-b: Chronos-2 zero-shot](#cell-47)
49. [תא 48 - 11d: GRU forecaster](#cell-48)

### פרק סיום

50. [תא 49 - כיצד הדשבורד עובד](#cell-49)

</div>

---

<a id="cell-0"></a>
<div dir="rtl">

### תא 0 - [markdown] - כותרת ומבוא כללי למחברת

**מה עושה:** תא הפתיחה של המחברת. תא markdown שנועד לתת למי שפותח את הקובץ ב-Jupyter תמונה מלאה של המסלול מלמעלה למטה עוד לפני שהוא רץ שורת קוד אחת. הוא פותח בכותרת "Business Activity - Live Footfall" ומסביר שהמחברת מריצה end-to-end דגימת YOLO קצרה משלך על מצלמת רחוב ציבורית (טורקיה / תאילנד / יפן / ארה"ב) ומשווה את מה שנמצא לדשבורד הענני שאוסף היסטוריה של 24 שעות מה-VM ב-GCP.

**למה:** המחברת אינה תסריט ליניארי אחד; היא מפעילה 11 שכבות ניתוח נפרדות (בדיקת פריים, פוטפול, אנומליות, dwell, re-ID, ציון עסקי, השוואה לדשבורד, השוואת אתרים, סיכום, calibration, חיזוי). בלי מפה קצרה בפתיחה, קורא חדש היה נאלץ לגלגל ולנחש. התא הזה הוא מפת הדרכים: הוא מפרט מה כל פרק עושה בשורה, כך שהקורא יודע מראש לאיפה מכוונים.

**אלטרנטיבות:** אפשר היה להשאיר את פתיחת המחברת ריקה ולתת ל-README במאגר להסביר; במקום זה בחרו לשכפל את המפה גם כאן, כי המפעיל בפועל פותח קודם את הקובץ `.ipynb` ולא את `README.md`, וסטודנטים שקוראים את המחברת בסקירה (למשל בהגשה אקדמית) לא בהכרח מגיעים לתיקיית `src/docs`.

**פלטים:** אין פלטים תפעוליים; זה תא טקסט בלבד. כשמריצים אותו הוא פשוט מוצג ב-Jupyter כטקסט מעוצב.

</div>

```markdown
# Business Activity - Live Footfall

Run this end-to-end to analyze **your own** short YOLO sample from a public
street camera (Turkey / Thailand / Japan / USA) AND compare it live to the
**cloud dashboard's 24-hour history** (pushed continuously by the collector
running on a GCP e2-micro).

What you get, top-to-bottom:

- Setup: dependency check, then load the detector (this notebook = YOLO26-m).
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

### תא 1 - [markdown] - מציאות הרשת: למה להריץ את זה מקומית

**מה עושה:** תא markdown קצר שמסביר לקורא מדוע יש לזרם של איסטנבול (`kamerayayin.ibb.istanbul` והישן `livestream.ibb.gov.tr`) בעיה שאינה תלויה במחברת בכלל: השרתים חסומים ב-allowlist מסביבות sandbox מוגבלות (בהן גם הסביבה שבה הרעיונות של הפרויקט התגבשו). על-כן, אם רוצים לראות את זרם ה-IBB באמת עובד, צריך להריץ את המחברת ממחשב ביתי, מ-VM פתוח, או מאפליקציה שכבר deployed. סגמנטי 1080p כבדים נמשכים ב-`head-only` (בערך 2.5 MB ראשונים) כדי שקבלת פריים בודד תישאר מהירה גם על קו איטי.

**למה:** בלי אזהרה מפורשת הקורא היה מגיע לתאי הזיהוי, מקבל `frame is None`, ומאבד זמן במקום להבין שהחסימה היא ברמת הרשת. התא ממקם את הבעיה בציבור (מדינות, ספקים, geo-blocks) עוד לפני שהמחברת מתחילה לרוץ.

**אלטרנטיבות:** אפשר היה להסתפק ב-warning שרץ אוטומטית כאשר `resolve_stream()` מחזיר שגיאה, אבל warning בקצה השני של השרשרת מגלה למשתמש את הבעיה מאוחר מדי. תא markdown למעלה, לפני התלויות, קובע ציפייה נכונה מלכתחילה.

**פלטים:** טקסט מעוצב בלבד.

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

### תא 2 - [markdown] - סולם המדינות של הגריד

**מה עושה:** תא markdown ארוך שמסביר את `CountryDirector`, לוגיקת ניהול המדינות של הקולקטור. הוא לא נעול על מדינה קבועה: תמיד יש 4 מצלמות ממדינה **אחת**, ויש סולם עדיפויות שהוא טורקיה, אז תאילנד, אז יפן, אז ארה"ב. ההסבר מפרט את שלושת המנגנונים ברמת הבריאות שמונעים "gaming" של הסולם: בריאות פר מצלמה (3 החמצות ברצף = מנוחה של 15 דקות; מצלמת `tvkur` של קוניה נחה כבר אחרי החמצה בודדת), מנתק-זרם ברמת ה-host (403/429 חוזרים = מנוחה של 20 דקות ובקשת גישוש בודדת מחזירה את ה-host), מעבר מדינה רק כשהמדינה הפעילה לא מסוגלת להעמיד ולו מצלמה חיה אחת, והתאוששות מיוחדת לפני דוח מתוזמן (הקולקטור בודק מדינות בעדיפות גבוהה יותר כדי להחזיר את טורקיה ברגע שה-block משתחרר).

**למה:** הפרויקט הוא "פעילות מסחרית בטורקיה", אבל בפועל טורקיה מבודדת גיאוגרפית לרוב זמן ה-VM (IBB מחזיר 403 כשה-IP הוא של GCP). בלי `CountryDirector` הדשבורד היה נשאר עם 4 tiles ריקים כל היום. הסולם מבטיח שהקורא רואה משהו חי גם אם טורקיה חסומה, ושהדוח משתלב בתזמון של המדינה שבפועל מוצגת (Bangkok לא נכנס ללילה באותה שעה כמו איסטנבול).

**אלטרנטיבות:** מדינה קבועה = דשבורד ריק. סיבוב round-robin בין מדינות = הרבה מעברים חסרי סיבה. הסולם המדורג הוא פשרה: טורקיה קודמת כשהיא זמינה, החו"ל תופס רק כשחייבים.

**פלטים:** תיאור טקסטואלי בלבד; הלוגיקה עצמה מיושמת ב-`app/country_director.py`.

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

### תא 3 - [markdown] - שני זמני-ריצה: המחברת החזקה מול ה-VM החלש

**מה עושה:** תא markdown שמציג בטבלה את שני זמני-הריצה של אותו pipeline זיהוי: המחברת הזאת (מקומית, מטרה: exploration, calibration, הוכחת דיוק) מול VM Collector בענן (24/7 aggregation ל-Firestore). המחברת רצה `yolo26m.pt` ב-`imgsz=960` על החומרה של המשתמש (ואפילו על GPU אם יש); ה-VM רץ `yolov8s.pt` ב-`imgsz=640` על `e2-micro` של 1GB. התא מסביר בגלוי למה ה-VM חלש יותר: 1GB RAM ו-2 vCPU shared לא מסוגלים לרוץ YOLO26-m ב-960 בלי לפוצץ תקציב זיכרון. הוא גם מציג את מדידת הפער (2026-08-05): `yolov8n@512` (עידן ישן) מצא 0 אנשים בטקסים ו-0 רכבים בסראצ'אנה; `yolov8s@960` מצא 5 ו-7; ‏`YOLO26-m@960` מצא 6 ו-16 פלוס אוטובוס. המסקנה: המחברת היא Reference, ה-VM הוא estimator זול תמידי.

**למה:** המחברת קיימת כדי לענות על שאלה שאי אפשר לענות עליה עם ה-VM לבד: "עד כמה הספירות של ה-VM נכונות?". תא זה מסביר שהתאום שלה מקומי בלבד (`turkey_business_activity_yolov8n.ipynb`, ב-`.gitignore`) טוען את המשקולות של ה-VM כדי לראות בדיוק מה ה-VM היה רואה, כלי אמיתי להשוואת apples-to-apples.

**אלטרנטיבות:** אפשר היה לנסות להריץ את YOLO26-m ב-VM עם swap וזיכרון וירטואלי, אבל ה-CPU הוא 0.25 vCPU מובטח, וזמן ריצה של דקות לפריים היה שובר את פרק ה-40 שניות של הסבב. הפתרון הנכון הוא לתת לענן להיות "cheap always-on" ולמחברת המקומית להיות "accurate reference on demand".

**פלטים:** טקסט מעוצב + טבלת השוואה בפורמט markdown.

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
straight digests. What the small model still misses vs this notebook is the
accuracy gap this notebook exists to measure.

**How big is the gap?** Measured 2026-08-05 on identical live frames: the old
`yolov8n@512` found **0** people at Taksim and **0** vehicles at Sarachane;
`yolov8s@960` found 5 and 7; **YOLO26-m@960 found 6 people and 16 vehicles
plus a bus.** So: this notebook is the accurate reference; the VM is the
cheap, always-on estimator. The calibration section quantifies the gap.

> There are two notebooks. **This one** (`turkey_business_activity.ipynb`, on
> GitHub) is the YOLO26-m reference. A local-only twin
> (`turkey_business_activity_yolov8n.ipynb` - the filename is historical) is
> identical except it loads the VM's pinned weights (`yolov8s` @640 since
> 2026-08-05) - run it to see EXACTLY what the VM sees.
```

<a id="cell-4"></a>
<div dir="rtl">

### תא 4 - [markdown] - כותרת פרק 0: Setup

**מה עושה:** תא markdown של שורה אחת: `## 0. Setup`. תפקידו סימון פרק בלבד. הוא נמצא ב-TOC של Jupyter (הכותרת נראית כלינק בסייד-בר) וחוסך למי שקורא את המחברת דילוג ידני בין תאים כדי לזהות איפה מתחיל הפרק החדש.

**למה:** המחברת מחולקת לפרקים ממוספרים (0. Setup, 1. Pick a camera, ... 11. Forecasting). כותרת נפרדת מאפשרת לחשוב על המחברת כמסמך ולא כמערך תאים.

**פלטים:** רק שם הפרק בגופן כותרת.

</div>

```markdown
## 0. Setup
```

<a id="cell-5"></a>
<div dir="rtl">

### תא 5 - [code] - בדיקת תלויות והתקנת חבילות חסרות

**מה עושה:** התא הראשון שבאמת עושה משהו. הוא מגדיר רשימת `REQUIREMENTS` של זוגות `(import_name, pip_name)` לכל חבילה שהמחברת צריכה: `cv2` (opencv-python-headless), `numpy`, `pandas`, `matplotlib`, `PIL` (Pillow), `ultralytics`, `yt_dlp`, `firebase_admin`, `ipywidgets>=8` ו-`urllib3`. הוא רץ בלולאה: מנסה `importlib.import_module` על כל חבילה. אם ה-import מצליח, מדפיס `OK v<version>` באמצעות עוזרת `_version()` שקוראת ל-`__version__` או `VERSION` או מחזירה `unknown`. אם ה-import נכשל, מוסיף את החבילה ל-`missing` ואז רץ עוד לולאה שמריצה `pip install -q` דרך `subprocess.check_call([sys.executable, '-m', 'pip', 'install', ...])`. אחרי ההתקנה יש `importlib.invalidate_caches()` ואז import חוזר כדי לאמת שאכן הותקן. אם היו חבילות חסרות, מדפיס בסוף הודעה שאומרת שאם התא הבא נופל עם `ModuleNotFoundError`, יש לעשות Kernel Restart.

**למה:** במחברת שמופצת דרך גיטהאב אי אפשר להניח שהמשתמש הריץ `pip install -r requirements.txt` כמו שצריך. המשתמש הטיפוסי פותח את הקובץ ב-Jupyter וללחץ Run All. תא זה מבטיח שהריצה לא תתפוצץ בתא הראשון על `ModuleNotFoundError: No module named 'ultralytics'`. הוא גם מדפיס גרסאות מותקנות כדי שכשהקורא מדווח באג תהיה לו רשומה של הסביבה.

**אלטרנטיבות:** קובץ `requirements.txt` בלבד היה מחייב את המפעיל לרוץ ב-shell לפני. תא `!pip install ultralytics numpy ...` בלי בדיקה קודמת היה מתקין מחדש כל ריצה ומאיטה משמעותית. הגישה של "בדוק ואז התקן רק חסר" היא ה-sweet spot: ריצה שנייה עלולה להיות רק גיבוב גרסאות בלי עלות רשת.

**פלטים:** שורות מהצורה:
```
import               pip package              status
------------------------------------------------------------------
cv2                  opencv-python-headless   OK  v4.11.0
numpy                numpy                    OK  v1.26.4
...
ultralytics          ultralytics              OK  v8.3.0
```
אם משהו חסר גם: `MISSING -> installing` ואחר-כך `installed v...`.

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

### תא 6 - [code] - טעינת המודל וקטלוג המצלמות

**מה עושה:** תא ה-import הרשמי של המחברת. הוא מבצע ארבעה דברים בסדר: (1) מייבא stdlib בסיסי (`sys`, `time`, `datetime`, `defaultdict`, `Path`), (2) מייבא `cv2`, `numpy`, `pandas`, `matplotlib` שכבר בטוח מותקנים אחרי תא 5, (3) מזהה את מיקום `src/` כדי לאפשר `from app.detect_core import ...` בין אם המחברת נפתחה מ-root הפרויקט או מתוך `src/` (לוגיקה: אם `Path.cwd() / 'src' / 'app'` הוא dir אז `_src_dir = Path.cwd() / 'src'`, אחרת `_src_dir = Path.cwd()`), (4) קורא ל-`load_model(str(_src_dir / MODEL_WEIGHTS))` עם `MODEL_WEIGHTS = 'yolo26m.pt'`. יצירת `DATA_DIR = _src_dir / 'data'` וה-`mkdir(parents=True, exist_ok=True)` מבטיחה שיהיה יעד מקומי לפלטי CSV. הוא גם מדפיס את שם המודל, ‏`active_cameras()` (רשימת מזהי מצלמות פעילים) ואת `GRID_CAMERAS` (רשימת מזהי 4 המצלמות של הגריד הענני).

**למה:** ההפרדה בין `_src_dir` לבין `Path.cwd()` היא הגנה מפני "המחברת פועלת אצלי אבל נשברת אצלך": פרויקטי דאטה סיינס פותחים מחברות משתי תיקיות שונות תלוי איך המשתמש הריץ את `jupyter lab` (מ-root או מ-src). המנגנון הזה עובד לשני המצבים בלי שינוי. טעינת המודל בפעם הראשונה מורידה כ-42MB של משקולות YOLO26-m; בפעמים הבאות זה טעינה מהדיסק בלבד (חיסכון של דקה בקירוב).

**אלטרנטיבות:** אפשר היה לקבוע `sys.path.append('src')` באופן קשיח, אבל זה עובד רק ממיקום ריצה אחד. אפשר היה גם להשתמש ב-`installed package` באמצעות `pip install -e .`, אבל זו מכשלה נוספת עבור סטודנטים שמעדיפים להריץ ישירות. הפתרון המקומי במחברת פשוט וגמיש.

**פלטים:** שלוש שורות: `model: yolo26m.pt (VM runs yolov8s @640)`, ‏`cameras available: [...]` (מזהי המצלמות הפעילות), ‏`dashboard grid (4 live cameras): [...]` (מה-VM). אם המשקולות לא קיימות מקומית, YOLO יוריד אותן אוטומטית (הודעה של Ultralytics).

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

# --- Model: the local STRONG reference detector ---------------------------
# This notebook is the accuracy reference - it runs a large, modern model to
# get the best counts possible. The 24/7 cloud VM runs yolov8s @640 on a
# 1 GB e2-micro (see the "two runtimes" note near the top). YOLO26 is the 2026
# generation (highest mAP, NMS-free); the 'm' (medium) size is the accuracy
# sweet spot that still runs on a laptop CPU. First run downloads ~42 MB.
# The local-only twin notebook (turkey_business_activity_yolov8n.ipynb) sets
# this to 'yolov8s.pt' to mirror EXACTLY what the VM sees.
MODEL_WEIGHTS = 'yolo26m.pt'
DATA_DIR = _src_dir / 'data'; DATA_DIR.mkdir(parents=True, exist_ok=True)
model = load_model(str(_src_dir / MODEL_WEIGHTS))
print('model:', MODEL_WEIGHTS, '(VM runs yolov8s @640)')
print('cameras available:', list(active_cameras()))
print('dashboard grid (4 live cameras):', GRID_CAMERAS)
```

<a id="cell-7"></a>
<div dir="rtl">

### תא 7 - [markdown] - כותרת: קטלוג מצלמות + בורר

**מה עושה:** תא markdown שמכריז על שלב הבחירה ומסביר את מה שקורה בתא 10. הוא מפרט את החוזה של הבורר: הבחירה מוגבלת ל-4 מצלמות, כולן **מאותה מדינה** (בדיוק כמו הגריד הענני), ומצלמות שנבחרו מוצגות כמזהים אחרי כל הכנסה. בסוף התא מסביר שהבחירה נשמרת בזיכרון של ה-kernel: אחרי הזנה חד-פעמית של 4 מספרים, Run All יזרום ישר לסוף. כדי לשנות בחירה: Kernel > Restart Kernel.

**למה:** הבורר בתא 10 הוא לב לב המחברת: כל שאר התאים תלויים במשתני הגלובל שהוא כותב (`SELECTED_CAMS`, `SELECTED_COUNTRY`, `SELECTED_CAMS_APPLIED`). אזהרה מפורשת "כלל 4 מאותה מדינה" מונעת ניסיונות של בחירה מעורבת שהיו נכשלים באמצע ומחייבים restart.

**פלטים:** טקסט בלבד.

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

### תא 8 - [markdown] - כותרת: קטלוג עם קישורים

**מה עושה:** תא markdown קצר שמסביר שהתא הבא (9) מציג HTML של הקטלוג עם קישורים לכל דף מקור (webcamera24 / IBB / tvkur / YouTube). ההנחיה למשתמש היא לחיצה על שם מצלמה תפתח את המקור בטאב חדש כדי לוודא שהזרם באמת מתנגן **לפני** שמזינים את מספרו בבורר בתא 10.

**למה:** הרבה מהזרמים הציבוריים במחברת נוטים "למות" ללא הודעה: IBB חסום מ-GCP, tvkur מחזיר 404 כשהמצלמה מושהית ליום תחזוקה, YouTube-Live יורד שידור. וידוא ידני שהזרם מתנגן מקצר את הלולאה של "בחרתי, ריצתי, קיבלתי MISS, בחרתי מחדש".

**פלטים:** טקסט בלבד; ה-HTML האינטראקטיבי מיוצר בתא 9.

</div>

```markdown
###### Camera catalog with LINKS - auto-generated from `app/cameras.py`

Every camera below is grouped by country and links to its source page
(webcamera24 / IBB / tvkur / YouTube). Click any name to open the live
stream in a new tab and verify it plays BEFORE you type its number into
the picker cell below.
```

<a id="cell-9"></a>
<div dir="rtl">

### תא 9 - [code] - קטלוג HTML אינטראקטיבי

**מה עושה:** מייצר קופסת HTML גלילה עם קטלוג כל המצלמות, מקובץ לפי מדינה, כאשר כל שם מצלמה הוא קישור לדף המקור. הקוד מייבא `CAMERAS`, `COUNTRIES`, `COUNTRY_ORDER` ו-`country_pool` מ-`app.cameras` תחת שמות פרטיים (`_CAT`, `_CO`, `_OR`, `_cp`) כדי לא לזהם את הסביבה הגלובלית של המחברת. מבנה הרינדור: `<div>` עם `max-height:360px` ו-`overflow:auto` (קטלוג ארוך של 25+ מצלמות לא יבלע את החלון), כותרת `All available camera streams` מודגשת, אז לולאה על כל מדינה ב-`COUNTRY_ORDER` (`turkey, thailand, japan, usa`). לכל מדינה: `<h4>` עם דגל אמוג'י ושם תצוגה, אחריו רשימת `<ol>` שבה כל `<li>` מקבל את המספר הרץ (`value="{_n}"`) של הקטלוג, לינק `<a target="_blank" rel="noopener">` לשדה `page` או `url` של המצלמה, ואופציונלית שם עיר בסוגריים בצבע אפור. בסוף כל השורות מחוברות למחרוזת אחת ומוצגות עם `IPython.display.HTML`.

**למה:** הקטלוג נבנה מ-`app/cameras.py` **בזמן ריצה**, כלומר תמיד מסונכרן עם הקוד ולא ידרוש עדכון ידני של המסמך כשמצלמה נוספת. הבחירה בהוצאת ה-`_prefix` על כל השמות היא הרגל של הקוד: תאי מחברת רואים את כל הגלובלים של תאים קודמים, ומשתני `_prefix` מסמנים "פרטי, אל תשתמש מבחוץ".

**אלטרנטיבות:** אפשר היה להשתמש ב-`pandas.DataFrame` עם pandas styler והתצוגה של Jupyter, אבל התוצאה הייתה טבלה בלי לינקים. אפשר היה גם להוציא Markdown, אבל Markdown לא תומך בגובה גליל וקטלוג עם 25 מצלמות היה תופס חצי מסך.

**פלטים:** קופסת HTML גלילה של כל המצלמות של הפרויקט, כשכל שם הוא קישור. בסוף הקופסה שורה אפורה `N cameras total. The picker below uses these same numbers.`.

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

<a id="cell-10"></a>
<div dir="rtl">

### תא 10 - [code] - בורר המצלמות (4 מספרים ממדינה אחת)

**מה עושה:** הבורר עצמו. תא ארוך (5066 תווים) שאחראי על כל תהליך בחירת המצלמות. שלבי הריצה: (1) מדפיס לטרמינל טקסטואלי את כל הקטלוג עם מספרים רצים ובניית `_CAT_IDS[i]` שממפה מספר תצוגה למזהה מצלמה; (2) פונקצית העזר `_country_of(cid)` שולפת את שדה `country` של מצלמה נתונה; (3) לוגיקת "keep pick" ‏(`_KEEP`) בודקת אם ריצה קודמת של אותו kernel כבר החילה בחירה תקינה (‏`SELECTED_CAMS_APPLIED == True`, אורך רשימה = 4, כל המזהים באותה מדינה) - אם כן, מדפיס `APPLIED (kept): {country} -> {picks}` וחוזר בלי לשאול; (4) אם לא, מריץ לולאת `input()` על 4 מספרים: כל הזנה עוברת ולידציה של isdigit + טווח, איסור כפילות, ואיסור על ערבוב מדינות (המצלמה הראשונה שנבחרה קובעת את המדינה, כל השאר חייבות להיות ממנה). כשהלולאה מסתיימת מוצלחה - מציב `SELECTED_CAMS`, ‏`SELECTED_COUNTRY`, ‏`SELECTED_CAMS_APPLIED = True`; (5) live probe: פותח פריים אחד מכל אחת מה-4 שנבחרו (‏`grab_frame(resolve_stream(...))`) ומדפיס `LIVE` או `DEAD` פר מצלמה - כך שאם 2 מ-4 מתות, המשתמש רואה זאת ב-apply-time ולא אחרי 5 סקציות של MISS; (6) טיפול חריגות: `EOFError` / `KeyboardInterrupt` מבטל את הבחירה בעדינות, ‏`StdinNotImplementedError` (הריצה תחת nbconvert / papermill) מדפיס הודעה מדריכה במקום להתפוצץ.

**למה:** הבורר הוא הפצה המרכזית של המחברת. הוא חייב להיות (א) ידידותי - הזנת מספר, לא פייתון; (ב) עמיד - בחירה שגויה לא תפיל את הריצה, המשתמש יקבל הודעת שגיאה מפורשת ויוכל לנסות שוב; (ג) sticky - Run All לא ידרוש הזנה מחודשת כל ריצה. Live probe הוא לקח שלמד המחברת ב-18.07: משתמשים היו בוחרים 4 מצלמות של IBB מ-GCP, ריצים, ומקבלים "no frame" בכל תא, בלי להבין שהמצלמות עצמן לא מתות אלא ה-IP חסום.

**אלטרנטיבות:** אפשר היה `ipywidgets.Dropdown` במקום `input()`, אבל dropdown לא תמיד רנדר טוב בסביבות Jupyter מרוחקות (VS Code, Colab). ‏`input()` עובד בכל מקום. חלוקה ל-4 תאים נפרדים ("בחר מצלמה 1", "בחר מצלמה 2"...) הייתה שוברת Run All ומחייבת לחיצה 4 פעמים.

**פלטים:** בלוק טקסט של הקטלוג, ואז 4 שורות של הזנה (`Camera 1 of 4 (number 1-25): `), ואז `APPLIED: turkey -> [...]`, ואז probe:
```
probing your picks (one frame each, ~5-20s per dead cam)...
   LIVE  Sarachane Square
   DEAD  Taksim Square Istanbul
   LIVE  Konya Hukumet Meydani
   LIVE  Beyazit Square
```
אם יש DEAD פותח את השורה ב-`WARNING: N of 4 picks are not delivering frames`.

</div>

```python
## CAMERA PICKER - one big catalog, pick 4 cameras by NUMBER.
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

<a id="cell-11"></a>
<div dir="rtl">

### תא 11 - [markdown] - כותרת: המצלמות שנבחרו

**מה עושה:** תא markdown קצר שמסביר שהתא הבא הוא ה-checkpoint היחיד שכל המחברת שרויה בו: לפני Apply הוא עוצר בעדינות (התנהגות צפויה ב-Run All ראשון); אחרי Apply הוא רושם את הבחירה הסופית שתשמש עד סוף הריצה.

**למה:** הפרדה בין "פאוזה מכובדת" (הרים חריגה אמיתית של Python אבל עם הודעה מקבילה שאומר "זו לא באג, זה הבורר") לבין "בחירה סופית תקינה" הופכת את המחברת לזרימה אחת רצופה שלא מתפוצצת אם לוחצים Run All בטעות לפני שהזנת מספרים.

**פלטים:** טקסט בלבד.

</div>

```markdown
### Picked cameras

The single checkpoint the rest of the notebook depends on. Before you Apply
in the picker above it stops politely (expected on a fresh run); after
Apply it records the final selection the whole run will use.
```

<a id="cell-12"></a>
<div dir="rtl">

### תא 12 - [code] - checkpoint של הבחירה

**מה עושה:** תא קצר יחסית (1024 תווים) שמכריע האם אפשר להמשיך. אם `SELECTED_CAMS_APPLIED` לא קיים / לא True, הוא מגדיר class חריגה משלו `_ApplyFirst` שמעקף את traceback הרגיל של Python עם 3 שורות ידידותיות שאומרות "PAUSED: no cameras selected yet". החריגה מרימה את עצמה ועוצרת את Run All. אם הבחירה כן הוחלה, הוא מציג HTML ירוק מודגש עם `APPLIED. COUNTRY = {ctry} | SELECTED_CAMS = [...]` בראש, רשימת בולטים של 4 המצלמות עם שמותיהן, ושורת אישור מפורשת "The rest of the notebook will analyse THESE 4 cameras from {ctry}".

**למה:** ה-`_render_traceback_` המותאם הוא הטריק המרכזי כאן: Python מרים חריגה, אבל במקום ערמת stack traces מוצג בלוק טקסט אמפתי שמסביר בדיוק מה לעשות. זהו הבדל UX משמעותי - קורא שאיננו Python-fluent יודע מיד מה לתקן, במקום להיבהל מ-`Exception at line 42`.

**אלטרנטיבות:** אפשר היה `if not SELECTED_CAMS_APPLIED: return` בפונקציה כרוכה, אבל תא Jupyter לא מכיל פונקציית main. אפשר היה `sys.exit()`, אבל זה סוגר את ה-kernel כולו. הרמת חריגה בעלת פורמט תצוגה מותאם היא הפתרון הטבעי לתא Jupyter.

**פלטים:** בלוק HTML ירוק (כאשר יש בחירה) עם השורה `APPLIED. COUNTRY = turkey | SELECTED_CAMS = ['sarachane_ibb', 'taksim_ibb', 'konya_hukumet', 'beyazit_square']` ורשימת שמות המצלמות. או ‏`PAUSED: no cameras selected yet` אם אין בחירה.

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

<a id="cell-13"></a>
<div dir="rtl">

### תא 13 - [markdown] - כותרת: 1. Pick a camera

**מה עושה:** תא markdown ארוך יחסית (1183 תווים) שמכריז על תחילת הפרק הראשון של הניתוח הממשי. הוא מסביר איפה חי הקטלוג (`app/cameras.py`), איך מאורגן ב-`COUNTRIES` / `country_pool`, מפרט שהקולקטור מטפס בסולם `Turkey -> Thailand -> Japan -> USA`, ומחדד ש-`SELECTED_CAMS[0]` הוא המצלמה שהתא הבא יבדוק. חלק נכבד של התא מוקדש להסבר מה `resolve_stream(cam)` מבצע לפי `kind` של הקטלוג: `hls` נעשה שימוש ישיר (kamerayayin / tvkur), `youtube` נפתר עם yt-dlp דרך android client (תאילנד / יפן / ארה"ב מגובות YouTube-Live), `skyline` דורש scrape חי של playlist עם token, `webcamera24` פותר את הנגן המוטמע (tvkur או YouTube). התא מזכיר גם שיש hosts שנפתרים רק מרשת פתוחה (IBB חסום מחוץ לטורקיה; skyline / webcamera24 מסובבים tokens).

**למה:** הפרק הראשון של המחברת הוא הגשר בין הבחירה (המשתמש נתן 4 מזהים) לבין ה-URL הפתיר בפועל. `resolve_stream` הוא הסודות הגדולים של הפרויקט: הוא מסתיר 5 אסטרטגיות שונות של resolve מאחורי חתימה אחת. אזכור מפורש של הבעיות הגיאוגרפיות כאן חוסך למשתמש בלבול כשהוא מגלה בהמשך שהבחירה שלו לא מפיקה פריים.

**פלטים:** טקסט בלבד. הקוד עצמו נעשה בתא 14.

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

<a id="cell-14"></a>
<div dir="rtl">

### תא 14 - [code] - שליפת המצלמה הראשונה מהבחירה

**מה עושה:** תא קצר (609 תווים) שמבצע את הצעד הבנאלי אך הכרחי: תופס את `SELECTED_CAMS[0]` (המצלמה הראשונה מהבחירה), טוען את שדות המצלמה מ-`CAMERAS[CAM_ID]`, וקורא ל-`resolve_stream(cam)` שמחזיר URL של HLS. שני משתני הגלובל `CAM_ID` ו-`stream_url` שהתא כותב הם הבסיס לכל תא בהמשך שרץ על מצלמה בודדת (תאים 16, 18, 22, 25). לפני הפעולה יש שוב checkpoint של `SELECTED_CAMS_APPLIED` (חוזר על עצמו כי המחברת חייבת להיות עמידה בפני "לחצתי על תא 14 בלי לרוץ את 10 קודם"). בסוף הוא מדפיס `{name} -> {stream_url}` כדי שהמשתמש רואה במפורש איזה URL של HLS נבחר.

**למה:** מנגנון "המצלמה הראשונה" מפריד את הניתוח הבא לרמה של מצלמה בודדת (שכבות 2-6) לבין ניתוח 4-מצלמות (שכבה 8, calibration). קורא שרוצה להשוות בעצמו לא צריך להתעסק במזהים - הוא בוחר את הסדר בבורר של תא 10.

**אלטרנטיבות:** אפשר היה לקבוע `CAM_ID = 'sarachane_ibb'` קשיח (וגם ההיסטוריה של המחברת הייתה כזאת פעם), אבל אז הבחירה של המשתמש בבורר הייתה מתעלמת. שינוי זה בוצע ב-2026-07-18 (audit של אותה תקופה) כדי לגרום למחברת לכבד את הבחירה של המשתמש.

**פלטים:** שורה אחת בסגנון: `Sarachane Square -> https://kamerayayin.ibb.istanbul/kamera/sarachane/master.m3u8`.

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

<a id="cell-15"></a>
<div dir="rtl">

### תא 15 - [markdown] - כותרת: 2. Single-frame check

**מה עושה:** תא markdown קצר של שתי שורות. תפקידו לקבוע ציפייה: לפני שנתחיל לאסוף סדרת זמן שלמה, כדאי לוודא שהזרם באמת נפתח, שהפריים בכלל מגיע, ושהמודל רואה משהו הגיוני. זו בדיקה של שפיות (sanity check) ולא ניתוח.

**למה:** הרבה מהעבודה בפרויקטי vision היא לגלות שהזרם השתנה מעברית, החליף codec, או שהמסך שחור. אם משלים 30 שניות של דגימות ואז מגלה שכל הפריימים ריקים, בוזבז זמן. פריים אחד + הצגה מיד בהתחלה חוסכת את הבזבוז הזה.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 2. Single-frame check

Confirm the stream decodes and YOLO sees the crowd before collecting anything.
```

<a id="cell-16"></a>
<div dir="rtl">

### תא 16 - [code] - פריים בודד + זיהוי YOLO + פלוט

**מה עושה:** קורא ל-`grab_frame(stream_url)` כדי לתפוס פריים בודד מה-URL שנקבע בתא 14. יש 3 מסלולים אפשריים: (1) הפריים None (הזרם למטה או geo-blocked) - מודפסת אזהרה מפורטת ("WARN: X returned no frame") עם הצעה קונקרטית: Kernel Restart, בחר מצלמה אחרת (או מדינה אחרת כי YouTube עובד מכל מקום), Run All שוב; (2) הפריים תקין - מודפס `frame.shape` (מספר שורות, עמודות, ערוצים) ואז `detect_and_count(model, frame)` שמחזיר dict `{'person': N, 'vehicles': M}`; (3) חוזר ומריץ `model.predict(frame, conf=0.35, classes=[0,1,2,3,5,6,7], verbose=False)[0]` ומצייר עם matplotlib את התוצאה עם bounding boxes. הפונקציה `res.plot()` של Ultralytics מחזירה תמונה עם BBox מסומנים; `cv2.cvtColor` הופך BGR ל-RGB (Ultralytics/OpenCV שומרות ב-BGR, matplotlib מציג ב-RGB); הכותרת של הפלוט היא שם המצלמה.

**למה:** רשימת ה-classes `[0,1,2,3,5,6,7]` היא בכוונה: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck. אלה 6 המחלקות שרלוונטיות לפוטפול עירוני (מדלגים על class 4 = airplane וכל 73 המחלקות הבאות של COCO). `conf=0.35` הוא סף אמצע הדרך: 0.25 היה מייצר יותר false positives בסצנות רועשות; 0.5 היה מפספס אנשים רחוקים על הכיכרות.

**אלטרנטיבות:** אפשר היה לפתוח stream כ-VideoCapture ולקרוא פריים אחד, אבל `grab_frame` (מודול `app.detect_core`) מטפל ב-headers של tvkur / IBB / skyline (הם דורשים Referer / Origin) שאותם `cv2.VideoCapture(url)` אינו יכול להעביר ב-Windows. אפשר היה גם להריץ `model.predict` פעמיים (פעם ל-counts, פעם לוויזואל) - הקוד עושה זאת בכוונה כי `detect_and_count` נותן מספרים "מנוקים" (הוא מסנן על אותן 6 מחלקות ומאחד את הכל תחת `vehicles`), ו-`model.predict` השני עם `res.plot()` נותן ויזואל שלם כולל שמות המחלקות המקוריים.

**פלטים:** אם ה-frame קיים - שורות `frame shape: (720, 1280, 3)` ו-`counts: {'person': 6, 'vehicles': 12}`, ואז תמונה עם ריבועים סביב כל אדם ורכב שזוהה, כותרת שם המצלמה, ללא צירים. אם לא - הודעת WARN + הוראות שיקום.

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

<a id="cell-17"></a>
<div dir="rtl">

### תא 17 - [markdown] - כותרת: 3. Footfall time series

**מה עושה:** תא markdown קצר שמסביר את הפילוסופיה של הדגימה הדלילה: לשאלת "כמה / מתי" לא צריך כל פריים - דגימה אחת כל 15 עד 30 שניות מספיקה בהחלט וגם עדינה יותר עם השרת. זו בדיוק הלוגיקה שהקולקטור מריץ 24/7.

**למה:** דגימה צפופה (25 fps) של זרם 1080p תגרום ל-YOLO לרוץ עם load של 100% CPU כל הזמן, תשרוף מכסת bandwidth, ותייצר סדרה עם autocorrelation גבוהה שלא תוסיף מידע. דגימה של אחת ל-20 שניות היא sweet spot: מקבלים את המבנה של סדרת הזמן בלי לבזבז חישוב.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 3. Footfall time series (sparse sampling)

For the **how much / when** question we don't need every frame - one sample every 15-30s is plenty and
is gentle on the server. This is the same logic the collector runs continuously.
```

<a id="cell-18"></a>
<div dir="rtl">

### תא 18 - [code] - איסוף סדרת פוטפול חיה + שמירה ל-CSV

**מה עושה:** מגדיר פונקציה `footfall_series(stream_url, cam_name, interval_s=20, duration_min=1.0)` שרצה בלולאת while עד `time.time() > t_end` (‏`t_end = now + duration_min * 60`). בכל איטרציה: תופס timestamp UTC, קורא ל-`grab_frame`, אם הצליח - קורא ל-`detect_and_count` ומקבל `{'person': N, 'vehicles': M}`; אם לא - שם NaN לשני המדדים. מוסיף שורה לרשימה עם `{ts, cam, person, vehicles}` ומדפיס שורת התקדמות (`[HH:MM:SS] person=X vehicles=Y`). אחרי הלולאה, מחזיר `pd.DataFrame(rows)`. הפעלה בפועל: `df = footfall_series(stream_url, cam['name'], interval_s=10, duration_min=1.0)` - כלומר מדגמן פר 10 שניות למשך דקה (6 דגימות בערך). לבסוף שומר את התוצאה ל-CSV בשם `data/footfall_{CAM_ID}.csv` ומציג `df.head()`.

**למה:** ההפרדה בין הפונקציה (שאפשר להזמין עם פרמטרים אחרים בהמשך המחברת - בתא 35 מזמינים אותה עם `duration_min=0.5`) לבין ההפעלה נותנת גמישות: המשתמש יכול להריץ מחדש רק את שורת ההפעלה עם `duration_min=5.0` בלי להעתיק את כל הפונקציה. הסימון `interval_s=10, duration_min=1.0` הוא דמו קצר; היוזר יכול להעלות לימים שלמים אם מריץ את הקולקטור עצמו.

**אלטרנטיבות:** אפשר היה להשתמש ב-`ThreadPoolExecutor` להריץ כמה מצלמות במקביל, אבל תא זה מריץ מצלמה אחת בכוונה (השאלה בשלב 3 היא "מה קורה בכיכר Sarachane בדקה הבאה", לא "מה קורה ב-4 מצלמות במקביל" - זו שאלה של שלב 8). ה-`sleep(interval_s)` הפשוט הוא בכוונה גם לא-drift-corrected: אם `detect_and_count` לוקח 2 שניות, המרווח האפקטיבי יהיה 12 שניות ולא 10. עבור דגימה של דקה בלבד זה זניח; עבור ריצה של יום שלם הקולקטור עושה חישוב מדויק יותר.

**פלטים:** 6 שורות של `[HH:MM:SS] person=X vehicles=Y` בזמן ההרצה, אז DataFrame עם 4 עמודות (`ts`, `cam`, `person`, `vehicles`), אז כתיבת CSV ל-`data/footfall_sarachane_ibb.csv` (או שם דומה), ואז 5 השורות הראשונות של ה-DataFrame בטבלת Jupyter.

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

<a id="cell-19"></a>
<div dir="rtl">

### תא 19 - [markdown] - כותרת: 4. Anomalies + peak-hour profile

**מה עושה:** תא markdown קצר שמסביר את שכבת שכבת הניתוח הבאה. אנומליה מוגדרת כ-rolling z-score > 3.5 על סדרת הפוטפול. שני סוגי אנומליות מוזכרים: זינוק פתאומי (אירוע / קידום / מחאה) או ירידה חריגה (סגירה / מזג אוויר). בנוסף מוצג פרופיל שעות שיא שאומר **מתי** חלון הפעילות המסחרית.

**למה:** בשלב 3 יש סדרה של אנשים ורכבים לאורך זמן. השאלה הבאה היא "מתי חורג משהו?". סף של 3.5 נבחר בכוונה מעל 3 - סף רגיל של 3 היה מייצר יותר מדי false positives לאור השונות המובנית של דגימות YOLO. שכבת peak-hour נותנת תמונה של קצב יום טיפוסי, מה שרלוונטי לשאלת "האם כדאי לפתוח כאן עסק ב-15:00 או ב-19:00".

**פלטים:** טקסט בלבד.

</div>

```markdown
## 4. Anomalies + peak-hour profile

**Anomaly = rolling z-score > 3.5** on the footfall series: a sudden surge (event/promotion/protest) or an
unusual drop (closure/weather). Peak-hour profile tells you *when* the commercial window is.
```

<a id="cell-20"></a>
<div dir="rtl">

### תא 20 - [code] - זיהוי אנומליות חסין + פרופיל שעתי

**מה עושה:** מגדיר `flag_anomalies(s, window=12, z=3.5, min_delta=3)` שמחשב z-score חסין: `med = s.rolling(window, min_periods=4).median()`, `mad = (s - med).abs().rolling(window, min_periods=4).median() * 1.4826`, `spread = mad.clip(lower=1.0)` (הרצפה של 1.0 בגלל שספירות הן שלמות ואי אפשר שסטיה תהיה 0), ואז `robust_z = (s - med) / spread`. הפונקציה מחזירה מסכה בוליאנית לפי שני תנאים: `|robust_z| > z` **וגם** `|s - med| >= min_delta` (התנאי השני מונע פליטת אזעקה על שינוי קטן במונחים אבסולוטיים). התא מריץ את הפונקציה על עמודת `person` של ה-`df` מתא 18, מוסיף עמודת `anomaly` בוליאנית, ומצייר 2 פאנלים: (1) גרף אנשים לאורך זמן עם נקודות אדומות על האנומליות (`scatter zorder=5`); (2) בר-פלוט של ממוצע אנשים לפי שעת יום ‏(`df.groupby('hour')['person'].mean()`).

**למה:** הבחירה ב-median/MAD (Median Absolute Deviation) על-פני mean/std היא חובה כאן. סדרות פוטפול הן heavy-tailed: אירוע גדול יכול להעיף את הממוצע גבוה, ואז השונות תעלה, ואז אירוע חדש לא ייחשב חריג כי הבסיס כבר "יודע" עליו. ‏Median לא מושפע מ-outliers; MAD (עם המקדם 1.4826 שממיר MAD לסטיית תקן שקולה תחת התפלגות נורמלית) נותן פיזור חסין. הבחירה ב-`window=12` היא לאחור על 12 דגימות (במרווח של 20 שניות = 4 דקות של היסטוריה), מספיק לזהות זינוק אבל לא ארוך מדי כדי לפספס אירוע חולף.

**אלטרנטיבות:** `Isolation Forest` היה יותר מתוחכם אבל דורש הרבה יותר דגימות (100+); על דקה של דגימות זה overkill. `EWMA` (exponentially weighted moving average) היה יותר "smooth" אבל פחות רגיש למעברים חדים - אותה בעיה כמו mean.

**פלטים:** לפעמים תקבל טבלה של 6 שורות עם עמודות `ts, cam, person, vehicles, anomaly`. שני פאנלים matplotlib בגודל 15x4: משמאל סדרת "Footfall over time (robust z)" עם קו כחול ונקודות עם `x` אדומות למקומות של anomaly; מימין בר-פלוט "Avg people by hour (peak-hour profile)".

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

<a id="cell-21"></a>
<div dir="rtl">

### תא 21 - [markdown] - כותרת: 5. Dwell-time / prolonged stops (tracking)

**מה עושה:** תא markdown באורך בינוני שמסביר למה השכבה הבאה דורשת שיטת דגימה שונה מהקודמות. השאלה "כמה זמן אדם או רכב נשאר לפני המצלמה?" דורשת **object tracking** (זהויות יציבות בין פריימים), ותנאי לעבודת tracking הוא פריימים **רצופים**. ההשלכה: כאן לא אפשר לדגום אחת ל-20 שניות. תא הקוד יעבוד עם `dense burst` (כמה fps במשך 60 שניות בערך) במקום דגימה דלילה. `model.track()` של Ultralytics (ByteTrack) מייחס לכל אובייקט id, ואז אפשר לאסוף כמה פריימים כל id נראה וכמה הוא זז. ההסבר מסיים בשני מפתחות תפעוליים: dwell גבוה + תנועה נמוכה = lingering (window-shopping / תור / רכב חונה); שיעור גבוה של lingering people הוא סיגנל חזק של איכות מסחרית (אנשים עוצרים, לא רק עוברים).

**למה:** הבחנה חשובה לפרויקט "פעילות מסחרית": לא כל אדם שעובר הוא לקוח פוטנציאלי. איש שרץ בכיוון האוטובוס לא מייצר עסקה. איש שעוצר לפני חלון ראווה למשך דקה כן. Dwell-time הוא ה-proxy המספרי הפשוט ביותר לתובנה הזאת.

**פלטים:** טקסט בלבד.

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

<a id="cell-22"></a>
<div dir="rtl">

### תא 22 - [code] - burst צפוף עם ByteTrack וחישוב dwell

**מה עושה:** מייבא `iter_frames` ו-`NAME_BY_ID` מ-`app.detect_core`, ומגדיר `dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.35)`. אלגוריתם: (1) מכין `defaultdict(int) frames_seen` ו-`defaultdict(list) centroids` ו-`dict track_cls`. (2) `n_frames = int(seconds * target_fps)` - למשל 30 שניות * 3 fps = 90 פריימים. (3) `stride = max(1, round(25 / target_fps))` - הזרמים המקוריים פועלים ב-25 fps בערך, לכן אם רוצים 3 fps אפקטיביים צריך לדלג על 8 פריימים בכל אחד; קריאה רצופה של 90 פריימים בלי stride הייתה דוחסת את החלון ל-3.6 שניות ומעלה dwell בפקטור 8. (4) לולאה על `iter_frames(stream_url, max_frames=n_frames, stride=stride)` - הפונקציה הזאת מטפלת ב-hosts עם headers חובה (tvkur, IBB, skylinewebcams) על-ידי הורדת הסגמנטים העדכניים ביותר עם ה-Referer/Origin המתאימים ופענוח מקומי, כי `cv2.VideoCapture(url)` ב-Windows אינו יכול להעביר headers. לכל פריים: `model.track(frame, persist=True, conf=conf, classes=[0,1,2,3,5,6,7], tracker='bytetrack.yaml', verbose=False)[0]`. אם יש `r.boxes.id` (יש טרקים אקטיביים), עוברים על כל תיבה: מגדילים `frames_seen[tid]`, מוסיפים centroid `(box[0], box[1])`, שומרים את class ID. (5) אחרי הלולאה, בונים DataFrame: לכל track, `dwell_s = round(n / target_fps, 1)` (מספר פריימים חלקי fps), `movement_px = np.linalg.norm(pts.max(0) - pts.min(0))` (המרחק בין הפינה הצפונית-מזרחית לדרומית-מערבית של תיבת ה-centroids). (6) `dwell = dwell_analysis(stream_url, seconds=30, target_fps=3, conf=0.25)` - מריץ בפועל.

**למה:** `stride` הוא הפרט הטכני המרכזי כאן. Iterating רגיל היה מייצר קבוצה של פריימים שנתפסו בשניות רצופות ומעלה dwell שקרי. `NAME_BY_ID` (בניגוד לרשימת המחלקות ‏[0,1,2,3,5,6,7] שהזכרנו מקודם) שומר מיפוי מ-class id ל-string (person, car, truck ...). `persist=True` מסנכרן את מאגר הטרקים על פני קריאות עוקבות של `model.track` - בלעדיו כל פריים היה מקבל id חדש ולא הייתה כל עקיבה.

**אלטרנטיבות:** DeepSORT או `StrongSORT` היו דורשים מודל embedding נוסף (עלות זיכרון). ByteTrack הוא tracker קל-משקל שעובד יופי על scenes אורבניים. אפשר היה גם `target_fps=1` כדי לחסוך CPU, אבל אז חלון של 30 שניות = 30 פריימים בלבד, ותנועה של אדם שהולך תראה כמו טלפורטציה.

**פלטים:** DataFrame `dwell` עם עמודות `track_id, class, dwell_s, movement_px`, ממוין לפי `dwell_s` יורד; `dwell.head(15)` מציג את 15 הטרקים הארוכים ביותר, למשל `[track_id=42, class=car, dwell_s=28.7, movement_px=12.4]`.

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

<a id="cell-23"></a>
<div dir="rtl">

### תא 23 - [code] - סימון עצירות ממושכות + linger rate

**מה עושה:** מגדיר 3 קבועים: `PERSON_DWELL_S=25`, `VEHICLE_DWELL_S=40`, `MAX_MOVE_PX=60`. סף גבוה יותר לרכב כי רכב חונה טבעית לזמן ארוך יותר מאדם. אם ה-DataFrame `dwell` לא ריק, מסנן שני תנאים במסך אחד: person עם `dwell_s >= 25` **או** non-person עם `dwell_s >= 40`, וגם `movement_px <= 60`. מדפיס `Prolonged stops detected: N` ומציג את הטבלה. אחר-כך מחשב `linger_rate = (person with dwell >= 25) / (total persons)` - השיעור של אנשים ש"התעכבו". מציג באחוזים.

**למה:** `linger_rate` הוא המדד העסקי המרכזי של השכבה. שיעור של 10% לינגר במקום קניות דרישה = כיכר עסקית שאנשים באמת "משתמשים" בה. שיעור של 60% לינגר = אזור המתנה (תחנה, סטרייק, מחאה). מספר קונקרטי שהופך דגימה של 30 שניות למדד יחיד שאפשר להשוות בין מצלמות.

**אלטרנטיבות:** אפשר היה להגדיר סף אחיד לכולם (30 שניות), אבל זה היה מייצר false positives על אנשים שהאט לרגע וגם false negatives על רכבים שעצרו לזמן קצר. הפרמטרים המובחנים מדויקים יותר.

**פלטים:** אם יש dwells: `Prolonged stops detected: 3` ואז DataFrame של 3 שורות (טרקים ארוכים), ואז `Linger rate (people who stayed >= 25s): 12%` (למשל). אם אין - הבלוק פשוט מדלג בשקט.

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

<a id="cell-24"></a>
<div dir="rtl">

### תא 24 - [markdown] - כותרת: 5b. Re-identification

**מה עושה:** תא markdown ארוך (1335 תווים) שמסביר למה נדרשת שכבת Re-identification מעבר לספירת פרסונים. הספירות של YOLO אומרות **כמה** אנשים גלויים ברגע מסוים, אבל הן סופרות פעמיים כל מי שמתעכב לפני המצלמה. כדי לענות "כמה לקוחות ייחודיים עברו היום?" או "האם זה אותו וואן משלוחים שראיתי אתמול?", צריך זהות מתמדת שנצמדת לאדם/רכב וחוצה פריימים, bursts, ואפילו ימים. ההסבר מפרט את המימוש ב-`app/reid.py`: לכל זיהוי YOLO קוצצים את ה-bounding box; בונים histogram מסכה של HSV (8x8x8 bins, פיקסלים עם V<30 מתעלמים כדי להרוג את הגוון הצהוב-נתריום של הלילה בקוניה); מוסיפים aspect ratio ושטח מנורמל; L2-normalize -> וקטור 514-ממדי. משווים לכל ישות שכבר במסד `data/reid.db` באותה מחלקה באמצעות cosine similarity: אם ה-best match מעל threshold (ברירת מחדל 0.92) - מעדכנים sightings + last_seen; אחרת רושמים ישות חדשה. התא מזהיר בגלוי שזו חתימה demo-grade: עובד יפה בשעות היום (בגדים שונים = היסטוגרמות שונות בברור), מייצר match שקריים בלילה כשהסצנה כולה צהובה - המסלול של פרודקשן הוא להחליף את `embed_crop()` ב-forward של OSNet/torchreid; ה-registry ב-SQLite מסביב נשאר.

**למה:** הפרויקט לא רק מעניין ב-"כמה נמצאים כרגע" אלא ב-"כמה שונים במהלך שעה" ו-"מי חוזר". חוסר יכולת להבחין בין "5 אנשים חדשים" ל-"אותם 3 אנשים שרואים 5 פעמים" הוא חוסר מהותי לניתוח מסחרי.

**פלטים:** טקסט בלבד.

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

<a id="cell-25"></a>
<div dir="rtl">

### תא 25 - [code] - אתחול מאגר ה-re-ID + טעינת המצלמה

**מה עושה:** התא מתחיל שוב עם checkpoint של `SELECTED_CAMS_APPLIED`. אחר-כך מייבא `load_model`, `grab_frame`, `detect_with_boxes`, `annotate` מ-`detect_core` ו-`ReidStore` מ-`app.reid`. מגדיר `REID_DB = _src_dir / 'data' / 'reid_notebook.db'` (כותב לתת-תיקייה `data`). לוגיקת close-then-delete: ניסיון לקרוא ל-`reid.close()` בבלוק try/except NameError - בפעם הראשונה `reid` לא קיים ולכן נופל בשקט; בריצה חוזרת של אותו kernel, ה-`ReidStore` הקודם עדיין מחזיק חיבור SQLite ל-DB, וב-Windows אי אפשר למחוק קובץ פתוח (`PermissionError [WinError 32]`). אחר-כך מנסה למחוק את קובץ ה-DB ‏(`Path(REID_DB).unlink(missing_ok=True)`) לצורך demo נקי; אם משהו עדיין מחזיק אותו (kernel יתום, antivirus), מדפיס הודעה שהריצה תמשיך עם רשומות קיימות. יוצר `reid = ReidStore(REID_DB, threshold=0.92)`. אז שולף `CAM_ID = SELECTED_CAMS[0]`, `cam = CAMERAS[CAM_ID]`, ‏`stream_url = resolve_stream(cam)`, ומדפיס `feeding re-ID from {cam name}`.

**למה:** ה-close-then-delete הוא אחד מהמעטים "פתרונות ל-Windows" שקיימים במחברת. במקומות רבים אחרים המחברת מגיעה קרוב לבעיית לוקינג דומה, אבל בגלל שה-SQLite חשוף ישירות ב-`reid.db`, יש כאן פוטנציאל קונקרטי לתקלות. הפתרון הוא נקי: או מנקים ל-demo, או ממשיכים עם רשומות קיימות. `threshold=0.92` הוא סף שמרני מאוד (cosine similarity חייבת להיות מעל 92%) שנבחר לפי מסלול טוב-ביום-רע-בלילה כפי שהוזכר בתא 24.

**אלטרנטיבות:** אפשר היה לפתוח קובץ נפרד לכל ריצה (timestamped), אבל אז אחרי כמה ריצות ה-`data/` הופך למשוגע. השימוש בקובץ יחיד עם ניקוי אופציונלי הוא פשרה הגיונית.

**פלטים:** אם אין lock: `reid_notebook.db cleared - fresh demo registry`. אחרת: הודעה על משתמש אחר שמחזיק. ואז שורה: `feeding re-ID from Sarachane Square`.

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

<a id="cell-26"></a>
<div dir="rtl">

### תא 26 - [code] - לולאת דגימה + עדכון ה-registry

**מה עושה:** מגדיר 3 קבועים: `N_SAMPLES = 8`, `INTERVAL_S = 5`, `CONF = 0.25`. אז לולאה של 8 איטרציות: ‏(1) `grab_frame(stream_url)` - אם None, מדפיס `[XX] miss` ומחכה `INTERVAL_S` שניות; (2) `counts, boxes = detect_with_boxes(model, f, conf=CONF)` - מקבל dict של ספירות ורשימה של תיבות; (3) `results = reid.update_from_frame(CAM_ID, f, boxes)` - מזין את כל התיבות ל-registry, מקבל בחזרה רשימת `ReidResult` עם `is_new` פר תיבה; (4) `new = sum(r.is_new for r in results)`, ‏`seen_again = len(results) - new`; (5) מוסיף שורה לרשימה `rows` עם sample index, people, vehicles, total detections, new_ids, seen_again; (6) מדפיס שורת התקדמות; (7) sleeps. אחרי הלולאה: `reid_df = pd.DataFrame(rows)` והצגה.

**למה:** `conf=0.25` (יותר נמוך מברירת המחדל של `detect_and_count` שהיא 0.35) נבחר מודע לזה שקוניה היא צילום זווית רחבה ואנשים רחוקים קטנים בפריים; סף מחמיר היה מפספס אותם. `INTERVAL_S=5` צפוף יותר מ-20 של תא 18 - כי re-ID דורש חתימות של אותם אנשים שנצפו לאורך זמן, ומדגמים דלילים מדי מפספסים אנשים שעברו במהירות.

**אלטרנטיבות:** אפשר היה `N_SAMPLES=100` להרים סטטיסטיקה טובה יותר, אבל זה יגרום למחברת לרוץ 8+ דקות רק על השלב הזה - טוב לקולקטור, מייגע ל-demo. 8 דגימות של 5 שניות = 40 שניות של השלב, גבול סביר.

**פלטים:** 8 שורות `[XX] person=N vehicles=M -> new=X seen_again=Y`. ואז טבלה של 8 שורות עם 6 עמודות: sample, person, vehicles, detections, new_ids, seen_again.

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

<a id="cell-27"></a>
<div dir="rtl">

### תא 27 - [code] - סיכום ישויות ייחודיות ומבקרים חוזרים

**מה עושה:** קורא ל-`reid.stats(CAM_ID)` שמחזיר dict עם: `total_unique` (סך הכל ישויות ייחודיות שנרשמו במצלמה זו), `total_sightings` (סך הכל פעמים שראינו אותן), ו-`per_class` (חלוקה לפי `person`, `car`, `bus` וכולי, שבה לכל class יש `unique`, ‏`total_sightings`, ו-`regulars` - כמה מהישויות ראינו לפחות 3 פעמים). מדפיס את שלושתם. אחר-כך `reid.top_regulars(CAM_ID, n=10)` מחזיר עד 10 ישויות עם ה-`sightings` הגבוה ביותר, ומודפסות כל אחת עם entity_id, class, sightings, first_seen, last_seen.

**למה:** זו הצגה קונקרטית של המדד המסחרי החשוב ביותר של השכבה: "כמה מבקרים חוזרים היו". סף של 3 sightings נקבע ב-`ReidStore` כברירת מחדל של "regular" - נחשב לחוזר רק מי שראינו אותו 3 פעמים או יותר, כדי לא לסמן כ-regular גם אחד שרק "התעכב" באותה דגימה.

**אלטרנטיבות:** אפשר היה להציג את זה כגרף (bar chart), אבל טבלה של 10 שורות עם timestamps היא מיידית לקריאה ומאפשרת למפעיל לרוץ ל-DB עם ה-entity_id ולראות את ה-metadata המלא של הישות.

**פלטים:**
```
Total unique entities (this camera): 42
Total sightings: 68
  person      unique=35  sightings=54  regulars(>=3)=4
  car         unique=5   sightings=8   regulars(>=3)=1
  motorcycle  unique=2   sightings=6   regulars(>=3)=1
Top returning entities:
  #  17  person    sightings=6  first=...  last=...
  #  23  car       sightings=4  first=...  last=...
  ...
```

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

<a id="cell-28"></a>
<div dir="rtl">

### תא 28 - [code] - גרפים של re-ID לאורך הריצה

**מה עושה:** מייצר 2 פאנלים אם יש לפחות 3 דגימות ב-`reid_df`. מוסיף ‏עמודה `returning_rate = seen_again / detections` (עם `replace(0, np.nan)` כדי להימנע מחלוקה באפס). פאנל 1: שני קווים על אותו ציר: `new_ids` (עיגולים) ו-`seen_again` (ריבועים) לפי sample #. פאנל 2: `returning_rate` לפי sample #, ציר y מ-0 ל-1 (כי זה שיעור), צבע ירוק. אם פחות מ-3 דגימות, מדפיס `Not enough samples for the returning-visitor plot.`.

**למה:** ה-`returning_rate` הוא ה-metric השני החשוב אחרי `regulars` מתא 27, והוא מבטא באופן דינמי (סטריפ של 8 דגימות) כמה מהמזוהים בכל רגע היו שם קודם. עלייה של ה-rate עם הזמן = הצטברות של regulars; שטיחות שלה סביב 0 = תנועה חדשה כל הזמן. הפאנל השמאלי מציג את המספרים הגולמיים; הימני מציג את היחס.

**אלטרנטיבות:** אפשר היה `stacked bar` של `new_ids` + `seen_again`, אבל גרף קו נותן מגמה ברורה יותר. אפשר היה `plotly` אינטראקטיבי, אבל matplotlib סטטי דיו לצורך המחברת.

**פלטים:** אם >= 3 דגימות: שני פאנלים matplotlib (13x4 סה"כ). אחרת: הודעת דילוג.

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

<a id="cell-29"></a>
<div dir="rtl">

### תא 29 - [markdown] - הערות איכות + מסלול פרודקשן OSNet

**מה עושה:** תא markdown (למרות שהוא כתוב כטקסט חופשי) שנועד לחשוף למשתמש מפורש שהאיכות של re-ID תלויה בסצנה. הוא מזהיר: בכיכר Konya Hukumet Meydani בלילה כל הסצנה בגוון נתריום צהוב אחיד, ואז re-ID מבוסס histogram יעשה over-merge של IDs. פתרון מיידי לוולידציה: להריץ על bazaars בשעות היום (בגדים בצבעים שונים), או להעלות threshold ל-0.97. פתרון פרודקשן: `pip install torchreid`, אז `from torchreid.utils import FeatureExtractor`, שימוש ב-`osnet_ain_x1_0` על CPU (או GPU). כמה שורות של `def embed_crop(crop, cls): return extractor([crop])[0].cpu().numpy()` מחליפים את ה-histogram; שאר `app/reid.py` נשאר בדיוק כפי שהוא. Embedding של OSNet ב-2048 ממדים שורד שינויי תאורה, שינויי פוזה, occlusions באופן טוב יותר בהרבה.

**למה:** הפרויקט הוא demo-grade בכוונה: מודל histogram עובד מהקופסא בלי הורדות נוספות ב-CPU-only. אבל חייבים לוודא שהמשתמש יודע איך לשדרג לפרודקשן. הכתיבה כאן היא "מה לעשות מחר בבוקר" - שלוש שורות pip + החלפת פונקציה אחת, וכל ההשקעה ב-registry, ה-schema, וההגיון של `sightings/regulars` נשמר.

**פלטים:** טקסט בלבד; לא מריץ שום דבר.

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

<a id="cell-30"></a>
<div dir="rtl">

### תא 30 - [markdown] - כותרת: 6. Is it worth opening a business here

**מה עושה:** תא markdown קצר שמכריז על הפרק העסקי. הוא מציג את הרעיון: לשלב שלושה signals לציון אחד 0-100 שאפשר לכייל את המשקולות שלו לפי סוג עסק. קפה רוצה linger גבוה; קיוסק רוצה throughput גבוה. שלושת ה-signals: **Volume** = median footfall (ביקוש גולמי), **Linger** = שיעור אנשים שעוצרים (engagement / conversion potential), **Consistency** = coefficient of variation נמוך (תנועה יציבה עדיפה על ספייקי).

**למה:** הפרויקט הוא "פעילות מסחרית", והמדד הסופי צריך להיות באחוזים או ב-0-100 לתמצות מובנה שאפשר להשוות בין אתרים. הבחירה בשלוש שכבות שונות (נפח, איכות תעסוקה, יציבות) מכסה 3 בעיות שונות של מיקום מסחרי.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 6. "Is it worth opening a business here?" - a simple score

Combine three signals into one 0-100 score. Tune the weights to your business type (a cafe wants high
*linger*; a kiosk wants high *throughput*).

- **Volume** - median footfall (raw demand).
- **Linger** - share of people who stop (engagement / conversion potential).
- **Consistency** - low coefficient of variation (steady traffic beats spiky).
```

<a id="cell-31"></a>
<div dir="rtl">

### תא 31 - [code] - חישוב ציון עסקי משוקלל

**מה עושה:** מגדיר `business_score(footfall_df, dwell_df, w=(0.5, 0.3, 0.2))` שמקבל את שני ה-DataFrames מהשכבות הקודמות ומחזיר dict. שלבי החישוב: (1) שולף `people = footfall_df['person'].dropna()`; אם ריק - מחזיר `{'volume_median': 0, 'linger_rate': 0, 'consistency': 0, 'score_0_100': None, 'note': 'insufficient data'}`. הבחירה להחזיר `None` בשדה הציון עצמו במקום 0.0 חשובה: 0.0 נראה כמו "אתר גרוע", בעוד ש-None אומר בגלוי "אין מספיק דאטה". (2) `volume = median(people)`. (3) `mean = mean(people)`; אם `mean > 0` אז `cv = std/mean`, `consistency = max(0, 1 - cv)`. אם `mean == 0` (חלון של כולם-אפס) אז `consistency = 1.0` (עקבי לחלוטין - אף אחד תמיד). (4) `linger`: `(is_person & dwell_s >= 25).sum() / max(1, is_person.sum())`. (5) `vol_norm = min(1.0, volume / 40.0)` - נורמליזציה שבה 40 אנשים/פריים נחשב "מלא"; אפשר לכייל לפי FOV של המצלמה. (6) `score = 100 * (w[0]*vol_norm + w[1]*linger + w[2]*consistency)` - שקלול ל-0-100. מדפיס את שם המצלמה ומחזיר את ה-dict.

**למה:** ה-guard על `people.dropna()` והפרדה בין "empty" ל-"all zero" הם תיקון מבאג שהיה בעבר: ‏הקוד הישן `if people.mean() else 1.0` נכשל כאשר mean = NaN (`NaN is truthy in Python`), מה שהיה מוליד NaN בהמשך שהחתך בשקט ל-0.0 ונקרא "אתר גרוע" במקום "אין דאטה". הגרסה החדשה נאמנה יותר: מבחין במפורש בין מקרים ומחזיר תוצאה מדויקת (או None).

**אלטרנטיבות:** משקלים אחרים לגמרי היו מתאימים לסוגי עסקים אחרים (‏`w=(0.8, 0.1, 0.1)` לקיוסק ‏"grab and go"; ‏`w=(0.2, 0.6, 0.2)` לקפה קונדיטוריה). הפונקציה מקבלת `w` כפרמטר בכוונה כדי לאפשר לזוזז.

**פלטים:** שם המצלמה בשורה, ואז dict בטבלה של Jupyter כמו:
```
{'volume_median': 6.0,
 'linger_rate': 0.12,
 'consistency': 0.87,
 'score_0_100': 34.2}
```

</div>

```python
def business_score(footfall_df, dwell_df, w=(0.5, 0.3, 0.2)):
    # Guard against an empty / all-NaN footfall series: the old
    # `if people.mean() else 1.0` fell through on NaN (NaN is truthy in
    # Python), producing NaN math that quietly clipped to 0.0 and read as
    # "bad site" instead of "insufficient data". Now we short-circuit
    # honestly.
    people = footfall_df['person'].dropna()
    if len(people) == 0:
        return {'volume_median': 0.0, 'linger_rate': 0.0,
                'consistency': 0.0, 'score_0_100': None,
                'note': 'insufficient data (no footfall samples)'}
    volume = float(people.median())
    mean = float(people.mean())
    if mean > 0:
        cv = float(people.std() / mean)
        consistency = max(0.0, 1 - cv)
    else:
        # Everyone-zero window: perfectly consistent but zero traffic - the
        # score reflects that via `vol_norm = 0`, no need to fake a cv.
        consistency = 1.0
    is_p = dwell_df['class'] == 'person'
    linger = float((is_p & (dwell_df['dwell_s'] >= 25)).sum()
                   / max(1, is_p.sum())) if len(dwell_df) else 0.0
    vol_norm = min(1.0, volume / 40.0)  # ~40 people/frame treated as 'very busy'; tune per camera FOV
    score = 100 * (w[0]*vol_norm + w[1]*linger + w[2]*consistency)
    return {'volume_median': round(volume, 1),
            'linger_rate': round(linger, 2),
            'consistency': round(consistency, 2),
            'score_0_100': round(score, 1)}

print(cam['name'])
business_score(df, dwell)
```

<a id="cell-32"></a>
<div dir="rtl">

### תא 32 - [markdown] - כותרת: 7. Compare with the live cloud dashboard

**מה עושה:** תא markdown בגודל בינוני (629 תווים) שהוא הכי חשוב בשכבה של השוואה. הוא מסביר לקורא שכל מה שהוא ראה עד כה היה **ניתוח מקומי** - דקה של דגימה על מצלמה אחת. הקולקטור בענן רץ ללא הפסקה על VM של GCP, ומצטבר 4 מצלמות x 24 שעות ל-Firestore, וה-HTML dashboard נרשם עליו. השוואת השניים עונה על 3 שאלות אמיתיות: (1) האם הרגע שדגמתי מייצג את היום כולו? (הדקה שלי מול הצ'רט של 24 השעות); (2) האם אני מגיע ל-peak, ל-valley, או לממוצע? (3) האם קרה אנומליה ב-24 השעות האחרונות שפספסתי בגלל שהדגימה שלי הייתה עכשיו?

**למה:** ה-value proposition המרכזי של המחברת הוא לא רק "מודל חזק" אלא "השוואה יומית". תא markdown זה מסביר במפורש שהיוזר לא מעלה שום דבר לשום מקום - הכל הוא HTML שקורא מ-Firestore. הפרדה נקייה בין "מה שהמחברת עושה" (חישוב) לבין "מה שהדשבורד עושה" (קריאה בלבד).

**פלטים:** טקסט בלבד. שים לב שהתא המקורי משתמש בקו-מפריד ארוך (הקידוד U+2014), זהו התוכן המקורי של המחברת והוא נשאר כפי שהוא בקוד המקור.

</div>

```markdown
## 7. Compare with the live cloud dashboard

The rest of this notebook was your **local** analysis — a minute of sampling on
one camera. The cloud collector has been running non-stop on a GCP VM,
accumulating 4 cameras × 24 hours into Firestore, and the HTML dashboard below
subscribes to that. Comparing the two answers real questions:

- Is the moment I sampled representative of the whole day? (my minute vs the 24h chart)
- Am I hitting a peak, a valley, or the average?
- Did any anomaly fire in the last 24 hours that I missed by sampling now?

Nothing here writes to Firestore — it's a plain HTML page that reads from it.
```

<a id="cell-33"></a>
<div dir="rtl">

### תא 33 - [code] - הפעלת הדשבורד המקומי עם המצלמות שבחרת

**מה עושה:** התא הגדול ביותר במחברת (5266 תווים). מבצע ארבעה דברים מקבילים: (1) **בונה `web/local_grid.json`** מהמצלמות שהמשתמש בחר, כך שהדשבורד יציג את הבחירה שלו ולא את הגריד של ה-VM; (2) **מפעיל שרת HTTP מקומי** על port פנוי בטווח 8000-8020; (3) **פותח את הדשבורד בדפדפן**; (4) **מטמיע את הדשבורד כ-IFrame** במחברת. פירוט מלא:

`_yt_embed(vid)` בונה URL של embed של YouTube עם autoplay + mute + playsinline + enablejsapi. `_local_grid_slot(i, cam_id)` מייצר את הרשומה של slot אחד: שולף את המצלמה מ-`CAMERAS[cam_id]`, בונה `where` מהעיר והמדינה, ומנסה למלא אחד משלושה שדות פוטנציאליים: `placeholder_hls` (זרם HLS ישיר), `placeholder_embed` (embed של YouTube), `placeholder_page` (fallback: פשוט קישור לדף המקור). לפי `kind` של המצלמה: אם `youtube` - חילוץ vid מ-URL עם regex `_YOUTUBE_RE` והכנסת iframe embed; אם `webcamera24` - `resolve_webcamera24` מחזיר את ה-master playlist ואז מנסים לחלץ tvkur id או youtube vid; אם `hls` - אם ה-URL הוא tvkur ("tvkur/{id}/..."), משתמש ב-proxy מקומי `/tvkur/{id}/master.m3u8` (מסביב לבעיית CORS/geo של IBB); אחרת ה-URL הישיר. אם `SELECTED_CAMS` קיים, כותב `web/local_grid.json` עם המצלמות המסודרות; אחרת - מוחק את הקובץ אם קיים (כך שהדשבורד יחזור ל-VM grid).

חלק שרת: משתמש ב-`DashboardHandler` מ-`app.dashboard_server`. מפעיל sweep על פורטים 8000-8020, בוחר את הראשון הפנוי, ומריץ `ThreadingHTTPServer` בת'רד רקע (`daemon=True`). שומר את השרת ב-`sys.modules['__main__']._dash_server` כדי שריצות חוזרות של אותו kernel יזהו את השרת הקיים ולא יפתחו חדש. פותח `dash_url = http://localhost:{PORT}/?mode=main` בדפדפן פעם אחת (‏`_dash_browser_opened` guard).

מציג `IFrame(dash_url, width='100%', height=640)` בתא של Jupyter - זה מטמיע את הדשבורד ישירות במחברת.

**למה:** הבחירה של `?mode=main` בבירור מסבירה שהדשבורד יש לו שני מצבים (main / VM-mirror). ב-main מוצגים panels שרלוונטיים למחברת הראשית: 2x2 grid של המצלמות שבחרת + Search + Snapshots. ב-VM-mirror (התאום) מוצג יותר - כי התאום שם משתמש בפיצ'רים של הקולקטור. הבחירה בפורט dynamic (‏8000-8020) מונעת התנגשות עם דשבורדים אחרים שכבר רצים (למשל אם המשתמש הריץ את הדשבורד ידנית מ-terminal).

**אלטרנטיבות:** אפשר היה להשתמש ב-`ngrok` לפתיחת public URL, אבל זה דורש הרשמה ושימוש בשירות חיצוני. שרת מקומי פשוט ובטוח. אפשר היה גם להשתמש ב-`jupyter-server-proxy` לחשוף את הדשבורד תחת ה-URL של Jupyter עצמו, אבל זה דורש התקנה נוספת.

**פלטים:** שורות כמו:
```
Dashboard grid -> YOUR 4 picked cameras (turkey): ['Sarachane Square', ...]
Dashboard server started at http://localhost:8000/?mode=main
```
אז HTML קטן עם קישור לדשבורד, ואז IFrame של הדשבורד עצמו (‏640px גובה) מוטמע במחברת עם live video tiles.

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
    print(f'Dashboard server started at http://localhost:{DASHBOARD_PORT}/?mode=main')

# `?mode=main` tells web/app.js which set of dashboard panels to render:
# see the "Dual-mode dashboard" comment block near the top of src/web/app.js.
dash_url = f'http://localhost:{DASHBOARD_PORT}/?mode=main'
if not getattr(_main, '_dash_browser_opened', False):
    try:    webbrowser.open(dash_url, new=2)
    except Exception: pass
    _main._dash_browser_opened = True

from IPython.display import display, HTML, IFrame
display(HTML(f'<p><b>Live dashboard</b> (your picked cameras; counts from the '
             f'cloud collector): <a href="{dash_url}" target="_blank">{dash_url}</a></p>'))
display(IFrame(dash_url, width='100%', height=640))
```

<a id="cell-34"></a>
<div dir="rtl">

### תא 34 - [markdown] - כותרת: 8. Compare multiple commercial sites

**מה עושה:** תא markdown קצר שמכריז על שכבת ההשוואה בין אתרים. השאלה עוברת מ-"מה קורה במצלמה X" ל-"איזה מצלמה עסוקה יותר". מריצים sampler של פוטפול על מספר מצלמות ומדרגים לפי פעילות - וזה בדיוק הקלט להחלטה של בחירת מיקום.

**למה:** ניתוח של מצלמה בודדת (פרקים 3-5) אינו מספיק לפרויקט מסחרי. הבחירה בין 4 אתרים דורשת השוואה משוקללת. פרק זה מספק את הקוד המפורש לזה.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 8. Compare multiple commercial sites

Loop the footfall sampler over several cameras to rank locations by activity - the input to a
site-selection decision.
```

<a id="cell-35"></a>
<div dir="rtl">

### תא 35 - [code] - דירוג המצלמות שבחרת לפי פעילות

**מה עושה:** מבצע checkpoint של `SELECTED_CAMS_APPLIED`. אחר-כך לולאה על `SELECTED_CAMS` (‏4 מצלמות שהמשתמש בחר). לכל מזהה: (1) שולף `c = CAMERAS[cid]`, ‏skip אם ריק או ללא URL; (2) `resolve_stream(c)` בתוך try/except; (3) probe מהיר: `grab_frame(url)` - אם None, skip עם הודעה "no frame from stream"; (4) `footfall_series(url, c['name'], interval_s=10, duration_min=0.5)` - חצי דקה, כלומר 3 דגימות. סופר `median_people` ו-`max_people` לתוך `summary`. בסוף: אם `summary` לא ריק - מציג DataFrame ממוין לפי `median_people` יורד; אחרת מדפיס "No camera... produced usable frames".

**למה:** duration_min=0.5 (30 שניות) x 4 מצלמות = 2 דקות של ריצה בערך - זמן סביר לניתוח השוואתי. הבחירה ב-`median_people` להשוואה (ולא ב-mean) חסינה יותר לספייק זמני. הצגה מדורגת לפי median = ממש קלט להחלטה.

**אלטרנטיבות:** אפשר היה להריץ במקביל את 4 המצלמות עם `ThreadPoolExecutor(max_workers=4)`, אבל היה כאן שיקול של לא לעשות DDOS לספקי הזרמים; ריצה sequential עדינה יותר. אפשר היה גם להוסיף `linger_rate` להשוואה (כמו בציון של תא 31), אבל בשכבה זו התמקדו רק בנפח כדי להישאר פשוטים.

**פלטים:** אם רץ בסדר: 4 בלוקים של דגימה (3 שורות כל אחד), ואז DataFrame ממוין:
```
                site   median_people   max_people
Sarachane Square     8.0             12
Taksim Square        6.5              9
...
```

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

<a id="cell-36"></a>
<div dir="rtl">

### תא 36 - [markdown] - כותרת: 9. Live summary

**מה עושה:** תא markdown קצר שמסביר את מטרת התא הבא: לרכז את הכל שהמחברת ראתה בריצה זו לבלוק אחד: האנומליות שסומנו על פני כל מצלמה שנדגמה, הסיכומים של re-ID, ופלוט קטן שמציג את כל האנומליות על ציר זמן אחד. הרצה חוזרת של המחברת יוצרת מחדש את הבלוק - אין timestamps ישנים מסשנים של אחרים.

**למה:** אחרי 8 פרקים של ניתוח, למשתמש קל להתאבד בפרטים. תא סיכום שממקם את הכל בגושים קצרים ברורים = "מה בפועל מצאנו?" - שירות טוב.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 9. Live summary - what did we find?

Pulls everything the notebook saw on this run into a single block: the anomalies
flagged across every sampled camera, the re-ID totals, and a tiny visualisation
plotting all anomalies on the same timeline. Re-running the notebook regenerates
this from scratch - no stale timestamps from someone else's session leak through.
```

<a id="cell-37"></a>
<div dir="rtl">

### תא 37 - [code] - איסוף כל הממצאים לסיכום אחד + פלוט

**מה עושה:** תא ארוך (3387 תווים) שאוסף את כל מה שנמצא בריצה נוכחית לבלוק סיכום. עטוף ב-try/except מעל הכל כדי שאם משהו לא הוגדר (מצב ריצה מקוצרת) הסיכום לא ייפול. פירוט: (1) מדפיס timestamp עכשיו ב-UTC + `Live camera for this run: {name}`. (2) אנומליות: אם `df` קיים ולא ריק, מסנן `df[anomaly==True]`, מוסיף `cam` column, ומוסיף ל-`anom_frames`. אם יש רשומות - מדפיס טבלה עם `to_string(index=False)`; אם 0 - מדפיס "0 (too few samples for the z-score window to trip, or the scene was steady)". (3) Re-ID rollup: אם `reid` קיים, קורא ל-`stats(CAM_ID)` ומדפיס total_unique, total_sightings, ואת ה-per_class breakdown. גם `top_regulars(CAM_ID, n=5)` אם קיימים. (4) פלוט תמידי (Always-on visual): אם ה-`df` תקין, בונה matplotlib figure 12x3.5 עם 3 סדרות: `people` (עיגולים כחולים), `vehicles` (ריבועים כתומים), ו-`anomaly` (X אדומים גדולים). כותרת, ציר Y, ציר X, legend, grid ‏alpha=0.3. (5) סיום עם 4 שורות של הפניות ל-collector + dashboard שיוצרים מצב שהמחברת לא רצה: איך להריץ את הקולקטור כ-daemon (`python -m app.collector --interval 20 --country turkey`), איך לפתוח דשבורד (`python serve.py`), URL (`http://localhost:8000`).

**למה:** הפלוט תמידי (`Always-on visual`) עם ה-comment המפורש בקוד: "‏Build the plot only when we have data; do NOT call ax.legend() on an empty axes (that produces the 'No artists with labels found' warning)." זה לקח שנפל בעבר. `.legend()` על axes ריק מייצר warning שגורם למי שקורא את הפלט לחשוב ש-something broke.

**אלטרנטיבות:** אפשר היה לפרסם JSON סיכום לקובץ, אבל הפורמט הטבעי ב-Jupyter הוא print + plot. הרעיון של סיכום שרץ בסוף הוא הרגל מ-CI/CD: לתת למפעיל תמונה מהירה אחרי הרבה שלבים.

**פלטים:** בלוק טקסט ארוך עם 3 חלקים (זמן, אנומליות, re-ID) + פלוט מטריאלי אחד + 4 שורות של הפניות ל-collector/dashboard.

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
        print(f"\nAnomalies flagged (robust rolling z > 3.5): {len(anom)}")
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

<a id="cell-38"></a>
<div dir="rtl">

### תא 38 - [markdown] - כותרת: 10. Accuracy calibration

**מה עושה:** תא markdown ארוך (853 תווים) שמסביר את פרק ה-calibration. הדשבורד אמין בדיוק כמו שYOLO אמין על המצלמות הספציפיות. הפרק הזה מודד את הדיוק: לוכד פריימים מ-4 המצלמות של הגריד, מריץ את הדטקטור בשני input sizes (‏640 - ברירת מחדל ישנה, 960 - ברירת מחדל נוכחית של הקולקטור), ואז המשתמש סופר בעצמו אנשים/רכבים ומקבל MAE + bias פר מצלמה ופר size. ה-workflow: **10a** לוכד פריימים + predictions ל-`data/calibration/`; **10b** מציג כל פריים ומקבל את `people,vehicles` בהזנה; **10c** מדפיס טבלת דיוק והמלצת conf/imgsz. פידבק חזרה ל-pipeline: ה-`imgsz` המנצח הולך ל-`--imgsz` של הקולקטור; מצלמה עם bias שיטתי מקבלת override של `"conf"` ב-`app/cameras.py` (‏`bias < 0` = הורד conf, `bias > 0` = העלה).

**למה:** זה החוב המדעי של הפרויקט: מודדים דיוק במפורש במקום לסמוך על YOLO. כל שכבה מעל זה בונה על ההנחה שהמספרים אמינים. Calibration הופך הנחה למדידה.

**פלטים:** טקסט בלבד.

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

<a id="cell-39"></a>
<div dir="rtl">

### תא 39 - [code] - 10a: לכידת פריימי calibration

**מה עושה:** אחרי checkpoint של SELECTED, מגדיר קבועים: `CALIB_DIR = data/calibration/`, `FRAMES_PER_CAM = 6` (4 x 6 = 24 פריימים - יעד של 20-30), `IMG_SIZES = (640, 960)`, `CALIB_CONF = 0.30`. אז לולאה על כל מצלמה ב-`SELECTED_CAMS` (או `GRID_CAMERAS` אם הבחירה לא הופעלה): מנסה `resolve_stream`, skip אם נכשל. לולאה פנימית של 6 פריימים: `grab_burst(url, n=1)` שולף פריים בודד; אם ריק - `MISS`; אם קיים - שומר `{stem}.jpg` ב-CALIB_DIR עם `cv2.imwrite`. אז לולאה נוספת על שני ה-input sizes: `detect_with_boxes(model, frame, conf=CALIB_CONF, imgsz=size)` מריץ בשתי גדלים ושומר `person_640, vehicles_640, person_960, vehicles_960`. גם שומר `{stem}_annotated.jpg` ‏ (`annotate` = הפריים עם BBoxes של המודל, ב-`imgsz=max(IMG_SIZES)=960`) - זה הפריים שיוצג ב-10b. שמירה של `entry` ל-`samples` ו-sleep של 2 שניות בין פריימים (כדי לתת לזרם להתקדם ולמנוע פריימים כמעט זהים). בסוף: `predictions.json` נשמר ומודפס `24 frames -> {path}`.

**למה:** שני ה-input sizes בבת אחת הם ליבת המחקר: לפריים אחד יש שני predictions, וההשוואה שלהם מול ה-ground truth מגלה איזה גודל עדיף לגרסת הקולקטור. שמירה של הפריים ה-annotated (ולא הגולמי) ב-10b חשובה מאוד: המשתמש רואה את מה שהמודל מצא ומרגיש שיש לו הרבה מידע לספור.

**אלטרנטיבות:** אפשר היה להוסיף גם `imgsz=1280`, אבל אז יותר predictions לפריים = יותר משתנים - וברוב המחשבים הביתיים 1280 חייב GPU בשביל להיות מעשי. שני sizes הוא המינימום המשמעותי.

**פלטים:** בלוקים של `{cam_id}: captured N frames` פר מצלמה, ואז `24 frames -> C:\...\data\calibration`.

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

<a id="cell-40"></a>
<div dir="rtl">

### תא 40 - [code] - 10b: תיוג אינטראקטיבי של פריימים

**מה עושה:** תא אינטראקטיבי (1955 תווים) שקורא ל-`input()` פר פריים. הוא מוגן בגלוי מפני Run All: הוא בודק `RUN_LABELING = globals().get('RUN_LABELING', False)`. אם False - מדפיס הוראות ("Labeling skipped. When you want to label: set RUN_LABELING = True..."). אם True - טוען את `predictions.json`, ולולאה על כל פריים: (1) קורא `{stem}_annotated.jpg`, מרנדר עם matplotlib בגודל 12x7, כותרת שמציגה את הספירות של המודל ב-960 (`model@960: person=X vehicles=Y`) - כך שהמשתמש רואה מה המודל מצא ואיפה, וסופר לפי ההנחיה. (2) `input(f"{stem} true 'people,vehicles' (Enter=skip, q=stop): ")`. (3) לוגיקת שיטת הזנה: Enter ריק = דלג, `q` = עצור לגמרי, מחרוזת של שני מספרים מופרדים בפסיק (`p_true, v_true = int(x) for x in raw.split(',')`) = תווים אמת. אם פרסינג נכשל - `could not parse - skipped`. שמירה של `labeled` list ולסוף כתיבה של `labeled.json`.

**למה:** ה-`RUN_LABELING = False` default הוא הגנה חשובה מ-Run All: תא זה חייב input אנושי ובלעדיו הוא יקפיא את כל המחברת עד timeout. הפיתרון הוא לגרום ל-Run All לדלג עליו לחלוטין. משתמש שרוצה לתייג משנה את הקבוע ב-cell code ומריץ תא זה בלבד. הבחירה להראות פריים annotated (עם ה-BBoxes של המודל) ולא פריים גולמי היא תזכורת למשתמש איפה המודל טעה או פספס - כך הוא סופר את הבסיס, לא את המודל.

**אלטרנטיבות:** אפשר היה לעטוף ב-`ipywidgets` עם dropdown ו-slider, אבל זה מוסיף תלות של מודול, ו-`input()` פשוט תמיד עובד. אפשר היה להוציא את זה לסקריפט CLI נפרד (`python -m app.calibrate`), אבל אז המשתמש היה מפוזר בין הרצות של המחברת ל-shell.

**פלטים:** אם RUN_LABELING = False: 3 שורות של הוראות. אם True: 24 פריימים מוצגים אחד אחרי השני, ליד כל אחד prompt של הזנה, ובסוף `labeled 24 frames -> ....\labeled.json`.

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

<a id="cell-41"></a>
<div dir="rtl">

### תא 41 - [code] - 10c: דוח דיוק MAE + bias

**מה עושה:** טוען את `labeled.json`. אם ריק - מרים חריגה מנומקת `_LabelFirst` שאומרת "PAUSED: no labeled frames yet - set RUN_LABELING = True in 10b, label a few frames, then run this cell again." אם יש - יוצר `cal = pd.DataFrame(rows)`. מחשב `overall`: לכל size (‏640, 960) x metric (person, vehicles): `err = cal[f'{metric}_{size}'] - cal[f'{metric}_true']`, `MAE = mean(abs(err))`, `bias = mean(err)` (שלילי = undercount, חיובי = overcount). מדפיס טבלה. מחשב `per_cam` על ה-size הכי גדול (`best = max(IMG_SIZES) = 960`): לכל camera x metric את MAE ו-bias. מדפיס. מסיים עם בלוק הסבר: **bias < 0 -> undercount שיטתי: הורד conf של המצלמה (הוסף למשל `"conf": 0.25` ב-app/cameras.py). bias > 0 -> overcount: העלה. אם MAE@960 < MAE@640 (טיפוסי לצילומים רחבים כאלה), השאר `--imgsz 960` של הקולקטור; אחרת חזור ל-640. הרץ שוב אחרי כל camera swap או שינוי משקולות.**

**למה:** MAE + bias הם הצמד המדעי הסטנדרטי למדידת דיוק של counting: MAE הוא "כמה טעינו בממוצע"; bias הוא "האם אנחנו נוטים לסה או לרוב". שילוב שלהם עונה על שתי שאלות שונות. הפורמט של הבלוק הסופי (עם "how to read this") הוא מדריך אקצייני קונקרטי - לא צריך לקרוא מסמך נוסף.

**אלטרנטיבות:** אפשר היה להוציא גם RMSE (root mean squared error) שמעניש טעויות גדולות יותר, אבל MAE יותר פרשני עבור counting - "בממוצע פספסנו 2 אנשים" נשמע ברור יותר מ-"RMSE 3.4". Precision/Recall היו רלוונטיים אם היינו סופרים per-object hits, אבל כאן אנחנו סופרים אגגריגייטים ולא detections יחידים.

**פלטים:** שני בלוקים של DataFrame עם MAE + bias, ואז 4 שורות של הסבר איך לקרוא אותם.

</div>

```python
## --- 10c. Accuracy report: MAE + bias per input size and per camera ---
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

<a id="cell-42"></a>
<div dir="rtl">

### תא 42 - [markdown] - כותרת: 11. Forecasting

**מה עושה:** תא markdown ארוך (1805 תווים) שמכריז על פרק הפרוגנוזה. כל מה שקרה עד עכשיו מדד את ההווה; הפרק הזה חוזה את העתיד הקרוב. הקולקטור כתב ספירות פר-דגימה ל-Firestore במשך שבועות - סדרת זמן אמיתית (עם רעש) פר מצלמה: פערים איפה שזרם מת, שינויי משטר איפה שהגריד עבר מדינה, ריתמוס יום/לילה. המחברת תעבור בסדר על 4 שלבים: (1) **Fetch + cache** - Firestore הוא זיכרון קצר-טווח של הקולקטור, לא ארכיון. ההיסטוריה שם חוזרת רק כמה ימים אחורה (רשומות ישנות נמחקו במסגרת cost cleanup), וה-tier החינמי מאפשר 50k reads/day. תא הראשון מושך פעם אחת מה שקיים ל-`data/footfall_history.csv` וכל ריצה שאחרי מושכת רק את ה-delta. ה-CSV הוא זיכרון ארוך-הטווח של הפרויקט; הוא צומח כל ריצה, והמודלים שממוקמים בהמשך משתפרים ככל שהוא גדול יותר. (2) **Resample** - דגימות גולמיות מגיעות כל 40-90 שניות עם jitter ופערים; כל המודלים עובדים על גריד נקי של 15 דקות פר מצלמה. (3) **Baseline ladder + ridge** - persistence, seasonal-naive (אותה שעה אתמול), פרופיל של hour-of-day/hour-of-week, ו-ridge regression closed-form על פיצ'רי lag/seasonal (numpy טהור, כ-15 שורות). כלל honesty: מודל מרוויח את מקומו רק אם הוא מנצח את seasonal-naive. (4) **GRU (torch)** - רשת recurrent קטנה (אותה torch שרצה YOLO - אפס התקנות חדשות) שקוראת את 24 השעות האחרונות ומחזה 12 שעות קדימה במכה אחת. משווים לפשוטים על אותו split. המנצח ב-1h horizon הוא מה שאופרטור היה שולח לפרודקשן.

**למה:** הפילוסופיה של "כל מודל חייב לנצח את seasonal-naive" היא ליבת המשמעת המדעית של הפרק. הרבה פרויקטים "מציגים" ML מתוחכם אבל לא בודקים אם הבסיס הפשוט (24h ago) עושה טוב יותר. כאן זה נאכף.

**פלטים:** טקסט בלבד.

</div>

```markdown
## 11. Forecasting - expected activity + deviation anomalies

Everything so far measures the PRESENT; this chapter predicts the NEAR
FUTURE. The collector has been writing per-sample counts to Firestore for
weeks - a real (messy) time series per camera: gaps where a stream died,
regime changes where the grid switched countries, day/night rhythm.

What happens here, in order:

1. **Fetch + cache** - Firestore is the collector's short-term memory, not
   an archive: history there only goes back a few days (old docs were
   purged during a cost cleanup), and the free tier allows ~50k document
   reads/day. So the first cell pulls whatever exists ONCE into
   `data/footfall_history.csv` and every later run fetches only the delta.
   The CSV is the project's long-term memory - it grows every time you run
   the cell, and the models below sharpen as it grows. Run it at least
   every couple of days so nothing is lost.
2. **Resample** - raw samples land every 40-90s with jitter and gaps; all
   models work on a clean 15-minute grid per camera.
3. **Baseline ladder + ridge** - persistence, seasonal-naive (same time
   yesterday), an hour-of-day/week profile, and a closed-form ridge
   regression on lag/seasonal features (pure numpy, ~15 lines). Honest
   rule: a model earns its keep only by beating seasonal-naive.
4. **GRU (torch)** - a small recurrent network (the same torch that runs
   YOLO - zero new installs) reads the last 24h and predicts the next 12h
   in one shot. Compared against the simple models on the SAME split -
   the champion at the 1h horizon is what an operator would ship.

The VM twin notebook runs only steps 1-2 + a numpy profile forecaster -
that version could run on the 1 GB e2-micro itself. This notebook is the
lab that proves which model deserves that job.
```

<a id="cell-43"></a>
<div dir="rtl">

### תא 43 - [code] - 11a: משיכת היסטוריה מ-Firestore

**מה עושה:** תא ארוך (3816 תווים) שאחראי על synchronization בין מקור הענני (Firestore) לבין קאש המקומי (`data/footfall_history.csv`). קבועים: `CACHE = DATA_DIR / 'footfall_history.csv'`, `READ_BUDGET = 30_000` (מקסימום reads בריצה אחת), `PAGE = 5_000` (גודל דף בקריאה). פונקציית `_find_credentials()` מחפשת JSON של service account ב-3 מקומות: משתני סביבה (`FIREBASE_CREDENTIALS`, `GOOGLE_APPLICATION_CREDENTIALS`), ואז ב-project layout (`_src_dir.parent / *firebase-adminsdk*.json`, ‏`_src_dir`, `Path.cwd()`). מחזיר path ראשון שמצא או None. `fetch_footfall_delta()`: (1) קורא קאש קיים; אם יש, שולף `since = cached['ts'].max()` (ISO strings משווים כמו שצריך); (2) מחפש credentials; אם אין ויש קאש - מדפיס אזהרה וממשיך עם קאש; אם אין credentials וגם אין קאש - מרים RuntimeError. (3) מאתחל firebase_admin ומקבל firestore client. (4) `q = db.collection('footfall'); if since: q = q.where('ts', '>', since)`, ואז `q.count().get()[0][0].value` הוא aggregation שנותן מספר docs בעלות של קריאה 1 לכל 1000 (חיסכון). (5) לולאת paging: שולף עד `min(n_new, READ_BUDGET)`, בקבוצות של 5000, מעדכן cursor. לכל doc: `{'ts', 'cam_id', 'person', 'vehicles', 'ok', 'night'}` הופך לשורה ב-`rows`. (6) `df = concat([cached, fresh])`, אז `.dropna` על ts+cam_id, `.drop_duplicates` על ts+cam_id, ‏`sort_values('ts')`. שמירה ל-CSV. מדפיס `cache now {N} rows`.

**למה:** ה-strategy של incremental + budget הוא הכרחי: הפרויקט הוא Firebase Spark plan עם 50k reads/day. אם היוזר יריץ את המחברת פעמיים ביום, הוא יעבור מכסה אם כל ריצה תמשוך את כל האוסף. פתרון: לשמור קאש מקומי ולמשוך רק delta. `q.count()` הוא aggregation query עלות-נמוכה שמונע לבזבז קריאות סתם על לספור. הבחירה ב-ISO strings ל-compare (במקום Timestamp objects) עובדת כי `2026-08-13T14:33:00Z` > `2026-08-13T14:32:59Z` באופן לקסיקוגרפי, וזה חוסך conversion.

**אלטרנטיבות:** אפשר היה להשתמש ב-Firestore snapshot listener (`onSnapshot`) כמו שהדשבורד עושה, אבל זה דורש חיבור פתוח מתמשך - לא מתאים למחברת שרצה פעם ביומיים. incremental pull הוא הפתרון הטבעי.

**פלטים:**
```
cache: 12345 rows (through 2026-08-12T18:30:00)
new docs on the server: 456
  fetched 456...
cache now 12801 rows, 2026-07-25 -> 2026-08-13, 4 cameras -> footfall_history.csv
```
ואז DataFrame עם top 12 מצלמות לפי rows count.

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

<a id="cell-44"></a>
<div dir="rtl">

### תא 44 - [code] - 11b: Resample לרשת 15 דקות + חיתוך עידן

**מה עושה:** תא ארוך (3953 תווים) עם 3 חלקים ברורים. (1) **Resampling**: מייבא `cam_tzinfo` מ-`app.collector`. קובע `BIN_MIN = 15`, `BINS_PER_DAY = 96`, `MIN_HOURS = 36` (מצלמה חייבת 36 שעות של bins observed כדי להיכלל). מסנן `hist[hist.ok == 1]` (רק דגימות מוצלחות), ממיר `ts` ל-datetime UTC, ממיר את הזרם ל-`.resample('15min').median()` פר מצלמה (median ולא mean כדי שאירוע אחד לא יגרור bin שלם). מוסיף עמודת `n` לספירת דגימות ב-bin. (2) **Local-time features**: לוקח את tzinfo של המצלמה, מחשב hod (hour-of-day bin, 0-95) ו-dow (day-of-week 0-6) מ-timestamp מקומי, ומוסיף `how = dow * 96 + hod` (hour-of-week bin, 0-671). ההערה בקוד מתעדת באג עבר: הקוד הקודם השתמש ב-`bins['local'] = bins['ts']` ואז tz_convert הוליד עמודה עם UTC אמיתי - כי pandas rebrand את התג של tz באי-הצבה. הפתרון הוא לחשב hod/dow ישירות מ-Series חדשה של tz_convert בכל איטרציה. (3) **Era detection**: קבוע `ERA_GAP_H = 72`. מזהה פערים גדולים מ-72 שעות בציר הזמן; אם יש - חותך את הכל לפני ה-gap האחרון. הסיבה מפורטת בהערה: האוסף מחזיק שאריות של epoch מוקדם יותר (הגריד המקורי מיוני) המופרד מהנוכחי בפער של שבועות (הקולקטור טייל למדינות אחרות ורשומות ישנות נמחקו). ערבוב epochs שובר כל time-based split. אז מסנן את `bins` לרק אחרי `era_start`. אח"כ מסנן `ELIGIBLE = מצלמות עם >= 144 observed bins` (‏36 שעות x 4 bins). מדפיס את הרשימה, את `span` (bins_obs, first, last, span_days). לבסוף heatmap של coverage: pivot מצלמה x יום, imshow עם cmap `YlGn` (0=לבן, גבוה=ירוק כהה), שם `bins/day` על ה-colorbar - כך שרואים איפה כל מצלמה הייתה live פר יום, החללים = מדינה אחרת או stream dark.

**למה:** ה-era cut הוא ה-decision critical של הפרק. בלעדיו כל מודל היה מגיב לפער של שבועות כאילו זה "אירוע חריג" - ה-target באוגוסט יופיע בlagים של יוני. חיתוך לעידן האחרון = נתונים homogeneous.

**אלטרנטיבות:** אפשר היה `resample('1H').mean()` בלי median, אבל אז אירוע אחד היה מטה bin שלם. אפשר היה גם `MIN_HOURS = 24` כדי לכלול מצלמות עם פחות היסטוריה, אבל אז המודלים אימנו על דאטה דל.

**פלטים:** אם יש era cut: הודעה `era cut: modeling on data from 2026-08-01 12:15 on (dropped 2345 bins of an older epoch across a >72h hole)`. ואז `eligible cameras (>= 36h observed): [...]`. ואז DataFrame span. ואז heatmap matplotlib של coverage per day.

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
# hod/dow are computed directly per-camera from a fresh tz_convert'd
# Series. An earlier version pre-init'd `bins['local'] = bins['ts']`
# (UTC-typed) and wrote tz_convert'd Istanbul times back into it; some
# pandas versions re-tag such an assignment to the destination column's
# tz, and `.dt.hour` then returned UTC hours - the ridge/GRU features
# lost their local rhythm. Extracting hod/dow directly avoids the trap.
bins['hod'] = 0
bins['dow'] = 0
for cam in bins.cam_id.unique():
    m = bins.cam_id == cam
    try:
        tz = cam_tzinfo(cam)
        local = bins.loc[m, 'ts'].dt.tz_convert(tz)
    except Exception:
        local = bins.loc[m, 'ts']                # legacy cam ids: keep UTC
    bins.loc[m, 'hod'] = (local.dt.hour * (60 // BIN_MIN)
                          + local.dt.minute // BIN_MIN).astype(int)
    bins.loc[m, 'dow'] = local.dt.dayofweek.astype(int)
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

<a id="cell-45"></a>
<div dir="rtl">

### תא 45 - [code] - 11b-x: פיצ'רים חיצוניים (מזג אוויר + חגים)

**מה עושה:** תא ארוך (5194 תווים) שמושך שני מקורות נתונים חיצוניים ומצרף אותם ל-bins של תא 44. חלק ראשון - חגים: `import holidays as _hol` בתוך try/except (אם `holidays` לא מותקן, `_HOLIDAYS_OK = False` וממשיך). חלק שני - מזג אוויר: `import requests as _rq`, ומ-`app.cameras` שולף `camera_latlon` ו-`camera_country_iso2`. מגדיר `WEATHER_CACHE = data/weather_history.csv`. פונקציית `_fetch_weather_one(lat, lon, start_date, end_date)` פונה ל-`archive-api.open-meteo.com/v1/archive` (חינם, ללא key) עם `hourly: temperature_2m, precipitation, cloud_cover, timezone: UTC`. מחזיר DataFrame עם ts, temp_c, precip_mm, cloud_pct.

`_load_weather_cached(country_key, lat, lon, start, end)` בודק אם יש קאש למדינה זו; מושך רק מה שחסר (‏אחרי `have.ts.max() + 1h`). מדפיס `weather merged: X / Y bins have temperature`.

מקבץ מצלמות `ELIGIBLE` לפי `country_key = f"{lat:.4f},{lon:.4f}"` - מדינה שלמה משתמשת בקואורדינטות ה-anchor שלה (מ-`COUNTRY_LATLON` ב-`app/cameras.py`). לכל קבוצה מושך פעם אחת ומצרף. אז לכל מצלמה: resample ל-15 דקות עם `ffill(limit=4)` (‏עד 1 שעה לאחור), ומצרף עם `bins.merge(weather_bins, on=['cam_id','ts'], how='left')`.

חלק שלישי - חגים: אם `_HOLIDAYS_OK`, יוצר `_hol_cache` פר ISO2 (‏TR, TH, JP, US); לכל שורה בודק אם `local_ts.date() in _hol_cache[iso]` -> `is_holiday = 1`. גם `is_weekend = (dow >= 5)`. מדפיס כמה bins סומנו כחג / weekend. לבסוף מציג 8 שורות אחרונות של המצלמה הראשונה עם `ts, person, temp_c, precip_mm, is_holiday, is_weekend`.

**למה:** ההערה המפורשת בתחילת התא היא הצידוק: "the lift is meaningful at 3h+ horizons where the seasonal profile is a weak prior (rainy Wednesday morning is not 'the median Wednesday morning')." המודלים הפשוטים (persistence, seasonal-naive) לא רואים מזג אוויר; ‏Ridge (בתא 46) יכול. חג לאומי מוריד תנועה במרכזי עסקים ומעלה בכיכרות אירועים; יום שישי בטורקיה != יום שני.

**אלטרנטיבות:** אפשר היה לפנות ל-Meteostat או NWS API, אבל Open-Meteo הוא ה-free-tier היחיד עם archive עולמי מלא בלי key. `camera_latlon` מאפשר להתפצל למצלמה ספציפית אם היא רחוקה מ-anchor המדינה - אבל בפועל אנומליות מזג אוויר משתנות בין ערים בפחות מ-1-2 C בשעה, ואת ה-lift של פיצ'רים בפרוגנוזה הן לא רואות.

**פלטים:** ‏הודעות של fetching, ואז `weather merged: 850 / 1200 bins have temperature`, ‏`holidays flagged: 45 bins   weekends: 320 bins`, ואז דוגמת 8 שורות של המצלמה הראשונה.

</div>

```python
# 11b-x. Exogenous features - weather (Open-Meteo, free, no key) + national
# holidays + weekend flag. Ridge in 11c uses them as extra columns; the
# lift is meaningful at 3h+ horizons where the seasonal profile is a weak
# prior (rainy Wednesday morning is not "the median Wednesday morning").
#
# Weather is fetched at country-anchor coordinates (see COUNTRY_LATLON in
# app/cameras.py) - a Konya vs Istanbul afternoon differ mostly by 1-2 C
# and rain-yes/no, which the anchor captures fine for the 15-min forecast
# bins. Set "lat"/"lon" on a specific CAMERAS entry to pin its own coords.
try:
    import holidays as _hol
    _HOLIDAYS_OK = True
except ImportError:
    print("holidays not installed - pip install holidays; skipping holiday feature")
    _HOLIDAYS_OK = False

import requests as _rq
from app.cameras import camera_latlon, camera_country_iso2

WEATHER_CACHE = DATA_DIR / "weather_history.csv"

def _fetch_weather_one(lat, lon, start_date, end_date):
    """One country's hourly weather - archive API. Free, no key."""
    r = _rq.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": lat, "longitude": lon,
                        "start_date": start_date, "end_date": end_date,
                        "hourly": "temperature_2m,precipitation,cloud_cover",
                        "timezone": "UTC"}, timeout=30)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    if not h.get("time"):
        return pd.DataFrame()
    return pd.DataFrame({
        "ts": pd.to_datetime(h["time"], utc=True),
        "temp_c":    h.get("temperature_2m"),
        "precip_mm": h.get("precipitation"),
        "cloud_pct": h.get("cloud_cover"),
    })

def _load_weather_cached(country_key, lat, lon, start_date, end_date):
    """Per-country cache in weather_history.csv; fetch only what's missing."""
    cached = pd.read_csv(WEATHER_CACHE) if WEATHER_CACHE.exists() else pd.DataFrame()
    if len(cached):
        cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
    have = cached[cached.country == country_key] if "country" in cached.columns else pd.DataFrame()
    have_max = have.ts.max() if len(have) else None
    fetch_start = start_date if have_max is None else max(
        start_date, (have_max + pd.Timedelta(hours=1)).strftime("%Y-%m-%d"))
    if fetch_start > end_date:
        return have
    fresh = _fetch_weather_one(lat, lon, fetch_start, end_date)
    if not len(fresh):
        return have
    fresh["country"] = country_key
    all_ = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
    all_ = all_.drop_duplicates(subset=["country", "ts"], keep="last").sort_values(["country", "ts"])
    all_.to_csv(WEATHER_CACHE, index=False)
    return all_[all_.country == country_key]

# Group eligible cameras by the country-anchor coord they'd fetch for.
_country_keys = {}
for cam in ELIGIBLE:
    ll = camera_latlon(cam)
    if ll is None: continue
    _country_keys[cam] = f"{ll[0]:.4f},{ll[1]:.4f}"

_by_key = {}
_start_d = bins.ts.min().date().isoformat()
_end_d   = min(bins.ts.max().date(),
               pd.Timestamp.utcnow().date() - pd.Timedelta(days=1)).isoformat()
for cam, ck in _country_keys.items():
    if ck in _by_key: continue
    ll = camera_latlon(cam)
    try:
        _by_key[ck] = _load_weather_cached(ck, ll[0], ll[1], _start_d, _end_d)
    except Exception as e:
        print(f"weather fetch failed for {cam} ({ck}): {e}")
        _by_key[ck] = pd.DataFrame(columns=["ts","temp_c","precip_mm","cloud_pct"])

# Per-camera merge (resample to 15min via ffill, then join on cam_id+ts).
_weather_frames = []
for cam, ck in _country_keys.items():
    w = _by_key.get(ck)
    if w is None or not len(w): continue
    w15 = (w[["ts","temp_c","precip_mm","cloud_pct"]]
             .set_index("ts").resample(f"{BIN_MIN}min").ffill(limit=4).reset_index())
    w15["cam_id"] = cam
    _weather_frames.append(w15)
weather_bins = pd.concat(_weather_frames, ignore_index=True) if _weather_frames else \
    pd.DataFrame(columns=["cam_id","ts","temp_c","precip_mm","cloud_pct"])

bins = bins.merge(weather_bins, on=["cam_id","ts"], how="left")
print(f"weather merged: {bins.temp_c.notna().sum()} / {len(bins)} bins have temperature")

# Holidays (local date) + weekend.
if _HOLIDAYS_OK:
    _hol_cache = {}
    def _is_holiday(cam, local_ts):
        iso = camera_country_iso2(cam)
        if not iso: return 0
        if iso not in _hol_cache:
            try: _hol_cache[iso] = _hol.country_holidays(iso)
            except Exception: _hol_cache[iso] = set()
        return int(local_ts.date() in _hol_cache[iso])
    bins["is_holiday"] = [_is_holiday(c, t) for c, t in zip(bins.cam_id, bins.local)]
else:
    bins["is_holiday"] = 0
bins["is_weekend"] = (bins.dow >= 5).astype(int)
print(f"holidays flagged: {bins.is_holiday.sum()} bins   weekends: {bins.is_weekend.sum()} bins")

# Preview a slice.
_prev_cam = ELIGIBLE[0] if ELIGIBLE else None
if _prev_cam:
    print(f"\nSample rows for {_prev_cam}:")
    print(bins[bins.cam_id == _prev_cam][["ts","person","temp_c","precip_mm",
                                           "is_holiday","is_weekend"]].tail(8).to_string(index=False))
```

<a id="cell-46"></a>
<div dir="rtl">

### תא 46 - [code] - 11c: baseline ladder + Ridge

**מה עושה:** תא ארוך (5736 תווים) שהוא לב פרק החיזוי. מגדיר `HORIZONS = [1, 4, 12, 48]` (‏15דק / 1ש / 3ש / 12ש קדימה) ו-`LAM = 3.0` (ridge penalty). מסנן `work = bins[bins.cam_id.isin(ELIGIBLE)]`. מוסיף `y` שגישור פערים מיקרו (‏`ffill(limit=2)` = עד 30 דקות). מחלק train/test על `t_split = min + (max - min) * 0.75` (‏25% אחרונים = test). מדפיס הודעה על גודל חלון ה-test בשעות.

`profile_of(train, span_days)` מחזיר פרופיל לפי hour-of-week (‏`how`) אם ההיסטוריה >= 7 ימים, אחרת לפי hour-of-day (`hod`). ‏`evaluate(df_cam, yhat_col)` מחזיר mean absolute error על ה-test.

לולאה על כל horizon: לכל מצלמה בונה DataFrame `f` עם `y, hod, how, dow, target=y.shift(-h), persistence=y, seasnaive24=y.shift(96-h), lag1..lag4, lag96, roll4, roll12, t_hod, t_how`. מאחד ל-`F = concat(frames)`. מחשב `profile` על ה-train ומטמיע את הפרופיל כעמודה (`prof.get((cam, slot), NaN)`). מחשב ridge: פיצ'רים הם `[lag1..lag4, lag96, roll4, roll12, _exo (temp_c, precip_mm, cloud_pct, is_holiday, is_weekend if merged), sin_h, cos_h, one-hot של camera]`. מרג'צ NaN של פיצ'רים (per-camera median). ‏Ridge closed-form: `w = solve(X^T X + LAM*I, X^T y)`. מנבא `Fm['ridge']`. שומר `F` ב-`per_h_frames[h]` (כדי שתא 48 יוכל להוסיף את התחזית של GRU לאותו DataFrame).

לולאה סופית: לכל מודל (`persistence, seasnaive24, profile, ridge`) x כל horizon - MAE. בונה `verdict` = pivot של mae לפי model x horizon. מסתיר עמודות שכולן NaN (חלון test עדיין קצר יותר מ-horizon). ממיין לפי `1h` אם קיים אחרת לפי 15min. מחשב skill vs seasonal-naive: `skill = 1 - verdict / seasnaive24` (חיובי = יותר טוב). מדפיס.

**למה:** הבחירה ב-Ridge closed-form (`np.linalg.solve`) ולא sklearn היא בכוונה: מקל על התלויות ומראה בגלוי איך המודל עובד. `LAM = 3.0` הוא regularization בינוני; חלש מדי היה overfitting על השבועיים של דאטה, חזק מדי היה מבטל את כוח המשקולות של הפיצ'רים. `pd.get_dummies(cam_id)` נותן למודל הבנה של יחסים בין מצלמות.

**אלטרנטיבות:** אפשר היה RandomForest או XGBoost, אבל בשביל CSV של 5000-10000 שורות עם 10 פיצ'רים, Ridge הוא sweet spot: מהיר, פרשני, קל להעביר ל-VM. אפשר היה גם `train_test_split(random=True)`, אבל time-series חייבת time-based split.

**פלטים:** הודעה `train < 2026-08-10 12:00 <= test (72h test window)`. אז DataFrame `verdict`:
```
             15min   1h     3h    12h
model
persistence  1.234   3.456  6.789 ...
seasnaive24  2.345   4.567  5.234 ...
profile      2.111   4.222  5.111 ...
ridge        1.056   2.987  4.888 ...
```
ואז `skill` DataFrame עם ערכים כמו 0.35 (35% שיפור).

</div>

```python
# 11c. Baseline ladder + closed-form ridge - who beats seasonal-naive?
#
# Honesty framework: every fancy model must beat "the same time yesterday"
# (seasonal-naive) to justify existing. The test window is the last 25% of
# the timeline, never touched during fitting. With a short cache this is a
# single holdout; as the CSV grows past ~2 weeks it automatically becomes
# a proper multi-day test.
HORIZONS = [1, 4, 12, 48]                     # 15min / 1h / 3h / 12h ahead
LAM = 3.0                                     # ridge penalty

work = bins[bins.cam_id.isin(ELIGIBLE)].copy()
# Models may bridge camera micro-gaps (<= 30 min); longer gaps stay NaN.
work['y'] = work.groupby('cam_id')['person'].transform(
    lambda s: s.ffill(limit=2))
t_split = work.ts.min() + (work.ts.max() - work.ts.min()) * 0.75
print(f'train < {t_split:%Y-%m-%d %H:%M} <= test   '
      f'({(work.ts.max() - t_split).total_seconds() / 3600:.0f}h test window)')

def profile_of(train, span_days):
    """Median count per local time slot. hour-of-week when history covers
    a week (weekday rhythm != weekend), hour-of-day before that."""
    key = 'how' if span_days >= 7 else 'hod'
    prof = train.groupby(['cam_id', key]).y.median()
    return key, prof

def evaluate(df_cam, yhat_col):
    m = df_cam.ts > t_split
    err = (df_cam.loc[m, yhat_col] - df_cam.loc[m, 'target']).abs()
    return err.mean()

span_days_all = (work.ts.max() - work.ts.min()).total_seconds() / 86400
rows = []
per_h_frames = {}
for h in HORIZONS:
    frames = []
    for cam, dfc in work.groupby('cam_id'):
        dfc = dfc.sort_values('ts').reset_index(drop=True)
        f = pd.DataFrame({'ts': dfc.ts, 'cam_id': cam, 'y': dfc.y,
                          'hod': dfc.hod, 'how': dfc.how, 'dow': dfc.dow})
        f['target'] = dfc.y.shift(-h)                       # y at t+h
        f['persistence'] = dfc.y
        f['seasnaive24'] = dfc.y.shift(BINS_PER_DAY - h)    # y at t+h-24h
        for lag in (1, 2, 3, 4):
            f[f'lag{lag}'] = dfc.y.shift(lag)
        f['lag96'] = dfc.y.shift(BINS_PER_DAY)
        f['roll4'] = dfc.y.rolling(4).mean()
        f['roll12'] = dfc.y.rolling(12).mean()
        # target's local slot (for the profile forecast at t+h)
        f['t_hod'] = dfc.hod.shift(-h)
        f['t_how'] = dfc.how.shift(-h)
        frames.append(f)
    F = pd.concat(frames, ignore_index=True)

    key, prof = profile_of(F[F.ts <= t_split], span_days_all)
    tkey = 't_hod' if key == 'hod' else 't_how'
    F['profile'] = [prof.get((c, s), np.nan)
                    for c, s in zip(F.cam_id, F[tkey])]

    feat_cols = ['lag1', 'lag2', 'lag3', 'lag4', 'lag96', 'roll4', 'roll12']
    # 11b-x additions: exogenous features - keep those actually merged in
    _exo = [c for c in ('temp_c', 'precip_mm', 'cloud_pct', 'is_holiday', 'is_weekend')
            if c in F.columns and F[c].notna().any()]
    feat_cols = feat_cols + _exo
    # Fill exogenous NaN with per-camera median so a missing weather row
    # never kills the whole row's usability in the ridge design matrix.
    if _exo:
        F[_exo] = F.groupby('cam_id')[_exo].transform(lambda s: s.fillna(s.median()))
    Fm = F.dropna(subset=feat_cols + ['target', 't_hod']).copy()
    Fm['sin_h'] = np.sin(2 * np.pi * Fm.t_hod / BINS_PER_DAY)
    Fm['cos_h'] = np.cos(2 * np.pi * Fm.t_hod / BINS_PER_DAY)
    cams_oh = pd.get_dummies(Fm.cam_id, dtype=float)
    X = np.column_stack([Fm[feat_cols + ['sin_h', 'cos_h']].to_numpy(float),
                         cams_oh.to_numpy()])
    y = Fm.target.to_numpy(float)
    tr = (Fm.ts <= t_split).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0).clip(1e-6)
    Xs = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    w = np.linalg.solve(Xs[tr].T @ Xs[tr] + LAM * np.eye(Xs.shape[1]),
                        Xs[tr].T @ y[tr])
    Fm['ridge'] = Xs @ w
    F = F.merge(Fm[['ts', 'cam_id', 'ridge']], on=['ts', 'cam_id'], how='left')

    per_h_frames[h] = F
    for model in ('persistence', 'seasnaive24', 'profile', 'ridge'):
        mae = evaluate(F.dropna(subset=['target', model]), model)
        rows.append({'model': model, 'h': h, 'mae': mae})

verdict = pd.DataFrame(rows).pivot(index='model', columns='h', values='mae')
verdict.columns = [f'{h * BIN_MIN}min' for h in HORIZONS]
# A horizon column goes all-NaN while the test window is still shorter
# than that horizon (no target has an answer yet) - hide it with a note
# instead of printing a wall of NaN; it fills in as the cache grows.
_dead = [c for c in verdict.columns if verdict[c].isna().all()]
if _dead:
    print(f'horizons not yet scoreable (test window too short): '
          f'{", ".join(_dead)} - they appear as the cache grows')
    verdict = verdict.drop(columns=_dead)
verdict = verdict.round(3)
# Sort key: prefer the 1h column, but fall back to whatever is left when
# only the 15-min horizon survives (the old `verdict.columns[1]` blew up
# on a cache that only produced one scoreable column).
if not verdict.columns.empty:
    _sort_col = verdict.columns[1] if len(verdict.columns) > 1 \
        else verdict.columns[0]
    verdict = verdict.sort_values(_sort_col)
# Skill vs seasonal-naive: a zero-MAE seasonal-naive baseline (perfectly
# stable stream) would make the ratio blow up to inf, so we mask zeros
# to NaN and print an honest "n/a" instead of misleading infinities.
if 'seasnaive24' in verdict.index and not verdict.columns.empty:
    _baseline = verdict.loc['seasnaive24'].replace(0, np.nan)
    skill = (1 - verdict / _baseline).round(2)
else:
    skill = pd.DataFrame(index=verdict.index, columns=verdict.columns)
print('\nMAE (people, per 15-min bin):'); print(verdict)
print('\nskill vs seasonal-naive (positive = better):'); print(skill)
```

<a id="cell-47"></a>
<div dir="rtl">

### תא 47 - [code] - 11c-b: Chronos-2 zero-shot

**מה עושה:** תא ארוך (3439 תווים) שמנסה מודל foundation של Amazon לסדרות זמן. `try/except ImportError` על `from chronos import ChronosPipeline`: אם `chronos-forecasting` לא מותקן, מדפיס הוראה `pip install chronos-forecasting` ומדלג על התא. אם קיים: `_chronos = ChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu', torch_dtype=torch.float32)` (הורדה של 250MB בפעם הראשונה). לולאה על HORIZONS: לכל מצלמה, בונה `context = train values.ffill()` כ-numpy. אם `len(context) < BINS_PER_DAY` (‏96 = פחות מיום היסטוריה) - skip. ‏מריץ `_chronos.predict(context=torch.tensor(context), prediction_length=h, num_samples=20)` ולוקח את ה-median של הדגימות (`fc.median(dim=0).values.numpy().flatten()`). מצמיד את h התחזיות הראשונות לשורות ה-test הראשונות של המצלמה. חישוב MAE ל-chronos פר horizon; דחיסה של `_vc = pd.DataFrame(chronos_rows).pivot(...)` וצירוף ל-`verdict` הקיים. מחדש את `skill` DataFrame.

**למה:** Chronos-2 הוא מודל transformer שאומן על מיליארדי סדרות זמן שאינן קשורות לפוטפול. הבדיקה כאן היא honest: האם מודל general-purpose מנצח את הסימפלים? אם כן - זהו bar חדש ל-VM twin (11c-VM) להתייחס אליו; אם לא - יש הוכחה קשה שהפרופיל+lags נושאים את רוב הסיגנל בדאטה הזה, ואין הצדקה למודל מורכב יותר. גם מדינה זו מייצגת יחסי כוחות: mind-scale models (‏Chronos) מול classic models (‏Ridge) - השאלה של "האם transformer זעיר מרוויח את מקומו".

**אלטרנטיבות:** מודלים נוספים בקטגוריה - TimesFM (Google), Moirai (Salesforce), Lag-Llama - כולם דומים. Chronos נבחר כי הוא חופשי, זמין ב-HuggingFace, ורץ ב-CPU בפחות מדקה למצלמה.

**פלטים:** בפעם הראשונה: `loading chronos-2 (first run downloads ~250 MB)...`. אחר-כך MAE `verdict` מעודכן עם שורת chronos בנוסף, ו-`skill` מעודכן.

</div>

```python
# 11c-b. Chronos-2 - Amazon's time-series foundation model, zero-shot.
#
# The baseline ladder above proves what SIMPLE models can do; this cell
# asks whether a general-purpose transformer trained on billions of
# unrelated time series beats them on our small footfall CSV WITHOUT any
# fitting on our data. If it does, that becomes the new bar the on-VM
# operational forecaster (twin notebook 11c-VM) has to justify itself
# against; if it doesn't, we have hard evidence the profile+lags carry
# most of the signal on this dataset and no more model complexity is
# warranted.
#
# Chronos-2 downloads ~250 MB of weights on first run and needs torch (we
# already have it - YOLO runs on it). CPU inference: a few seconds per
# camera per horizon. Fully skips if the package isn't installed - do
# `pip install chronos-forecasting` in the notebook's env to enable.
try:
    from chronos import ChronosPipeline
    import torch
    _CHRONOS_OK = True
except ImportError:
    print("chronos not installed - pip install chronos-forecasting  (skipping this cell)")
    _CHRONOS_OK = False

if _CHRONOS_OK:
    print("loading chronos-2 (first run downloads ~250 MB)...")
    _chronos = ChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    chronos_rows = []
    for h in HORIZONS:
        F = per_h_frames[h].copy()
        F["chronos"] = np.nan
        for cam, dfc in F.groupby("cam_id"):
            dfc = dfc.sort_values("ts").reset_index(drop=True)
            train_mask = dfc.ts <= t_split
            context = dfc.loc[train_mask, "y"].ffill().to_numpy(dtype=np.float32)
            if len(context) < BINS_PER_DAY:
                continue                             # need at least 1 day of history
            try:
                ctx_t = torch.tensor(context)
                fc = _chronos.predict(context=ctx_t, prediction_length=h, num_samples=20)
                # fc: [num_samples, prediction_length]; take the median forecast.
                yhat = fc.median(dim=0).values.numpy().flatten()
            except Exception as e:
                print(f"chronos failed for {cam} h={h}: {type(e).__name__}: {e}")
                continue
            # Attach the first h forecasts to the first h test rows for this cam.
            test_idx = dfc.index[(~train_mask)][:h]
            if len(test_idx) == 0:
                continue
            F.loc[test_idx, "chronos"] = yhat[:len(test_idx)]

        mae = evaluate(F.dropna(subset=["target", "chronos"]), "chronos") \
              if F["chronos"].notna().any() else np.nan
        chronos_rows.append({"model": "chronos", "h": h, "mae": mae})
        per_h_frames[h]["chronos"] = F["chronos"]

    if chronos_rows:
        _vc = pd.DataFrame(chronos_rows).pivot(index="model", columns="h", values="mae")
        _vc.columns = [f"{h * BIN_MIN}min" for h in HORIZONS]
        # Drop horizons the earlier verdict already hid as too-short-test.
        _vc = _vc[[c for c in _vc.columns if c in verdict.columns]].round(3)
        verdict = pd.concat([verdict, _vc])
        if "seasnaive24" in verdict.index and not verdict.columns.empty:
            _baseline = verdict.loc["seasnaive24"].replace(0, np.nan)
            skill = (1 - verdict / _baseline).round(2)
        print("\nMAE with Chronos-2 (people, per 15-min bin):"); print(verdict)
        print("\nskill vs seasonal-naive:"); print(skill)
```

<a id="cell-48"></a>
<div dir="rtl">

### תא 48 - [code] - 11d: GRU forecaster

**מה עושה:** התא הכי גדול בפרק ה-forecasting (5341 תווים). מגדיר `INPUT_BINS = 96` (‏24h) אם ה-`span_days_all >= 4` אחרת `48` (‏12h). ‏`OUT_BINS = 48` (‏12h forecast), `STRIDE = 4` (חלון חדש כל שעה). בונה חלונות: לכל מצלמה, מחשב `scale = 95th percentile of y in train` (‏clip ל-1 לפחות), ואז features = 5 עמודות: `y/scale, vehicles/scale, sin(2pi*hod/96), cos(2pi*hod/96), one-hot של camera`. לכל stride: `xin = feats[s:s+96]`, `tgt = y[s+96:s+96+48]/scale`. אם >50% מהחלון הוא NaN או כל target הוא NaN - skip. ממיר ל-tensor. חלוקה: `t_split` מפריד train/test, ומהtrain 1/6 האחרונים = val (early stopping). מדפיס גדלים.

`GRU_TRAINED = len(tr_idx) >= 8 and len(val_idx) >= 1` - אם קטן מדי, מדלגים.

`GRUForecast` = מודול קטן: `nn.GRU(n_feat, hidden=32, batch_first=True) -> nn.Linear(32, 48)`. הגודל `hidden=32` הוא כ-15K פרמטרים - הדאטה הוא אלפי חלונות, לא מיליונים; יותר גדול = memorization.

`masked_mse(pred, tgt)` מתעלם NaN בתוך target. אימון: 80 epochs, batch 128, Adam, lr 1e-3, early stopping עם patience=10. שמירת `best_state` והחזרת המצב הטוב. פלוט loss curve (train + val).

בסוף: אם `GRU_TRAINED`, מריץ predict על test, מנרמל בחזרה (`P * scales`), ומחשב MAE לכל horizon (‏only if h <= OUT_BINS=48). מוסיף שורת `gru` ל-`verdict`. מדפיס `final ladder` ממוין + `champion at 1h: {model}`.

**למה:** ההגבלה של `hidden=32` היא לקח מ-overfitting: על CSV קטן, מודל גדול יזכור טרסה. GRU מעל LSTM כי פחות פרמטרים ולפעמים ביצועים דומים. הבחירה של `masked_mse` (במקום להשליך שורות NaN מתוך התיקה) שומרת על גודל batch עקבי - חשוב בטרנינג ל-CPU.

**אלטרנטיבות:** אפשר היה `Transformer`, אבל ליום היסטוריה של שבועיים = overkill; ‏GRU של 15K פרמטרים מהיר יותר ומספיק. אפשר היה `nn.LSTM`, אבל GRU עם פחות פרמטרים באותה איכות. `n-BEATS` היה יותר מוכן ל-multi-horizon אבל דורש ספרייה נוספת.

**פלטים:** `windows: train 245, val 41, test 62`. אם עודף: `early stop @ epoch 34 (val 0.0234)`. פלוט של curve. `final ladder (MAE, people): ...`. `champion at 1h: ridge` (או `gru` אם ניצח).

</div>

```python
# 11d. GRU - a small recurrent forecaster (torch is already here: YOLO runs on it).
#
# The GRU reads the last 24h (96 bins x [count, vehicles, time-of-day
# encoding, camera id]) and emits the next 12h (48 bins) in one shot. Tiny
# on purpose: hidden=32 is ~15k parameters - the dataset is thousands of
# windows, not millions, and a bigger net would just memorize it. Trains
# on CPU in well under a minute; early stopping on a time-ordered
# validation slice guards against overfitting the short history.
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
# Short cache = shorter input window, so training windows exist from day
# one; once the cache passes ~4 days the window grows to a full 24h.
INPUT_BINS = BINS_PER_DAY if span_days_all >= 4 else BINS_PER_DAY // 2
OUT_BINS, STRIDE = 48, 4

cams_idx = {c: i for i, c in enumerate(ELIGIBLE)}
Xw, Yw, Wend = [], [], []
for cam, dfc in work.groupby('cam_id'):
    dfc = dfc.sort_values('ts').reset_index(drop=True)
    yv = dfc.y.to_numpy(float)
    scale = max(np.nanpercentile(yv[dfc.ts <= t_split], 95), 1.0)
    feats = np.column_stack([
        np.nan_to_num(yv / scale, nan=0.0),
        np.nan_to_num(dfc.vehicles.to_numpy(float) / scale, nan=0.0),
        np.sin(2 * np.pi * dfc.hod / BINS_PER_DAY),
        np.cos(2 * np.pi * dfc.hod / BINS_PER_DAY),
        np.repeat([np.eye(len(ELIGIBLE))[cams_idx[cam]]], len(dfc), 0)
        .reshape(len(dfc), -1)[:, 0:len(ELIGIBLE)],
    ])
    for s in range(0, len(dfc) - INPUT_BINS - OUT_BINS, STRIDE):
        xin = feats[s:s + INPUT_BINS]
        tgt = yv[s + INPUT_BINS:s + INPUT_BINS + OUT_BINS] / scale
        if np.isnan(yv[s:s + INPUT_BINS]).mean() > 0.5 or np.isnan(tgt).all():
            continue                        # window mostly unobserved: skip
        Xw.append(xin); Yw.append(tgt)
        Wend.append((dfc.ts.iloc[s + INPUT_BINS], cam, scale))
Xw = torch.tensor(np.array(Xw), dtype=torch.float32)
Yw = torch.tensor(np.array(Yw), dtype=torch.float32)
t_end = pd.Series([t for t, _, _ in Wend])
test_m = (t_end > t_split).to_numpy()
tr_idx = np.where(~test_m)[0]
val_n = max(1, len(tr_idx) // 6)
val_idx, tr_idx = tr_idx[-val_n:], tr_idx[:-val_n]
print(f'windows: train {len(tr_idx)}, val {len(val_idx)}, test {test_m.sum()}'
      + ('   (thin history - treat as a demo until the cache grows)'
         if len(tr_idx) < 500 else ''))
# Not enough windows to fit anything honest? Skip cleanly - the GRU joins
# the ladder automatically once the cache holds a few dense days.
GRU_TRAINED = len(tr_idx) >= 8 and len(val_idx) >= 1
if not GRU_TRAINED:
    print('too few training windows - skipping the GRU this run; '
          're-run after the cache grows a couple of days.')

class GRUForecast(nn.Module):
    def __init__(self, n_feat, hidden=32, out=OUT_BINS):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out)
    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn[-1])

def masked_mse(pred, tgt):
    m = ~torch.isnan(tgt)
    return ((pred[m] - tgt[m]) ** 2).mean()

net = GRUForecast(Xw.shape[-1])
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
best, best_state, patience, curve = np.inf, None, 0, []
for epoch in range(80 if GRU_TRAINED else 0):
    net.train()
    perm = np.random.permutation(tr_idx)
    tot = 0.0
    for i in range(0, len(perm), 128):
        b = perm[i:i + 128]
        opt.zero_grad()
        loss = masked_mse(net(Xw[b]), Yw[b])
        loss.backward(); opt.step()
        tot += float(loss) * len(b)
    net.eval()
    with torch.no_grad():
        vl = float(masked_mse(net(Xw[val_idx]), Yw[val_idx]))
    curve.append((tot / len(perm), vl))
    if vl < best - 1e-5:
        best, best_state, patience = vl, net.state_dict(), 0
    else:
        patience += 1
        if patience >= 10:
            print(f'early stop @ epoch {epoch} (val {best:.4f})'); break
if best_state is not None:
    net.load_state_dict(best_state)

if curve:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([c[0] for c in curve], label='train')
    ax.plot([c[1] for c in curve], label='val')
    ax.set_title('GRU loss (masked MSE, normalized)')
    ax.set_xlabel('epoch'); ax.legend()
    plt.tight_layout(); plt.show()

# Test MAE at the ladder's horizons, de-normalized back to people-counts.
if GRU_TRAINED and test_m.sum():
    net.eval()
    with torch.no_grad():
        P = net(Xw[test_m]).numpy()
    T = Yw[test_m].numpy()
    scales = np.array([s for (_, _, s), m in zip(Wend, test_m) if m])
    gru_row = {}
    for h in HORIZONS:
        if h <= OUT_BINS:
            err = np.abs(P[:, h - 1] - T[:, h - 1]) * scales
            gru_row[f'{h * BIN_MIN}min'] = np.nanmean(err)
    verdict.loc['gru'] = pd.Series(gru_row).round(3)
# Same guard as cell 46: fall back to the 15-min column when the test
# window is still too short for a 1h horizon to have scored.
if not verdict.columns.empty:
    _sort_col = verdict.columns[1] if len(verdict.columns) > 1 \
        else verdict.columns[0]
    print('\nfinal ladder (MAE, people):')
    print(verdict.sort_values(_sort_col))
    champion = verdict.sort_values(_sort_col).index[0]
    print(f'\nchampion at {_sort_col}: {champion}')
else:
    print('\nno horizons scoreable yet - the cache is too short')
```

<a id="cell-49"></a>
<div dir="rtl">

### תא 49 - [markdown] - כיצד הדשבורד עובד

**מה עושה:** תא הסיום של המחברת (1854 תווים). תא markdown שהוא בעצם תיעוד של הדשבורד שהוצג בתא 33: איך הוא מורכב, מה כל טאב עושה, איך מוטמעות ‏המצלמות שלך. מסביר שהדשבורד הוא HTML self-contained (`src/web/index.html`) שמוגש דרך `app.dashboard_server`. הוא מרנדר את המצלמות שבחרת דרך `data/web/local_grid.json` שנכתב מ-`SELECTED_CAMS`. הכל מקומי - שום דבר לא מועלה.

**טאבים ב-main-mode** (`?mode=main`): (1) **Analysis** - הגריד החי 2x2 + KPIs, פלוס טבלאות של anomaly / operational-events / heatmap למטה. מספרי 24h מגיעים מ-Firestore (הקולקטור על ה-VM כותב שם ללא הפסק). (2) **Search** - חיפוש דמיון תמונות + browse לפי class/time של pools של review + live-samples של הקולקטור (‏מגובה `POST /api/search`). (3) **Snapshots** (‏main בלבד) - בכל לחיצה על "Snapshot grid" בכותרת, המחברת בונה PNG 2x2 של 4 ה-tiles ושומרת ב-`src/web/snapshots/user/YYYYMMDD_HHMMSS.png`. הטאב מציג כל PNG כ-thumbnail (click = פתח/הורד; אייקון פח = מחק אחד; Clear all = מחק הכל).

**לא נראים ב-main mode**: Reinforcement-learning tab (דורש רשומות review מתמידות - main הוא ephemeral), Model view - live strip (hardcoded ל-Turkey cams של ה-VM - לא רלוונטי כשבוחרים מדינה אחרת), Window analysis (מפעיל בלבד).

**פורט**: תא הפתיחה עושה auto-scan 8000-8020. אם הכל תפוס - הודעה ברורה.

**Restart**: reload של הדף (‏Ctrl+F5) אחרי שינוי של קובץ `src/web/*.html` או `.js`. `?v=NN` cache-buster ב-index.html מקודם בכל שינוי.

**למה:** תיעוד קונקרטי של אילו טאבים המשתמש רואה ואילו מוסתרים בכוונה מסביר למי שרואה את הדשבורד למה יש הבדל בין הצילומים במסמכים שונים - main נראה שונה מ-VM twin.

**פלטים:** טקסט בלבד. תא הפאונל של המחברת.

</div>

```markdown
## How this dashboard works

The dashboard below is a self-contained HTML page (`src/web/index.html`)
served by a tiny local HTTP server (`app.dashboard_server`). It renders
**your picked cameras** (the 4 you chose in the picker) using
`data/web/local_grid.json`, which the previous cell wrote from
`SELECTED_CAMS`. Nothing here is uploaded anywhere.

**Tabs (main-mode)** — the URL is `?mode=main`, which hides panels that
apply only to the VM-mirror twin notebook:

- **Analysis** — the 2x2 live-video grid + KPIs, plus the anomaly /
  operational-events / heatmap tables further down. The 24h numbers come
  from Firestore (the collector on the GCP VM writes there continuously).
- **Search** — image-similarity + class/time browse over the collector's
  review + live-samples pools (backed by `POST /api/search`).
- **Snapshots** *(main only)* — every time you hit the **📸 Snapshot
  grid** button in the header, this notebook composes a 2x2 PNG of the
  four Analysis tiles and saves it under
  `src/web/snapshots/user/YYYYMMDD_HHMMSS.png`. This tab lists every
  PNG as a thumbnail (click → open/download; 🗑 → delete one; Clear all
  → wipe the folder).

**Not visible in main mode**: Reinforcement-learning tab (needs
persistent review data — main is ephemeral, each run is a fresh pick),
Model view - live strip (hardcoded to the VM's Turkey cams — irrelevant
when you pick a different country), and Window analysis (operator-only
one-shot).

**Port** — the launch cell auto-scans **8000-8020** and binds the first
free port, so opening the notebook a second time on the same machine
doesn't override an existing dashboard. If everything is busy you'll get
a clear error message.

**Restart** — reload the page (Ctrl+F5) after changing any
`src/web/*.html` or `.js` file. The cache-buster `?v=NN` in index.html
is bumped whenever a file changes.
```
