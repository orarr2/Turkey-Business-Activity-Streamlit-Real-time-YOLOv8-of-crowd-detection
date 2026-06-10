"""Remove demo_bus/demo_zidane rows that were inserted by scripts/demo_model.py.
Those entries pollute the streamlit camera selector with non-camera entities."""
import sqlite3, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB = Path(__file__).resolve().parent.parent / "data" / "footfall.db"
conn = sqlite3.connect(str(DB))
deleted = conn.execute("DELETE FROM footfall WHERE cam_id LIKE 'demo_%'").rowcount
conn.commit()
print(f"deleted {deleted} demo rows")
for row in conn.execute("SELECT cam_id, COUNT(*) FROM footfall GROUP BY cam_id"):
    print(row)
conn.close()
