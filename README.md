# BKK Traffic: พยากรณ์รถติดกรุงเทพฯ
---

## สิ่งที่ต้องมีก่อน

| เครื่องมือ | เวอร์ชัน | หมายเหตุ |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | 0.11 ขึ้นไป | ใช้จัดการ Python และ dependency ทั้งหมด |
| Git | ล่าสุด | |
| Python | 3.12 | ไม่ต้องติดตั้งเอง `uv` จะดาวน์โหลดให้ตามไฟล์ `.python-version` |


ติดตั้ง `uv` (ถ้ายังไม่มี):

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## การติดตั้ง

```powershell
# 1) clone repo
git clone <repo-url>
cd BKKTraffic_Project

# 2) สร้าง virtual env และติดตั้ง library ทั้งหมดตาม uv.lock (รวม dev tools)
uv sync

# 3) ลงทะเบียน Jupyter kernel
uv run python -m ipykernel install --user --name bkk-traffic --display-name "BKK Traffic (3.12)"

# 4) ตั้งค่า git hooks (ล้าง output ของ notebook และรัน ruff ก่อน commit)
uv run nbstripout --install
uv run pre-commit install

# 5) คัดลอกไฟล์ environment
copy .env.example .env      # macOS/Linux: cp .env.example .env
```

ตรวจว่าติดตั้งสำเร็จ:

```powershell
uv run python -c "import lightgbm, shap, polars, pandera; print('OK')"
```

> **ห้ามใช้ `pip install` ตรง ๆ** ถ้าต้องเพิ่ม library ให้ใช้ `uv add <ชื่อ>` (หรือ `uv add --dev <ชื่อ>` สำหรับเครื่องมือพัฒนา) แล้ว commit `pyproject.toml` กับ `uv.lock` ไปพร้อมกัน ทุกคนในทีมจะได้ environment ตรงกัน

---

## การรันใช้งาน

> คำสั่งทั้งหมดรันจาก root ของโปรเจกต์ และขึ้นต้นด้วย `uv run` ซึ่งจะเรียกใช้ virtual env ให้อัตโนมัติ ไม่ต้อง activate เอง

### EDA และ Notebook

```powershell
uv run jupyter lab
```

เมื่อเปิด notebook แล้วให้เลือก kernel **"BKK Traffic (3.12)"** และตั้งชื่อไฟล์แบบ `NN_ชื่อผู้ทำ_หัวข้อ.ipynb` เช่น `01_aphichet_traffy_eda.ipynb`


จากนั้นเปิด <http://localhost:5000> จะเห็น metrics (F1, recall เริ่มติดใหม่, MAE), parameters และ model artifact ของทุก run

### Dashboard (ตอนนี้ยังไม่มี)

```powershell
uv run streamlit run dashboard/app.py
```

จากนั้นเปิด <http://localhost:8501> จะเห็นแผนที่จุดเสี่ยงรถติด +1 ชม. เทียบกับค่าจริง และหน้าวิเคราะห์ปัจจัย

### ทดสอบและตรวจโค้ด

```powershell
uv run pytest                  # unit tests + leakage / as-of cutoff tests
uv run ruff check . --fix      # lint
uv run ruff format .           # จัดรูปแบบโค้ด
```

---

## โครงสร้างโปรเจกต์

```
BKKTraffic_Project/
├── configs/
│   └── config.yaml            # seed, วันที่ split, threshold, path ต่าง ๆ
├── data/                      # ไม่ commit ลง Git (ยกเว้น .gitkeep)
│   ├── raw/                   # ข้อมูลดิบ: read-only ห้ามแก้ไข
│   ├── interim/               # ข้อมูลระหว่างทาง (clean แล้ว)
│   ├── processed/             # feature table พร้อมเทรน
│   └── external/              # ปฏิทินวันหยุด ฯลฯ
├── src/bkk_traffic/
│   ├── ingest/                # loader ของแต่ละแหล่งข้อมูล
│   ├── validate/              # pandera schema / quality gate
│   ├── features/              # สร้าง feature (lag, ฝน, ปฏิทิน, ถนนข้างเคียง ...)
│   ├── models/                # baseline, LightGBM, SHAP
│   └── pipelines/             # Prefect flows ที่ร้อยทุกขั้นเข้าด้วยกัน
├── notebooks/                 # EDA และการทดลอง
├── dashboard/                 # Streamlit app
├── tests/                     # pytest (รวม leakage tests)
├── reports/figures/           # กราฟสำหรับรายงาน
├── docs/                      # data dictionary, model card
├── pyproject.toml             # รายการ dependency
└── uv.lock                    # lock เวอร์ชันของทุก library (ต้อง commit)
```

---

## แนวปฏิบัติในการทำงานร่วมกัน

- **Branch:** ห้าม push ตรงเข้า `main` ให้แตก branch ตามงาน เช่น `de/itic-ingest`, `ds/lightgbm-baseline`, `mle/dashboard` แล้วเปิด Pull Request
- **ข้อมูล:** ห้าม commit ไฟล์ข้อมูลดิบ, `.env` หรือ `mlruns/` (`.gitignore` กันไว้แล้ว)
- **Reproducibility:** ใช้ seed จาก `configs/config.yaml` เท่านั้น และให้ทุก run ของการเทรน log ลง MLflow
- **Leakage:** feature ทุกตัวต้องรู้ค่าได้ ณ เวลา t ส่วนค่า free-flow, historical average และ feature จาก Traffy ให้คำนวณจากช่วง train เท่านั้น ถ้าเพิ่ม feature ใหม่ต้องเขียน as-of test ใน `tests/` ด้วย
- **Notebook:** โค้ดที่ใช้ซ้ำให้ย้ายไปไว้ใน `src/bkk_traffic/` แล้ว import กลับมาใช้ใน notebook
