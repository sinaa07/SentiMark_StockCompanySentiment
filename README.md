## SentiMark - Stock Company Sentiment Analysis 💲📈
[Dashboard Screenshot](./docs/dashboard.png)
End-to-end dashboard that pairs a FastAPI backend, FinBERT sentiment pipeline, and a Next.js frontend to surface real-time financial news sentiment for NSE-listed companies.

## Tech Stack ⚙️

- FastAPI + Uvicorn backend (Python 3.11)
- Local FinBERT inference (PyTorch, Hugging Face Transformers)
- Next.js 15 UI with Tailwind CSS
- SQLite for stock master data and news caching


## Repository Layout 📁

```
stock-sentiment-dashboard/
├── backend/
│   ├── app/                 # FastAPI application + sentiment pipeline
│   ├── config/              # Runtime configuration stubs
│   ├── data/                # SQLite DBs, NSE master CSV, cached news
│   ├── finbert_training/    # Utilities to fine-tune/evaluate FinBERT
│   └── requirements.txt
├── frontend/                # Next.js client (App Router)
├── docs/                    # Design notes & planning docs
└── README.md
```

---

## Prerequisites

- Python **3.11** (recommended)
- `pip` ≥ 23.x (or Poetry/Pipenv if you prefer)
- Node.js **18.18+** (Next.js 15 compatible); Node 20/22 works well
- One JavaScript package manager (`pnpm` ≥ 9.x recommended, `npm` ≥ 10 also works)
- Git LFS (if you plan to version model artifacts)

Optional but useful:

- Google Gemini API key (for LLM-backed news search)
- Hugging Face account/token for private model downloads or dataset pushes

---

## Initial Setup

## Backend Setup (FastAPI + FinBERT)

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # macOS / Linux
   # .venv\Scripts\activate        # Windows PowerShell
   ```

2. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the repository root (next to this README):

   ```bash
   cat <<'EOF' > .env
   GEMINI_API_KEY=your_google_gemini_key
   HUGGINGFACE_TOKEN=your_optional_hf_token
   EOF
   ```

   - `GEMINI_API_KEY` is required for the LLM-powered web search fallback.
   - `HUGGINGFACE_TOKEN` is only needed if you access private models or push to Hugging Face Hub.

4. **Bootstrap the NSE stock database (first-time only)**

   ```bash
   cd backend
   python -m app.services.db_setup
   cd ..
   ```

   This reads `backend/data/nse_master.csv` and regenerates `backend/data/nse_stocks.db`.
   You can supply a different CSV path if you maintain your own master file.

5. **Start the FastAPI server**

   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Uvicorn will expose the API at `http://localhost:8000`. Interactive docs are available at `/docs`.

6. **(Optional) Run the pipeline as a CLI sanity check**

   In another terminal with the same virtual environment active:

   ```bash
   cd backend
   python -m app.core.pipeline
   ```

   The first invocation will download the FinBERT weights and can take a few minutes.

---

## Frontend Setup (Next.js Dashboard) 📊

1. **Install dependencies**

   ```bash
   cd frontend
   pnpm install              # or: npm install / yarn install
   ```

2. **Configure environment (optional)**

   The default API base is `http://localhost:8000` (see `frontend/app/api/*/route.ts`). If you need to point to a different backend, create `frontend/.env.local` and add:

   ```
   NEXT_PUBLIC_API_BASE_URL=http://your-backend-host:8000
   ```

   Then update the fetch calls or create a small helper to read from that env var.

3. **Start the development server**

   ```bash
   pnpm dev                  # defaults to http://localhost:3000
   ```

   The Next.js server proxies API calls under `/api/*` to the backend routes.

---

## FinBERT Fine-Tuning & Evaluation Suite

Utilities for generating datasets, fine-tuning, and evaluating FinBERT live in `backend/finbert_training/`.
Run these scripts from that directory (they share the backend virtual environment):

```bash
cd backend/finbert_training

# 1. Generate synthetic training / validation data
python dataset_preparation.py --out_dir data --synth_per_class 800 --seed 42

# 2. Fine-tune FinBERT on the generated data
python model_finetune.py \
  --train_csv data/synthetic_train.csv \
  --val_csv data/synthetic_val.csv \
  --output_dir models/finetuned_finbert \
  --epochs 3 --batch_size 16 --lr 2e-5 --fp16 false

# 3. Evaluate against a labelled test set (replace with your real test CSV)
python evaluation.py \
  --test_csv data/real_test.csv \
  --model_dir models/finetuned_finbert \
  --baseline_model yiyanghkust/finbert-tone \
  --report_path reports/metrics.json
```

Model artifacts can be pointed to by updating `backend/app/core/finbert_client.py` if you want the API to use your fine-tuned weights.

---

## Data & Storage Notes 📝

- `backend/data/nse_master.csv` holds the latest NSE master list; update it periodically and rerun `db_setup`.
- `backend/data/nse_stocks.db` is regenerated by the setup script; commit only if it is intended as seed data.
- `backend/data/news_cache.db` grows as the pipeline caches articles. Delete it to force fresh news collection.
- Logs live under `backend/data/logs/`.

---

### 📌 Copyright

© 2025 Stock Sentiment Dashboard. All Rights Reserved.



