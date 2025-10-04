## Stock Company Sentiment Analysis
---

## 📂 Project Structure & Setup Guide

### 🧭 **Folder Structure**

```
stock-sentiment-dashboard/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── pipeline.py
│   │   │   ├── user_input_processor.py
│   │   │   └── ... (orchestration logic, LLM/search modules, etc.)
│   │   ├── services/
│   │   │   ├── db_manager.py
│   │   │   └── db_setup.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   ├── data/
│   │   ├── nse_master.csv
│   │   └── nse_stocks.db
│   │
│   ├── __init__.py
│   └── (venv, logs, config files, etc. if needed)
│
├── frontend/
│   └── ... (React / Inertia / other frontend code)
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

### 📌 **Path & Import Conventions**

* All commands are executed **from inside the `backend/` folder**.
  Always use the module form, **not** direct file execution:

  ```bash
  cd backend
  python -m app.core.pipeline
  python -m app.services.db_setup
  ```

* All data files (CSV, SQLite DB) live inside:

  ```
  backend/data/
  ```

  This keeps file resolution consistent across all modules.

* The `resolve_path()` helper in `app/utils.py` anchors at the `backend` directory:

  ```python
  def get_project_root():
      return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  ```

* **Never** use paths like `backend/data/...` inside the backend.
  Just use `data/...` or rely on the default when prompted — since the resolver already points to `backend`.

---

### 🧰 **Initial Setup**

1. **Install dependencies**

   ```bash
   cd backend
   pip install -r ../requirements.txt
   ```

   *(or use your preferred virtual environment / poetry / pipenv)*

2. **Prepare the database**

   ```bash
   python -m app.services.db_setup
   ```

   * Press **Enter** for the default CSV path (`data/nse_master.csv`) if your CSV is already in `backend/data`.
   * This step will create and populate `backend/data/nse_stocks.db`.

3. **Run the pipeline**

   ```bash
   python -m app.core.pipeline
   ```

4. **(Optional)** Test the user input processor directly

   ```bash
   python -m app.core.user_input_processor
   ```

---




