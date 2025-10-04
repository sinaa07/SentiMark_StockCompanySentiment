import os

def get_project_root() -> str:
    """Return the absolute path to the backend directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_path(relative_path: str) -> str:
    """
    Resolve a relative path (from backend root) to an absolute path.
    Example: resolve_path("data/nse_stocks.db") →
    /Users/.../stock-sentiment-dashboard/backend/data/nse_stocks.db
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
