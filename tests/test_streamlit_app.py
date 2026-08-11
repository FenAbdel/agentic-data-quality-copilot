import importlib.util
from pathlib import Path


def test_streamlit_app_file_exists():
    app_path = Path("app/streamlit_app.py")

    assert app_path.exists()


def test_streamlit_app_can_be_loaded():
    app_path = Path("app/streamlit_app.py")

    spec = importlib.util.spec_from_file_location(
        "streamlit_app",
        app_path,
    )

    assert spec is not None
    assert spec.loader is not None