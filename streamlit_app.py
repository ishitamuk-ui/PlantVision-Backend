"""Streamlit Community Cloud entry point alias."""
import runpy

if __name__ == "__main__":
    runpy.run_path("app.py", run_name="__main__")
