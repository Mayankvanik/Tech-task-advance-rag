cd docker && docker-compose up -d    # start Qdrant

uv run .\run_api.py  # FastAPI on :8000

uv run streamlit run streamlit_app/app.py # UI on :8501