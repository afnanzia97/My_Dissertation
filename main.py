import argparse


def main():
    parser = argparse.ArgumentParser(description="Medical Diagnosis Support System")
    parser.add_argument(
        "--mode",
        choices=["api", "ingest", "frontend"],
        default="api",
        help="api: start the FastAPI server | ingest: build the knowledge base | frontend: launch Streamlit UI",
    )
    args = parser.parse_args()

    if args.mode == "api":
        import uvicorn
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

    elif args.mode == "ingest":
        from src.ingestion.pipeline import run_ingestion
        from src.retrieval.vector_store import build_index
        run_ingestion()
        build_index()

    elif args.mode == "frontend":
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])


if __name__ == "__main__":
    main()
