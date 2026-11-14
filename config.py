import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables if .env file exists
env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = BASE_DIR / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "virtual-judge-secret-key-fast-2026-fall")
    
    # Check MySQL settings first, or fallback to SQLite for seamless local execution
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "verdictbox")

    if os.getenv("DATABASE_URL"):
        SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    elif db_user and db_pass:
        SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        # Default zero-friction offline SQLite fallback
        sqlite_db = BASE_DIR / "instance" / "virtual_judge.db"
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{sqlite_db}"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PROPAGATE_EXCEPTIONS = True

