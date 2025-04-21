from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DB_URL = f"mysql+pymysql://root:root@127.0.0.1:3307/recipe_db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)