from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from sqlalchemy import text

DB_URL= "sqlite:///./books.db"

engine= create_engine(DB_URL)
SessionLocal= sessionmaker(autocommit= False, autoflush= False, bind=engine)
Base= declarative_base()

def get_db():
    db= SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_connection():
    try:
        db= SessionLocal()
        db.execute(text("SELCT 1"))
        print('DB Connection sucessfull')
    except:
        print('Connection Failed')
    finally:
        db.close()
