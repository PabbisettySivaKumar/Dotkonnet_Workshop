from sqlalchemy.orm import Session
from fastapi import Depends
from model import Books
from schema import Book, BookCreate, BookTitle
from database import get_db

def get_books(db: Session):
    return db.query(Books).all()

def get_book(db: Session, book_id: int):
    return db.query(Books).filter(Books.id == book_id).first()

def create_book(db: Session, book: BookCreate):
    cre_book = Books(
        title=book.title,
        author=book.author,
        year=book.year,
        description=book.description
    )
    db.add(cre_book)
    db.commit()
    db.refresh(cre_book)
    return cre_book

def delete_book(db: Session, book_id: int):
    book = db.query(Books).filter(Books.id == book_id).first()
    if book:
        db.delete(book)
        db.commit()
    return book