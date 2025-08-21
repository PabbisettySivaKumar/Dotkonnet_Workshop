from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from database import engine, Base, get_db
from model import Books
from model_services import get_book, get_books, delete_book, create_book
from schema import BookTitle, BookCreate, Book
from sqlalchemy import create_engine

Base.metadata.create_all(bind=engine)

app= FastAPI()

@app.post('/createbooks', response_model=Book)
def create_books(book: BookCreate, db:Session= Depends(get_db)):
    return create_book(db=db, book=book)

@app.get('/book/{book_id}', response_model=BookTitle)
def read_book( book_id:int, db:Session= Depends(get_db)):
    book= get_book(db, book_id)
    if not book:
        return HTTPException(status_code= 404, detail='book not found')
    return book

@app.get('/gbooks', response_model=list[BookTitle])
def read_books(db:Session= Depends(get_db)):
    return get_books(db)


@app.delete('/books/{book_id}')
def del_book(book_id: int, db:Session=Depends(get_db)):
    book= delete_book(db, book_id)
    if not book:
        return HTTPException(status_code= 404, detail='book not found')





