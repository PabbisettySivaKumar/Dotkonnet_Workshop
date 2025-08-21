from pydantic import BaseModel

class BookTitle(BaseModel):
    title: str
    author: str
    description: str
    year: int

class BookCreate(BookTitle):
    pass

class Book(BookTitle):
    id: int

