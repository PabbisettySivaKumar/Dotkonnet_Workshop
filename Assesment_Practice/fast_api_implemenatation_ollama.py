from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import re
import requests
import os
import ollama

nltk.download('punkt')

app= FastAPI()

#Base Class

class UserInput(BaseModel):
    user_input: str

#extraction celcius

def extract_celcius(text: str):
    tokens= nltk.word_tokenize(text)
    numbers= None
    for token in tokens:
        if token.isdigit():
            numbers= float(token)
    
    if not numbers:
        raise ValueError("No Numeric value found in input")
    
    return numbers

def celcius_to_farenheit(celcius: float):
    return round((celcius*9/5)+32,2)

def ask_ai(celcius: float, fahrenheit: float, user_input:str):

    response= ollama.chat(
        model='mistral',
        messages=[
            {
                "role":"system", 
             "content":"you are a helpful assistant that explaint temperature conversion"
             },
            {
                "role":"user", 
                "content": f"the user said: '{user_input}'. The extracted Celcius is {celcius}."f"what is equivalent in Fahrenheit? ({fahrenheit}F)."
            }
        ],
    )
    print(response)
    if "messgae" in response:
        return response["messages"]["content"]
    return response
    

@app.post('/convert_temperatures')
def convert_temperature(input_data:UserInput):
    user_input= input_data.user_input
    celcius_val= extract_celcius(user_input)
    fahren_val= celcius_to_farenheit(celcius_val)
    ai_response= ask_ai(celcius_val, fahren_val, user_input)

    return {

        "original_input":user_input,
        "celsius_value":celcius_val,
        "fahrenheit_value":fahren_val,
        "ai_response": ai_response
    }


