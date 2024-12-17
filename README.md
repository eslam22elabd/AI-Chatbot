# AI Chatbot using GPT-2 and Streamlit

## Introduction
This project implements a chatbot interface using the GPT-2 model for generating responses. If a predefined answer exists in the CSV dataset (AI.csv), the chatbot will return it directly; otherwise, it generates a response using GPT-2.

## Features:
Streamlit Interface for easy user interaction.
Predefined answers: The bot checks a CSV file for pre-written answers.
GPT-2: Generates responses dynamically when no predefined answer is available.
## How to Run:
Ensure you have Python installed.
## Install dependencies:
# copy this 
pip install streamlit transformers pandas

## Run the app:

streamlit run main.py

## Files:
main.py: Main script to run the chatbot.
AI.csv: Contains predefined questions and answers.
