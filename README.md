# Disaster Response Pipeline Project

## Project Description
In this project we are building an API to classify and model messages that are sent when disasters strike. The model takes a user messages and classifies them into 1 of 36 categories that are already defined and sends the message to the correct disaster agency. In this project we will be working with data provided by [Figure Eight](https://www.figure-eight.com/) usinf real messages that have been sent.

## File Description
    app
    - template
        - master.html # main page of web app
        - go.html # classification result page of web app
    - run.py # Flask file that runs app
    data
        - disaster_categories.csv # data to process
        - disaster_messages.csv # data to process
    - process_data.py
        - InsertDatabaseName.db # database to save clean data to
    models
        - train_classifier.py
        - classifier.pkl # saved model
    README.md

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

3. Go to http://0.0.0.0:3000/

## Licensing, Authors, Acknowledgements
Thanks to the mentors on Ydacity knowledge for helpinh me solve some of my problems. As well as figure 8 providing us with the data to build our version of a API  model.



