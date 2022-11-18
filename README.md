# Heart-Disease-Prediction

Heart disease predictor with machine learning (ML) model integrated in a Django web application.

1. **Machine Learning Implementation**

To execute the model comparison program, used to compare the different the designed ML models, simply run the HeartPredStudy.py file. Navigate to the  DjangoAPI directory and execute the HeartPredStudy.py file, you can do so by running "python HeartPredStudy.py" in your terminal. Model Visualisation plots will be outputted, which will have to be closed for the program to resume running. In the terminal, one can see the results of the individual models. Finally, the models and their performances will be outputted in the ModelCompare.xlsx file.

**Things to Note**

- At the model creation stage, for each of the models, for faster testing, the second list of parameters are initialised which include the optimal hyperparameter values that have been previously identified. To perform the exhaustive operation of the gridsearchcv to find the hyperparameter values, one can comment out the second list of parameters to begin searching for the hyperparameter values again.
- Towards the bottom of the program, under the ANN section, are three tests to find the optimal parameter values for the MLPClassifier model. To perform those test runs, one can uncomment any of those 3 function calls at the bottom of that section.
- Also, at the bottom of the program are other models that have been tested, which have been commented out as they are found to produce less performance metrics values. One can uncomment them to view their performance accordingly.



2. **Django Web App Implementation**

For the Django web app, one can load the deployed and hosted version of the app by visiting https://heartassist.net/. Otherwise, to locally host and run the Django web application, there are two ways to do it, either using Docker or the built-in Django development server:


**Method 1: Using Docker**

Ensure that you have docker installed in your system.

1. Using your terminal enter the following command "sudo docker-compose -f DjangoAPI/docker-compose.yml up -d --build".
2. After that, the web app has finished building, you can access it through your web browser through the localhost port 8000, either by entering "http://127.0.0.1:8000/", "http://localhost:8000/" or "http://0.0.0.0:8000/".


**Method 2: Using the default built-in Django development server or Gunicorn WSGI HTTP server**

1. First, install the dependencies in the requirements.txt file by entering "pip install -r requirements.txt" in your terminal. Ensure you have pip installed for this. Also ensure that you are in the same directory as the requirements.txt file. You may create a virtual environment (venv) for installing these dependencies.
2. After that, navigate to the DjangoAPI directory to be in the same directory as the manage.py file.
3. Then, enter "python manage.py runserver" (for default Django development server) or "gunicorn DjangoAPI.wsgi:application --bind 0.0.0.0:8000" (for gunicorn WSGI HTTP server) in your terminal.
4. After that, the web app has finished building, you can access it through your web browser through the localhost port 8000,
   either by entering "http://127.0.0.1:8000/", "http://localhost:8000/" or "htt://0.0.0.0:8000/".
