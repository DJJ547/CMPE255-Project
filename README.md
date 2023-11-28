# CMPE255-Project
This is a web application with a predictive model that will determine whether a user is healthy, pre-diabetic, or diabetic by analyzing user input from a health survey.

## Backend Project Layout
```
CMPE255-Project/
├── models/
    ├── __init__.py
    ├── decision_tree.py
    ├── knn.py
├── statics/
    ├── diabetes_binary_health_indicators_BRFSS2015.csv
├── templates/
│   ├── index.html
    ├── result.html
├── utils/
│   ├── process_data.py
├── app.py
├── index.css
├── README.md
├── requirements.txt
```

## How to run our Python Flask Back-end:
- Make sure you installed your interpreter with a virtual environment '<your path>\teamproject-team-alpha-1\Backend' (venv)
```
python -m venv myenv
```
- make sure the path is '<your path>\teamproject-team-alpha-1\Backend', activate your venv first with this command:
```
venv\Scripts\activate
```
- then run:
```
 pip install -r requirements.txt
```
- Now you can start your backend locally with:
```
python app.py
```
- A link should be in output console, or you can use the link: http://127.0.0.1:5000/
- when you finsih the session of programming. deactivate the virtual environemnt:
```
deactivate
```