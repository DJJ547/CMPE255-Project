from flask import Flask, render_template, request, json, Response
from flask_cors import CORS
from utils.process_data import calculate_bmi, determine_heavy_drinker
from models.knn import predict_with_knn_model
from models.decision_tree import predict_with_decision_tree_model


app = Flask(__name__, template_folder="templates")
CORS(app)

DATASET_COLUMN_ORDER = ['hbp', 'hc', 'cholCheck', 'bmi', 'smoker', 'stroke', 'heart', 'physicalAct', 'fruit', 'veggies', 'heavyAlcohol', 'healthcare', 'doctor', 'healthScore', 'mental', 'physical', 'walk', 'gender', 'age', 'education', 'income']
MODEL_TO_USE = 'decision_tree'

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def submit():
    survey_data = request.get_json()
    print(survey_data)
    bmi = calculate_bmi(survey_data['weight'], survey_data['height_ft'], survey_data['height_in'])
    print(bmi)
    survey_data['bmi'] = bmi
    is_heavy_drinker = determine_heavy_drinker(survey_data['gender'], survey_data['alcohol'])
    print(is_heavy_drinker)
    survey_data['heavyAlcohol'] = is_heavy_drinker

    # put data back in order
    user_input = []
    for name in DATASET_COLUMN_ORDER:
        user_input.append(int(survey_data[name]))
    user_input = [user_input]

    if MODEL_TO_USE == 'knn':
        result = predict_with_knn_model(user_input)
        print("predicted result: ", int(result[0]))
    elif MODEL_TO_USE == 'decision_tree':
        result = predict_with_decision_tree_model(user_input)
        print("predicted result: ", int(result[0]))

    result = result.tolist()[0]
    res = {'error': False, 'prediction': result}
    return Response(json.dumps(res), status=200)


if __name__ == '__main__':
    app.run(debug=True)
