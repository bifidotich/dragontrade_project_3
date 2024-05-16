import os
from flask import Flask, request, render_template
import json
import tempfile
from tensorflow import keras

app = Flask(__name__)


def compare_models(model1, model2):
    comparison_result = []
    for layer1, layer2 in zip(model1.layers, model2.layers):
        layer_diff = {}
        layer_diff['Layer Name'] = layer1.name
        params1 = layer1.get_config()
        params2 = layer2.get_config()
        param_diff = []

        for key in params1:
            param1 = params1[key]
            param2 = params2.get(key)
            match = param1 == param2
            param_diff.append((key, param1, param2, match))

        layer_diff['Parameter Diff'] = param_diff
        comparison_result.append(layer_diff)

    optimizer1 = model1.optimizer.get_config()
    optimizer2 = model2.optimizer.get_config()
    optimizer_match = optimizer1 == optimizer2

    loss1 = model1.loss
    loss2 = model2.loss
    loss_match = loss1 == loss2

    comparison_result.append({
        'Layer Name': 'Optimizer',
        'Parameter Diff': [('Optimizer', optimizer1, optimizer2, optimizer_match)]
    })

    comparison_result.append({
        'Layer Name': 'Loss',
        'Parameter Diff': [('Loss', loss1, loss2, loss_match)]
    })

    return comparison_result


def save_previous_file(key, filename):
    previous_files = load_previous_files()
    previous_files[key] = filename
    with open("previous_files.json", "w") as file:
        json.dump(previous_files, file)


def load_previous_files():
    if os.path.exists("previous_files.json"):
        with open("previous_files.json", "r") as file:
            return json.load(file)
    return {"model1": None, "model2": None}


def save_temporary_file(file):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)
    file.save(temp_file_path)
    return temp_file_path


@app.route("/", methods=["GET", "POST"])
def index():
    model1 = None
    model2 = None
    comparison_result = None
    previous_files = load_previous_files()

    if request.method == "POST":
        if "model1" in request.files:
            temp_file_path1 = save_temporary_file(request.files["model1"])
            model1 = keras.models.load_model(temp_file_path1)
            save_previous_file("model1", request.files["model1"].filename)
        if "model2" in request.files:
            temp_file_path2 = save_temporary_file(request.files["model2"])
            model2 = keras.models.load_model(temp_file_path2)
            save_previous_file("model2", request.files["model2"].filename)

        if model1 and model2:
            comparison_result = compare_models(model1, model2)

    return render_template("compare_models.html", comparison_result=comparison_result, previous_files=previous_files)


if __name__ == "__main__":
    app.run(debug=True)
