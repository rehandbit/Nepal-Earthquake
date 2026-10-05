import os, sys, json
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect
from src.pipeline.prediction_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)
app.secret_key = "earthquake-damage-prediction-secret-key"

UPLOAD_FOLDER = "artifacts/uploads"
OUTPUT_FOLDER = "artifacts/predictions"
EVALUATION_REPORT_PATH = "artifacts/evaluation_report.json"
CLASS_LABELS = [1, 2, 3]  # 1=low, 2=medium, 3=almost complete destruction

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_evaluation_report():
    """Reads the saved evaluation report (produced by model_evaluation.py)."""
    if not os.path.exists(EVALUATION_REPORT_PATH):
        return None
    with open(EVALUATION_REPORT_PATH, "r") as f:
        return json.load(f)


def build_confusion_matrix_table(cm):
    """Turns the raw confusion matrix (list of lists) into an HTML table with row/column labels."""
    header = "<tr><th>Actual \\ Predicted</th>" + "".join(f"<th>{lbl}</th>" for lbl in CLASS_LABELS) + "</tr>"
    rows = ""
    for i, row in enumerate(cm):
        rows += f"<tr><th>{CLASS_LABELS[i]}</th>" + "".join(f"<td>{val}</td>" for val in row) + "</tr>"
    return f"<table class='cm-table'>{header}{rows}</table>"


@app.route("/", methods=["GET"])
def dashboard():
    """Model Performance Dashboard - shows how the trained model performed on the test set."""
    report = load_evaluation_report()

    if report is None:
        flash("No evaluation report found yet. Run the training pipeline first "
              "(python -m src.pipeline.train_pipeline) to generate artifacts/evaluation_report.json.")
        return render_template("dashboard.html", report=None)

    cm_table = build_confusion_matrix_table(report["confusion_matrix"])

    metrics = {
        "Accuracy": f"{report['accuracy']:.4f}",
        "F1 (macro)": f"{report['f1_macro']:.4f}",
        "F1 (weighted)": f"{report['f1_weighted']:.4f}",
        "Precision (macro)": f"{report['precision_macro']:.4f}",
        "Recall (macro)": f"{report['recall_macro']:.4f}",
        "ROC-AUC (macro, OVR)": f"{report['roc_auc_macro_ovr']:.4f}" if report.get("roc_auc_macro_ovr") else "N/A",
    }

    return render_template(
        "dashboard.html",
        report=report,
        metrics=metrics,
        cm_table=cm_table,
        classification_report_text=report["classification_report"]
    )


@app.route("/predict-page", methods=["GET"])
def predict_page():
    """Renders the CSV upload form for generating predictions on new buildings."""
    return render_template("predict.html")


# ---------------------------------------------------------------------------
# Single-building prediction (dropdowns + checkboxes, no CSV needed)
# ---------------------------------------------------------------------------

NUMERIC_FIELDS = [
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
    "count_floors_pre_eq", "age", "area_percentage", "height_percentage",
    "count_families"
]

CATEGORICAL_OPTIONS = {
    "land_surface_condition": ["n", "o", "t"],
    "foundation_type": ["h", "i", "r", "u", "w"],
    "roof_type": ["n", "q", "x"],
    "ground_floor_type": ["f", "m", "v", "x", "z"],
    "other_floor_type": ["j", "q", "s", "x"],
    "position": ["j", "o", "s", "t"],
    "plan_configuration": ["a", "c", "d", "f", "m", "n", "o", "q", "s", "u"],
    "legal_ownership_status": ["a", "r", "v", "w"],
}

SUPERSTRUCTURE_FIELDS = [
    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber", "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
    "has_superstructure_other"
]

SECONDARY_USE_FIELDS = [
    "has_secondary_use", "has_secondary_use_agriculture", "has_secondary_use_hotel",
    "has_secondary_use_rental", "has_secondary_use_institution", "has_secondary_use_school",
    "has_secondary_use_industry", "has_secondary_use_health_post",
    "has_secondary_use_gov_office", "has_secondary_use_use_police", "has_secondary_use_other"
]


@app.route("/predict-form", methods=["GET"])
def predict_form():
    """Renders the single-building form (dropdowns + checkboxes)."""
    return render_template(
        "predict_form.html",
        numeric_fields=NUMERIC_FIELDS,
        categorical_options=CATEGORICAL_OPTIONS,
        superstructure_fields=SUPERSTRUCTURE_FIELDS,
        secondary_use_fields=SECONDARY_USE_FIELDS
    )


@app.route("/predict-single", methods=["POST"])
def predict_single():
    """Builds a one-row dataframe from the form and predicts its damage grade."""
    try:
        form = request.form
        data = {"building_id": [1]}  # dummy id, PredictPipeline expects this column

        for field in NUMERIC_FIELDS:
            data[field] = [int(form.get(field, 0))]

        for field in CATEGORICAL_OPTIONS:
            data[field] = [form.get(field)]

        for field in SUPERSTRUCTURE_FIELDS + SECONDARY_USE_FIELDS:
            data[field] = [1 if form.get(field) == "on" else 0]

        single_df = pd.DataFrame(data)

        pipeline = PredictPipeline()
        result_df = pipeline.initiate_predict_pipeline(single_df)
        predicted_grade = int(result_df["damage_grade"].iloc[0])

        grade_labels = {1: "Low damage", 2: "Medium damage", 3: "Almost complete destruction"}

        return render_template(
            "predict_form.html",
            numeric_fields=NUMERIC_FIELDS,
            categorical_options=CATEGORICAL_OPTIONS,
            superstructure_fields=SUPERSTRUCTURE_FIELDS,
            secondary_use_fields=SECONDARY_USE_FIELDS,
            predicted_grade=predicted_grade,
            predicted_label=grade_labels[predicted_grade]
        )

    except Exception as e:
        logging.error(str(e))
        flash(f"Something went wrong: {str(e)}")
        return redirect("/predict-form")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles the CSV upload, runs predictions, shows a preview + download link."""
    try:
        if "file" not in request.files:
            flash("No file uploaded. Please choose a CSV file.")
            return redirect("/predict-page")

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect("/predict-page")

        if not file.filename.endswith(".csv"):
            flash("Please upload a .csv file.")
            return redirect("/predict-page")

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        logging.info(f"Received file: {file.filename}")

        raw_df = pd.read_csv(input_path)
        if "building_id" not in raw_df.columns:
            flash("CSV must contain a 'building_id' column.")
            return redirect("/predict-page")

        pipeline = PredictPipeline()
        result_df = pipeline.initiate_predict_pipeline(raw_df)

        output_filename = "predictions_" + file.filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        result_df.to_csv(output_path, index=False)
        logging.info(f"Saved predictions to {output_path}")

        preview_html = result_df.head(20).to_html(index=False, classes="preview-table")
        return render_template(
            "predict.html",
            preview=preview_html,
            download_file=output_filename,
            total_rows=len(result_df)
        )

    except Exception as e:
        logging.error(str(e))
        flash(f"Something went wrong: {str(e)}")
        return redirect("/predict-page")


@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)