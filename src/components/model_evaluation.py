import os, sys, json
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

@dataclass
class ModelEvaluationConfig:
    model_path : str = os.path.join('artifacts', 'model.pkl')
    report_path :str = os.path.join('artifacts', 'evaluation_report.json')


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, test_array, class_label =(1,2,3)):
        try:
            logging.info('Loading Trained Model --')

            model = load_object(self.model_evaluation_config.model_path)
            X_test ,y_test = test_array[:, :-1], test_array[:, -1]

            logging.info('Generating Predictions on test set')

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            precision_macro = precision_score(y_test, y_pred, average="macro")
            recall_macro = recall_score(y_test, y_pred, average="macro")

            logging.info(f"Accuracy : {accuracy:.4f} | F1 macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted}")

            roc_auc = None

            y_proba = model.predict_proba(X_test)
            y_test_binary = label_binarize(y_test, classes=list(range(len(class_label))))
            roc_auc = roc_auc_score(y_test_binary, y_proba, average="macro", multi_class="ovr")
            logging.info(f'Roc-AUC (macro, ovr) : {roc_auc:.4f}')

            cm = confusion_matrix(y_test, y_pred)
            class_report_test = classification_report( y_test, y_pred, target_names=[str(c) for c in class_label])

            logging.info(f'\n {class_report_test}')

            report = {
                "accuracy" : accuracy,
                "f1_macro" : f1_macro,
                "f1_weighted" :  f1_weighted,
                "precision_macro" : precision_macro,
                "recall_macro" : recall_macro,
                "roc_auc_macro_ovr" :  roc_auc,
                "confusion_matrix" :  cm.tolist(),
                "classification_report" :  class_report_test
            }

            os.makedirs(os.path.dirname(self.model_evaluation_config.report_path), exist_ok=True)
            with open(self.model_evaluation_config.report_path, "w")as f:
                json.dump(report, f ,indent=4)
            logging.info(f"Saved evaluation report to {self.model_evaluation_config.report_path}")

            return report


        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    test_arr = np.load(os.path.join('artifacts', 'test_arr.npy'))
    evaluator = ModelEvaluation()
    final_report = evaluator.initiate_model_evaluation(test_arr)
    print(json.dumps(final_report, indent=4))