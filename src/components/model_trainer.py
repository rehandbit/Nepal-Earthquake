import os, sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

from sklearn.utils.class_weight import compute_sample_weight

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    trained_model_path : str = os.path.join('artifacts', 'model.pkl')
    use_class_weight: bool = False
    tune_best_model: bool =True


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arrary, test_array):
        try:
            logging.info('Spliting train/test array into x and y')
            X_train, y_train = train_arrary[:, :-1], train_arrary[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]


            sample_weights = None
            if self.model_trainer_config.use_class_weight:
                sample_weights = compute_sample_weight('balanced', y_train)
                logging.info('Using class-balanced sample weight during training')

            
            models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000, solver='lbfgs', random_state=42, n_jobs= -1
                ),
                'Random Forest' : RandomForestClassifier(
                    n_estimators=200, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs= -1
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8,
                    objective = 'multi:softprob', num_class = 3, random_state= 42, n_jobs= -1, eval_metric='mlogloss'
                )
            }
            trained_models = {}
            report = {}


            for model_name, model in models.items():
                logging.info(f'Model Training started. {model_name}')
                if sample_weights is not None:
                    model.fit(X_train, y_train, sample_weight = sample_weights)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                report[model_name] = {
                    'test_accuracy' : accuracy_score(y_test, y_pred),
                    'test_f1_macro' : f1_score(y_test, y_pred, average="macro")
                }
                trained_models[model_name] = model



                logging.info(
                    f"{model_name} -> accuracy: {report[model_name]['test_accuracy']:.4f}, "
                    f"f1_macro: {report[model_name]['test_f1_macro']:.4f}"
                )


            best_model_name = max(report, key=lambda name: report[name]['test_f1_macro'])
            best_model = trained_models[best_model_name]

            logging.info(f"Best model before tuning : {best_model_name} (F1-macro: {report[best_model_name]['test_f1_macro']:.4f})")


            #hyperparameter tuning
            if self.model_trainer_config.tune_best_model and best_model_name in ('Random Forest', 'XGBoost'):
                logging.info(f' TUning {best_model_name} with RandomSearchCV')

                if best_model_name == 'RandomForest':
                    param_distributions = {
                        'n_estimators' : [100, 200, 300, 400],
                        'max_depth' : [10, 15, 20 ,25, None],
                        'min_samples_split': [2,5,10],
                        'min_samples_leaf': [1,2,3,5],
                        'max_features': ['sqrt', 'log2']
                    }
                    estimator = RandomForestClassifier(random_state=42, n_jobs= -1)
                else:
                    param_distributions = {
                        'n_estimators' : [100, 200, 300, 400,500,600,800],
                        'max_depth' : [3,4,5,6,7,8,9,10],
                        'learning_rate':[0.005,0.01,0.05,0.1,0.15,0.2, 0.3],
                        'colsample_bytree': [0.5,0.6,0.7,0.8,0.9,1.0],
                        'subsample': [0.5, 0.6,0.7,0.8,0.9,1.0],
                        'min_child_weight': [1,2,3,5,7],
                        'gamma': [0, 0.1, 0.2, 0.3],
                        'reg_alpha': [0, 0.01, 0.1, 1],
                        'reg_lambda': [0.5, 1, 1.5, 2],
                    }
                    estimator = XGBClassifier(
                        objective = 'multi:softprob', num_class =3, random_state =42, n_jobs=-1, eval_metric='mlogloss'
                    )
                random_search = RandomizedSearchCV(
                    estimator=estimator, param_distributions=param_distributions, n_iter=50, cv=5, scoring='f1_macro',
                    random_state=42, n_jobs=-1, verbose=1
                )
                fit_kwargs = {"sample_weight": sample_weights} if sample_weights is not None else {}
                random_search.fit(X_train, y_train, **fit_kwargs)

                best_model = random_search.best_estimator_
                tuned_pred = best_model.predict(X_test)
                tuned_f1 = f1_score(y_test, tuned_pred, average="macro")

                logging.info(f"Best params found: {random_search.best_params_}")
                logging.info(f"Tuned {best_model_name} test F1-macro: {tuned_f1:.4f}")

                report[best_model_name + "_tuned"] = {
                    "test_accuracy": accuracy_score(y_test, tuned_pred),
                    "test_f1_macro": tuned_f1,
                    "best_params": random_search.best_params_
                }
            print(classification_report(y_test, best_model.predict(X_test)))    

            #save model
            save_object(self.model_trainer_config.trained_model_path, best_model)
            logging.info(f'Model save to {self.model_trainer_config.trained_model_path}')


            return best_model_name, report

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    train_arr = np.load(os.path.join('artifacts', 'train_arr.npy'))
    test_arr = np.load(os.path.join('artifacts', 'test_arr.npy'))

    trainer = ModelTrainer()
    best_name , full_report = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f' Best Model: {best_name}')
    print(f' Full report: {full_report}')
