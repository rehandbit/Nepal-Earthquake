import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


if __name__ == '__main__':
    try:
        logging.info('Training Pipeline Started')
        
        data_ingestion = DataIngestion()
        train_path , test_path = data_ingestion.inititate_data_ingestion()


        data_transformation = DataTransformation()
        train_arr, test_arr , preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        model_trainer = ModelTrainer()
        best_model_name, training_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        model_evaluation = ModelEvaluation()
        final_report  =model_evaluation.initiate_model_evaluation(test_arr)


        logging.info('Training Pipeline COmpleted')

        print(f'best model -> {best_model_name}')
        print(f'\n Training report : ')
        for name, metrics in training_report.items():
            print(f'\n {name} : {metrics}')

        print(f'\n FInal Evaluation on test set :')
        print(f"\n Accuracy : {final_report['accuracy']:.4f}")
        print(f"\n F1 - (macro) : {final_report['f1_macro']:.4f}")
        print(f"\n F1 - (Weighted) : {final_report['f1_weighted']:.4f}")
        print(f"\n ROC AUC (macro ovr) : {final_report['roc_auc_macro_ovr']:.4f}")

        print(f'\n Model Save to :artifacts.model.pkl')

        print(f'preprocessor save to : {preprocessor_path}')
        print(f'Full evaluation report on : artifacts/evaluation_report.json')

    except Exception as e:
        raise CustomException(e,sys)