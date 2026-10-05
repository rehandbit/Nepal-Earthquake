import sys, os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipelineConfig:
    model_path :str = os.path.join('artifacts', 'model.pkl')
    preprocessor_path :str= os.path.join('artifacts', 'preprocessor.pkl')

class PredictPipeline:
    def __init__(self):
        self.predictpipelineconfig = PredictPipelineConfig()

    def initiate_predict_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info('Loading model and preprocessing bundle')
            model = load_object(self.predictpipelineconfig.model_path)
            bundle = load_object(self.predictpipelineconfig.preprocessor_path)

            preprocessor = bundle['preprocessor']
            label_encoder = bundle['label_encoder']
            age_median = bundle['age_median']
            geo_freq_maps = bundle['geo_freq_maps']

            df = raw_df.copy()
            building_ids = df['building_id']

            df["age"] = df['age'].replace(995, age_median)

            superstructure_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
            secondary_use_cols = [c for c in df.columns if c.startswith("has_secondary_use_") and c != "has_secondary_use"]

            df["total_superstructure_materials"] = df[superstructure_cols].sum(axis=1)
            df["total_secondary_uses"] = df[secondary_use_cols].sum(axis=1)


            for col, freq_map in geo_freq_maps.items():
                df[col + "_freq_enc"] = df[col].map(freq_map).fillna(0)

            df = df.drop(columns=list(geo_freq_maps.keys()))

            X = df.drop(columns = 'building_id')
            x_transformed = x_transformed.toarray()


            prediction_encoded = model.predict(x_transformed)
            prediction_final = label_encoder.inverse_transformed(prediction_encoded.astype(int))

            logging.info(f"Generated {len(prediction_final)} predictions")

            result_df = pd.DataFrame({
                'Building_id' : building_ids,
                'damage_grade' : prediction_final
            })
            return result_df

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    input_path = os.path.join('data', 'test_values.csv')
    output_path = os.path.join('artifacts', 'submission.csv')

    raw_test_df = pd.read_csv(input_path)
    
    pipeline = PredictPipeline()

    submission_df = pipeline.predict(raw_test_df)
    submission_df.to_csv(output_path, index=False)

    print(f'Saved prediction to {output_path}')
    print(submission_df.head())