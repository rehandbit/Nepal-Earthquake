import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts', 'preprocessor.pkl')
    train_arr_path : str = os.path.join('artifacts', 'train_arr.npy')
    test_arr_path : str = os.path.join('artifacts', 'test_arr.npy')

    
    id_column :str = 'building_id'

    geo_columns : tuple = ('geo_level_1_id','geo_level_2_id','geo_level_3_id')
    target_columns :str = 'damage_grade'
    categorical_columns: tuple = (
        'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
        'other_floor_type','position','plan_configuration','legal_ownership_status'
    )


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor_object(self, numeric_features, categorical_features):
        numerical_pipeline = Pipeline(steps =[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))

        ])


        preprocessor = ColumnTransformer(transformers=[
            ('numeric', numerical_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])

        return preprocessor


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data Transformation Started')
            logging.info('Reading Train and Test csv')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #cleaning
            age_median = train_df.loc[train_df['age'] != 995, 'age'].median()
            train_df['age'] = train_df['age'].replace(995, age_median)
            test_df['age'] = test_df['age'].replace(995, age_median)

            logging.info(f"Replaced age == 995 with train_df['age'] median value {age_median}")

            # Feature Engineering

            superstructure_cols = [c for c in train_df.columns if c.startswith('has_superstructure_')]
            secondary_use_cols = [c for c in train_df.columns if c.startswith('has_secondary_use_') and c != "has_secondary_use"]

            for frame in (train_df, test_df):
                frame['total_superstructure_materials'] = frame[superstructure_cols].sum(axis = 1)
                frame['total_secondary_uses'] = frame[secondary_use_cols].sum(axis = 1)
            
            logging.info(f'Added engineered feature : total_superstructure_materials , total_secondary_uses')

            # frequency encoded 'GEO'
            geo_freq_map = {}
            for col in self.transformation_config.geo_columns:
                freq_map = train_df[col].value_counts(normalize = True)
                geo_freq_map[col] = freq_map

                train_df[col + "_freq_enc"] = train_df[col].map(freq_map)
                test_df[col + "_freq_enc"] = test_df[col].map(freq_map).fillna(0)

            train_df = train_df.drop(columns = list(self.transformation_config.geo_columns))
            test_df = test_df.drop(columns = list(self.transformation_config.geo_columns))


            logging.info('Frequence encoded geo_level columns done')

            # separate feature
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_df[self.transformation_config.target_columns])
            y_test = label_encoder.transform(test_df[self.transformation_config.target_columns])

            logging.info(f'Target classes : {label_encoder.classes_} -> encoded as {list(range(len(label_encoder.classes_)))}')

            drop_cols = [self.transformation_config.id_column, self.transformation_config.target_columns]

            x_train_df = train_df.drop(columns = drop_cols)
            x_test_df = test_df.drop(columns = drop_cols)


            # fit preprocessor 
            categorical_features = list(self.transformation_config.categorical_columns)
            numeric_features = [c for c in x_train_df.columns if c not in categorical_features]

            preprocessor = self.get_preprocessor_object(numeric_features, categorical_features)

            logging.info(f'Fitting preprocessor on training data')

            x_train_transformed = preprocessor.fit_transform(x_train_df)
            x_test_transformed = preprocessor.transform(x_test_df)

            if hasattr(x_train_transformed, 'toarray'):
                x_train_transformed = x_train_transformed.toarray()

            if hasattr(x_test_transformed, 'toarray'):
                x_test_transformed = x_test_transformed.toarray()

            # combining

            train_arr = np.c_[x_train_transformed, y_train]
            test_arr = np.c_[x_test_transformed, y_test]

            logging.info(f"Final train array shape: {train_arr.shape}, test array shape : {test_arr.shape}")


            # saving artifacts
            preprocessing_bundle = {
                'preprocessor': preprocessor,
                'label_encoder': label_encoder,
                'age_median' : age_median,
                'geo_freq_map' :geo_freq_map,
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
            }

            save_object(self.transformation_config.preprocessor_obj_file_path, preprocessing_bundle)
            logging.info(f'saved preprocessing bundlge to {self.transformation_config.preprocessor_obj_file_path}')

            np.save(self.transformation_config.train_arr_path, train_arr)
            np.save(self.transformation_config.test_arr_path, test_arr)

            return(
                train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            CustomException(e, sys)

if __name__ == '__main__':
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
    
    print('Preprocessor saved at ' , preprocessor_path)
    print('train array shape', train_arr.shape)
    print('test array shape', test_arr.shape)


