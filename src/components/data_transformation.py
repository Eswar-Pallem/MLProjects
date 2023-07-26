import contextlib
import sys
from dataclasses import dataclass
# The `import` statement in Python is used to import modules or specific objects from modules. It
# allows you to use functions, classes, and variables defined in other Python files or modules in your
# current Python file.
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            num_columns=['reading_score', 'writing_score']
            cat_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns:{cat_columns}')
            logging.info(f'numerical columns:{num_columns}')

            preprocessor= ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns),
                ]

                )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
    
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessor object')


            preprocessing_obj=self.get_data_transformation_object()

            target_col_name='math_score'
            num_columns=['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]

            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_df[target_col_name]


            logging.info('Applying preprocessing on training and test dataframes')
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)
            
            train_array=np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_array=np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]
            
            logging.info('Saved Preprocessed Objects')
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)