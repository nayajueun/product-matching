import os
from utils import load_file, get_similarity_sbert, get_similarity_keyval
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import util, losses
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Pipeline():
    def __init__(self, input_dir, output_dir, finetuning=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.finetuning = finetuning

    def train(self):
        df = self.parse_data()
        if self.finetuning:
            sbert_model = self.finetune_sbert(
                df=df,
                output_dir=self.output_dir,
                n_epochs=1
            )
        else: sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        numerical_df = self.compute_features(
            df=df,
            ft_model=sbert_model
        )
        self.train_model(
            num_data=numerical_df
        )

    def parse_data(self, input_dir=None):
        """
        Loads dataframe and extracts necessary info
        :param input_dir: data_dir that contains data to train
        :return: DataFrame
        """
        # TODO: load csv from directory.
        input_file = './data/computers_train/computers_train_medium.json.gz'
        df = pd.read_json(input_file, compression='gzip', lines=True)
        df = df[['title_left', 'description_left', 'keyValuePairs_left', 'brand_left',
                 'title_right', 'description_right', 'keyValuePairs_right', 'brand_right', 'label']]
        return df

    def finetune_sbert(self, df, output_dir, n_epochs=1):
        """
        Finetuning the BERT model
        :param df:
        :param output_dir:
        :param n_epochs:
        :return:
        """
        ft_train_df = df.loc[(df.label == 1)]

        descriptions_left = list(ft_train_df['description_left'])
        descriptions_right = list(ft_train_df['description_right'])

        train_samples = []
        for desc_left, desc_right in zip(descriptions_left, descriptions_right):
            if desc_left is not None and desc_right is not None:
                train_samples.append(InputExample(
                    texts=[desc_left, desc_right]
                ))

        train_dataloader = DataLoader(train_samples, batch_size=8)
        title_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        loss = losses.MultipleNegativesRankingLoss(title_model)
        warmup_steps = int(len(train_dataloader) * n_epochs * 0.1)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples, name='sts-dev')

        title_model.fit(
            train_objectives=[(train_dataloader, loss)],
            evaluator=evaluator,
            epochs=n_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=output_dir
        )
        return title_model

    def get_similarity_vectorizer(self, value_left, value_right, vectorizer):
        if value_left is None or value_right is None:
            return None

        if value_left == value_right:
            return 1.0

        tf_idf_left = vectorizer.transform([value_left])
        tf_idf_right = vectorizer.transform([value_right])
        sim = cosine_similarity(tf_idf_left, tf_idf_right)[0][0]
        return sim

    def compute_features(self, df, ft_model):
        # TODO: encode all sentences at once to sbert (currently too inefficient)
        num_df = pd.DataFrame()
        text_df = pd.concat([df['title_left'], df['title_right']])
        vectorizer = TfidfVectorizer()
        vectorizer = vectorizer.fit(text_df.values)
        num_df['title_sim'] = df.apply(
            lambda x: self.get_similarity_vectorizer(x['title_left'], x['title_right'], vectorizer), axis=1)
        print("title done")

        num_df['description_sim'] = df.apply(
            lambda x: get_similarity_sbert(ft_model, (x['description_left'], x['description_right'])), axis=1)
        print("description done")

        text_df = pd.concat([df['brand_left'], df['brand_right']])
        vectorizer = TfidfVectorizer()
        vectorizer = vectorizer.fit(text_df.values)
        num_df['brand_sim'] = df.apply(lambda x: self.get_similarity_vectorizer(x['brand_left'], x['brand_right'], vectorizer), axis=1)

        return df

    def train_model(self, num_data):
        y = np.array(num_data['label'])
        X = num_data.drop(columns=["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
        # create xgboost matrices

        Train = xgb.DMatrix(X_train, label=y_train)
        Test = xgb.DMatrix(X_test, label=y_test)
        # set the parameters
        parameters = {'learning_rate': 0.3,
                      'max_depth': 2,
                      'colsample_bytree': 1,
                      'subsample': 1,
                      'min_child_weight': 1,
                      'gamma': 0,
                      'random_state': 1500,
                      'eval_metric': "auc",
                      'objective': "binary:logistic"}

        model = xgb.train(params=parameters,
                          dtrain=Train,
                          num_boost_round=200,
                          evals=[(Test, "Yes")],
                          verbose_eval=50)

        # PRedictions
        pred = model.predict(Test)
        pred = np.where(pred > 0.5, 1, 0)

        confusion_matrix = confusion_matrix(y_test, pred)
        report = classification_report(y_test, pred)
        print(report)
