from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import util, losses

class SBERTTrainer():
    def __init__(self, df, output_dir, n_epochs=1):
        self.df = df
        self.output_dir = output_dir
        self.n_epochs = n_epochs

    def run(self):
            ft_train_df = self.df.loc[(self.df.label == 1)]

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
            warmup_steps = int(len(train_dataloader) * self.n_epochs * 0.1)
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples, name='sts-dev')

            title_model.fit(
                train_objectives=[(train_dataloader, loss)],
                evaluator=evaluator,
                epochs=self.n_epochs,
                evaluation_steps=1000,
                warmup_steps=warmup_steps,
                output_path=self.output_dir
            )
            return title_model
