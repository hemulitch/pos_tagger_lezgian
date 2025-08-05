# training script
import torch
import random
import numpy as np

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model():
    set_seed(42)

    columns = {0: 'text', 1: 'pos'}
    data_folder = 'pos_data'

    corpus = ColumnCorpus(
        data_folder,
        columns,
        train_file='train.txt',
        test_file='test.txt',
        dev_file='dev.txt'
        )
    
    label_type = 'pos'
    tag_dictionary = corpus.make_label_dictionary(label_type=label_type)
    
    embedding = TransformerWordEmbeddings(
        model="xlm-roberta-base",
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True
    )

    
    tagger = SequenceTagger(hidden_size=256,
                        embeddings=embedding,
                        tag_dictionary=tag_dictionary,
                        tag_type=label_type,
                        use_crf=True,
                        use_rnn=False)

    trainer = ModelTrainer(tagger, corpus)
    trainer.fine_tune(base_path='model', 
                  learning_rate=3e-5,
                  mini_batch_size=4, 
                  max_epochs=10,
                  embeddings_storage_mode='none',
                  shuffle=True,
                  train_with_dev=True
                  )


if __name__ == '__main__':
    train_model()

