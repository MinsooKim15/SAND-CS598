from mimic3-benchmarks.mimic3benchmark.readers import InHospitalMortalityReader
from mimic3-benchmarks.mimic3models.preprocessing import Discretizer, Normalizer
from mimic3-benchmarks.mimic3models import metrics
from mmimic3-benchmarks.imic3models import keras_utils
from mimic3-benchmarks.mimic3models import common_utils
import argparse
import os
from mimic3-benchmarks.mimic3models.in_hospital_mortality import utils
from custom.readerToDataLoader import *
from utils.trainer import *
from core.model import *
import torch.optim as optim
from torch import *
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
def test_in_hospital_mortality():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_heads', type=int, default=n_heads, help='Number of attention heads')
    parser.add_argument('--factor', type=int, default=factor, help='Factor to multiply the hidden size by')
    parser.add_argument('--num_class', type=int, default=num_class, help='Number of output classes')
    parser.add_argument('--num_layers', type=int, default=num_layers, help='Number of transformer layers')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
    args = parser.parse_args()
    n_heads = args.n_heads
    factor = args.factor
    num_class = args.num_class
    num_layers = args.num_layers
    learning_rate = args.learning_rate

    train_reader = InHospitalMortalityReader(dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/train',
                                             listfile='mimic3-benchmarks/data/in-hospital-mortality/train_listfile.csv',
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/train',
                                           listfile='mimic3-benchmarks/data/in-hospital-mortality/val_listfile.csv',
                                           period_length=48.0)

    discretizer = Discretizer(timestep=1.0,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    test_reader = InHospitalMortalityReader(dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/test',
                                           listfile='mimic3-benchmarks/data/in-hospital-mortality/test_listfile.csv',
                                           period_length=48.0)
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    train_raw = utils.load_data(train_reader, discretizer, normalizer=None)
    trainLoader = readerToDataLoader(train_raw)
    N, seq_len, in_feature = train_raw[0].shape
    val_raw = utils.load_data(val_reader, discretizer, normalizer=None)
    valLoader = readerToDataLoader(val_raw)
    test_raw = utils.load_data(test_reader, discretizer, normalizer=None)
    testLoader = readerToDataLoader(test_raw)
    # in_feature = 23
    # seq_len = 256
    # n_heads = 32
    # factor = 12  # M
    # num_class = 2
    # num_layers = 4  # N 6
    # 1e-5
    clf = NeuralNetworkClassifier(
        SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers, dropout_rate=0.4),
        nn.CrossEntropyLoss(),
        optim.Adam, optimizer_config={"lr": learning_rate, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}
    )

    # training network
    clf.fit(
        {"train": trainLoader,
         "val": valLoader},
        epochs=10
    )

    # evaluating
    clf.evaluate(testLoader)
    # save
    clf.save_to_file("save_params/")