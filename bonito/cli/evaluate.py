"""
Bonito model evaluator
"""

import os
import time
import torch
import numpy as np
from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.training import ChunkDataSet
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_data, load_model, concat, permute, get_parameters_count

from torch.utils.data import DataLoader

import warnings


def main(args):

    # TODO(jasminequah)
    warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().")

    poas = []
    init(args.seed, args.device)

    print("* loading data")

    directory = args.directory
    if os.path.exists(os.path.join(directory, 'validation')):
        directory = os.path.join(directory, 'validation')

    testdata = ChunkDataSet(
        *load_data(
            limit=args.chunks, directory=directory
        )
    )
    dataloader = DataLoader(testdata, batch_size=args.batchsize, pin_memory=True)
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=args.min_coverage)

    for w in args.weights.split(','):

        seqs = []

        print("* loading model", w)
        model = load_model(args.model_directory, args.device, weights=w)
        params_count = get_parameters_count(model)

        print("* calling")
        t0 = time.perf_counter()

        with torch.no_grad():
            for data, *_ in dataloader:
                if half_supported():
                    data = data.type(torch.float16).to(args.device)
                else:
                    data = data.to(args.device)

                log_probs = model(data)

                if hasattr(model, 'decode_batch'):
                    seqs.extend(model.decode_batch(log_probs))
                else:
                    seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

        duration = time.perf_counter() - t0

        refs = [decode_ref(target, model.alphabet) for target in dataloader.dataset.targets]

        accuracies = []
        insertions = []
        deletions = []
        substitutions = []
        len_seq_evals = []
        mean_seq_length = 0

        with open('bonito_eval.txt', 'w') as bonito_eval:
            for ref, seq in zip(refs, seqs):
                bonito_eval.write("%s\n%s\n" % (ref, seq))
                if len(seq):
                    acc, insertion, deletion, substitution, len_seq_eval = accuracy_with_cov(ref, seq)
                    accuracies.append(acc)
                    if acc != 0.0:
                        insertions.append(insertion)
                        deletions.append(deletion)
                        substitutions.append(substitution)
                        len_seq_evals.append(len_seq_eval)
                    mean_seq_length += len(ref)
                else:
                    accuracies.append(0.)
                    mean_seq_length += len(ref)
        mean_seq_length /= len(refs)

        if args.poa: poas.append(sequences)

        print("* mean      %.3f%%" % np.mean(accuracies))
        print("* median    %.3f%%" % np.median(accuracies))
        print("* mean ins  %.3f%%" % np.mean(insertions))
        print("* mean dels %.3f%%" % np.mean(deletions))
        print("* mean subs %.3f%%" % np.mean(substitutions))
        print("* seq eval  %.3f%%" % np.mean(len_seq_evals))
        print("* mean len  %.3f" % mean_seq_length)
        print("* time      %.3f" % duration)
        print("* samples/s %.3E" % (args.chunks * data.shape[2] / duration))
        print("* # params  %s" % params_count) # Number of non-zero parameters to measure model sparsity

    if args.poa:

        print("* doing poa")
        t0 = time.perf_counter()
        # group each sequence prediction per model together
        poas = [list(seq) for seq in zip(*poas)]
        consensuses = poa(poas)
        duration = time.perf_counter() - t0
        accuracies = list(starmap(accuracy_with_coverage_filter, zip(references, consensuses)))

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--directory", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=1000, type=int)
    parser.add_argument("--batchsize", default=96, type=int)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser
