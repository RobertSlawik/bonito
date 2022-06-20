#!/usr/bin/env python3

"""
Bonito model compression.
"""

import os
import csv
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from bonito.util import load_data, load_model, load_symbol, init, default_config, default_data, get_parameters_count
from bonito.training import ChunkDataSet, load_state, train, test, func_scheduler, cosine_decay_schedule, CSVLogger

import toml
import torch
import numpy as np
import torch.nn.utils.prune as prune
import torch.quantization as quantization
import torchprof
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import LSTM, Linear

import warnings

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training and overwrite files." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)
    config = toml.load(args.config)
    os.makedirs(workdir, exist_ok=True)

    # TODO(jasminequah)
    warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().")

    # Loading training & validation data



    # Writing config file to workdir
    argsdict = dict(training=vars(args))
    chunk_config = {}
    chunk_config_file = os.path.join(args.directory, 'config.toml')
    if os.path.isfile(chunk_config_file):
        chunk_config = toml.load(os.path.join(chunk_config_file))
    toml.dump({**config, **argsdict, **chunk_config}, open(os.path.join(workdir, 'config.toml'), 'w'))

    

    if args.quantize:
        # QUANTISATION (currently can only run on CPU (see torch documentation))

        device = torch.device('cpu')


        # Loading pretrained model
        assert(args.pretrained) # Can only compress pretrained model
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False, weights=args.weights)
        optimizer = AdamW(model.parameters(), amsgrad=False, lr=args.lr)
        torch.save(model.state_dict(), os.path.join(workdir, "weights.orig.tar"))
        criterion = model.seqdist.ctc_loss if hasattr(model, 'seqdist') else None

        print("[Quantizing]")
        print('Here is the floating point version of this module:')
        print(model)
        print('')
        quantized_lstm = quantization.quantize_dynamic(model, {LSTM, Linear}, dtype=torch.qint8)
        print('and now the quantized version:')
        print(quantized_lstm)

        # compare the sizes
        f=print_size_of_model(model,"fp32")
        q=print_size_of_model(quantized_lstm,"int8")
        print("{0:.2f} times smaller".format(f/q))


        torch.save(quantized_lstm.state_dict(), os.path.join(workdir, "quantized_weights_final.tar"))



    else: 
        # PRUNING
         # Loading training & validation data

        print("[loading data]")
        train_data = load_data(limit=args.chunks, directory=args.directory)
        if os.path.exists(os.path.join(args.directory, 'validation')):
            split = np.floor(len(train_data[0]) * 0.25).astype(np.int32)
            train_data = [x[:split] for x in train_data]
            valid_data = load_data(limit=args.val_chunks, directory=os.path.join(args.directory, 'validation'))
        else:
            print("[validation set not found: splitting training set]")
            split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
            valid_data = [x[split:] for x in train_data]
            train_data = [x[:split] for x in train_data]
        train_loader = DataLoader(ChunkDataSet(*train_data), batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(ChunkDataSet(*valid_data), batch_size=args.batch, num_workers=4, pin_memory=True)

        # Loading pretrained model
        assert(args.pretrained) # Can only compress pretrained model
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False, weights=args.weights)
        optimizer = AdamW(model.parameters(), amsgrad=False, lr=args.lr)
        torch.save(model.state_dict(), os.path.join(workdir, "weights.orig.tar"))
        criterion = model.seqdist.ctc_loss if hasattr(model, 'seqdist') else None
        
        if args.profile:
            with torchprof.Profile(model, use_cuda=True, profile_memory=False) as prof:
                test(model, device, valid_loader, criterion=criterion)
            
            print(prof.display(show_events=False))
            
        else:
            val_loss, val_mean, val_median = test(model, device, valid_loader, criterion=criterion)
            print("\n[start] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(workdir, val_loss, val_mean, val_median))
            with open(os.path.join(workdir, 'accuracy.txt'), 'w') as accuracy_log:
                accuracy_log.write("[start] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(workdir, val_loss, val_mean, val_median))



            print("[Pruning]")   
            for pruning_iter in range(1, args.pruning_iterations + 1):
                # Pruning
                print("Before pruning, model has %d params\n" % get_parameters_count(model))
                parameters_to_prune = model.get_parameters_to_prune()
                pruning_amount = 1 - (1 - args.prune_level) ** pruning_iter
                print("Pruning amount: %.3f" % pruning_amount)
                if args.structured:
                    for module, param in parameters_to_prune:
                        prune.ln_structured(module, param, amount=args.prune_level, n=1, dim=0)
                else:
                    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=args.prune_level)

                print("After pruning, model has %d params\n" % get_parameters_count(model))

                # Finetuning pruned model between iterations
                lr_scheduler = func_scheduler(
                    optimizer, cosine_decay_schedule(1.0, 0.1), args.epochs * len(train_loader),
                    warmup_steps=500, start_step=last_epoch*len(train_loader)
                )

                val_loss, val_mean, val_median = test(model, device, valid_loader, criterion=criterion)
                print("\n[prune {}] [untuned] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(pruning_iter, workdir, val_loss, val_mean, val_median))
                with open(os.path.join(workdir, 'accuracy.txt'), 'a') as accuracy_log:
                    accuracy_log.write("\n[prune {}] [untuned] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(pruning_iter, workdir, val_loss, val_mean, val_median))

                for epoch in range(1 + last_epoch, args.epochs + 1 + last_epoch):
                    try:
                        with CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                            train_loss, duration = train(
                                model, device, train_loader, optimizer, criterion=criterion,
                                use_amp=args.amp, lr_scheduler=lr_scheduler,
                                loss_log = loss_log
                            )

                        torch.save(model.state_dict(), os.path.join(workdir, "weights_%s_%s.tar" % (pruning_iter, epoch)))

                        val_loss, val_mean, val_median = test(
                            model, device, valid_loader, criterion=criterion
                        )
                    except KeyboardInterrupt:
                        break

                    print("\n[prune {}] [epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                        pruning_iter, epoch, workdir, val_loss, val_mean, val_median
                    ))

                    with open(os.path.join(workdir, 'accuracy.txt'), 'a') as accuracy_log:
                        accuracy_log.write("\n[prune {}] [epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                            pruning_iter, epoch, workdir, val_loss, val_mean, val_median
                        ))

                    with CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                        training_log.append(OrderedDict([
                            ('time', datetime.today()),
                            ('duration', int(duration)),
                            ('pruning_iter', pruning_iter),
                            ('epoch', epoch),
                            ('train_loss', train_loss),
                            ('validation_loss', val_loss),
                            ('validation_mean', val_mean),
                            ('validation_median', val_median)
                        ]))

                torch.save(model.state_dict(), os.path.join(workdir, "weights_prune_%s.tar" % pruning_iter))

            # Making pruned parameterisation permanent
            for module, param in parameters_to_prune:
                prune.remove(module, param)

            # prep_for_save() follows this: https://github.com/pytorch/pytorch/issues/33618
            model.prep_for_save()

            torch.save(model.state_dict(), os.path.join(workdir, "weights_final.tar"))
            print("After pruning, model has %d params\n" % get_parameters_count(model))

            # Sparsifying
            model_state = model.state_dict()
            for param_tensor in model_state:
                model_state[param_tensor] = model_state[param_tensor].to_sparse()
            torch.save(model_state, os.path.join(workdir, "weights_final_sparse.tar"))


   



def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--directory", default=default_data)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--val_chunks", default=1000, type=int)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--pretrained", default="dna_r9.4.1@v3.2")
    parser.add_argument("--weights", default="0",type=str) # Suffix of weights file to use
    parser.add_argument("--prune_level", default=0.6, type=float)
    parser.add_argument("--structured", action="store_true", default=False)
    parser.add_argument("--pruning_iterations", default=1, type=int)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    return parser
