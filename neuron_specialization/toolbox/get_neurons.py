#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")



from collections import defaultdict

def main(cfg: DictConfig, override_args=None, save_neuron_path=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)



    forward_hooks = []
    layer_inputs, layer_outputs = defaultdict(defaultdict), defaultdict(defaultdict)
    def layer_forward_hook(layer_name, sub_layer_name):
        def hook(module, inp, out):
            layer_inputs[layer_name][sub_layer_name]=inp[0].data.cpu()
            layer_outputs[layer_name][sub_layer_name]=out.data.cpu()
            del inp, out
        return hook

    def register_activation_hooks(model, forward_hooks, which_coder, which_module):
        enc_layers = getattr(model, 'encoder').layers
        dec_layers = getattr(model, 'decoder').layers

        for focus_layers in [enc_layers, dec_layers]:
            if focus_layers == enc_layers:
                which_coder = 'encoder'
            else:
                which_coder = 'decoder'
            for i, _ in enumerate(focus_layers): #model.model.encoder.layers
                for sub_layer_name, child_module in focus_layers[i].named_children():
                    if sub_layer_name != 'self_attn' and 'dropout_module' not in sub_layer_name:
                        if which_module in sub_layer_name:
                            layer_name = f"{which_coder}_layer_{i}"
                            forward_hook = child_module.register_forward_hook(layer_forward_hook(layer_name, sub_layer_name))
                            forward_hooks.append(forward_hook)  # 将挂钩添加到列表中
        return

    # register hooks
    register_activation_hooks(model, forward_hooks, which_coder='encoder', which_module='fc2')
    enc_layer_names = ["encoder_layer_"+str(i) for i in range(len(model.encoder.layers))]
    dec_layer_names = ["decoder_layer_"+str(i) for i in range(len(model.decoder.layers))]
    layer_names = enc_layer_names+dec_layer_names
    ffn_dim = model.encoder.layers[0].fc1.out_features
    alive_neurons = {i:torch.zeros([ffn_dim]).long() for i in layer_names}

    def neuron_count(layer_inputs):
        temp = {}
        for l in layer_inputs:
            temp[l] = torch.nonzero(layer_inputs[l]['fc2']) 
            temp[l] = temp[l][:,-1] #first column is the idx we dont need
            temp[l] = torch.bincount(temp[l], minlength=ffn_dim)
        return temp

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

            temp = neuron_count(layer_inputs)
            alive_neurons = {j:alive_neurons[j]+temp[j] for j in alive_neurons}

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

    import pickle
    with open("{}/{}/activations.pkl".format(save_neuron_path,saved_cfg.task.lang_pairs[0]), "wb") as pickle_file:
            pickle.dump(alive_neurons, pickle_file)

def cli_main():

    parser = options.get_validation_parser()
    parser.add_argument('--save_neuron_path', type=str, help='save_neuron_path')
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_parser.add_argument('--save_neuron_path', type=str, help='save_neuron_path')
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args,save_neuron_path=args.save_neuron_path
    )


if __name__ == "__main__":
    cli_main()