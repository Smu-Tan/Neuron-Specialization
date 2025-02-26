import os
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
def nested_defaultdict():
    return defaultdict(list)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from fairseq.checkpoint_utils import torch_persistent_save
import argparse

def load_act(lang_pair,dir):
    with open(dir+'/{}/activations.pkl'.format(lang_pair), "rb") as pickle_file:
        act = pickle.load(pickle_file)
    return act

def calculate_iou_from_sets(set_a,set_b):
    # Calculate the Intersection and Union
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    # Calculate the Intersection over Union (IoU) score
    iou_score = len(intersection) / len(union)
    return iou_score

def cumulative_neuron_activation_indices_torch(activations, k):

    # Calculate the target number of activations (k percent of total)
    target_activations = activations.sum() * (k / 100)

    # Sort the activations in descending order and get the indices
    sorted_activations, indices = torch.sort(activations, descending=True)

    # Compute the cumulative sum of sorted activations
    cumsum_activations = torch.cumsum(sorted_activations, dim=0)

    # Find the index where cumulative sum reaches or exceeds the target activations
    target_index = torch.where(cumsum_activations >= target_activations)[0][0] + 1

    # Get the original indices of the neurons up to the target index
    neuron_indices = indices[:target_index]

    return neuron_indices

def load_neurons(l,dir):
    neurons = {}
    for i in l:
        pair = 'en-'+i
        neurons[pair] = load_act(pair,dir)
        pair = i+'-en'
        neurons[pair] = load_act(pair,dir)
    return neurons


def main(args):

    neuron_dir = args.neuron_dir
    save_mask_dir = args.save_mask_dir
    save_fig_dir = args.save_fig_dir
    mask_format_path = args.mask_format_path
    fc1_neuron_dim = args.fc1_neuron_dim
    fc1_weight_dim = args.fc1_weight_dim
    x2en_enc_k = args.x2en_enc_k
    x2en_dec_k = args.x2en_dec_k
    en2x_enc_k = args.en2x_enc_k
    en2x_dec_k = args.en2x_dec_k


    l = ['de','nl','sv','da','af','lb','fr','es','it','pt','ro','oc','ru','cs','pl','bg','uk','sr','hi','bn','kn','mr','sd','gu','ar','he','ha','mt','ti','am']
    neurons=load_neurons(l,neuron_dir)
    default = torch.arange(fc1_neuron_dim)
    top_k_neurons = defaultdict(nested_defaultdict)
    neurons_new = defaultdict(nested_defaultdict)

    for lp in neurons:
        for layer in neurons[lp]:
            src,tgt = lp.split('-')
            # en2x
            if src == 'en':
                if 'encoder' in layer:
                    x = cumulative_neuron_activation_indices_torch(neurons[lp][layer], en2x_enc_k)
                else:
                    x = cumulative_neuron_activation_indices_torch(neurons[lp][layer], en2x_dec_k)
            # x2en
            else:
                if 'encoder' in layer:
                    x = cumulative_neuron_activation_indices_torch(neurons[lp][layer], x2en_enc_k)
                else:
                    x = cumulative_neuron_activation_indices_torch(neurons[lp][layer], x2en_dec_k)
            top_k_neurons[lp][layer] = x
            neurons_new[lp][layer] = torch.isin(default, x).int()

    neuron_2_weight = defaultdict(nested_defaultdict)
    for lp in neurons_new:
        for layer in neurons_new[lp]:
            neuron_2_weight[lp][layer] =  neurons_new[lp][layer].unsqueeze(1).expand([fc1_neuron_dim,fc1_weight_dim]).bool()


    weight = defaultdict(nested_defaultdict)
    for lp in neuron_2_weight:
        lass = torch.load(mask_format_path)
        for key in list(lass.keys()):
            if 'fc1' in key:
                weight[lp][key] = neuron_2_weight[lp]['_'.join(key.split('.')[:3]).replace('layers', 'layer')]
            #if 'fc1' not in key:
            #    weight[lp][key] = torch.ones_like(lass[key])
            #else:
            #    weight[lp][key] = neuron_2_weight[lp]['_'.join(key.split('.')[:3]).replace('layers', 'layer')]

    for lp in weight:
        torch_persistent_save(dict(weight[lp]), "{}/{}.pt".format(save_mask_dir, lp))
    

    layers = list(neurons_new['en-de'].keys())
    lps = list(neurons_new.keys())

    df = pd.DataFrame(columns=['pair', 'sparsity', 'layer'])

    for layer in layers:
        for lp in lps:
            src,tgt = lp.split('-')
            if src=='en':
                type='en2x'
            else:
                type='x2en'
            s = (1-torch.count_nonzero(neurons_new[lp][layer])/fc1_neuron_dim).item()
            df = pd.concat([df,pd.DataFrame.from_dict([{'pair': lp, 
                                                        'type': type,
                                                        'layer': '_'.join(layer.split('oder_layer_')), 
                                                        'sparsity': s}])])

    sns.lineplot(df, x='layer', y='sparsity', hue='type')
    plt.xticks(rotation=45)
    plt.axvline(x = 5.5,  
                linestyle = "--", c='black') 
    plt.title('Sparsity of different fc1 layers over all language pairs')
    plt.savefig("{}/sparsity.png".format(save_fig_dir))



    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    

    parser.add_argument('--neuron-dir', type=str, help='neuron_dir')
    parser.add_argument('--mask-format-path', type=str, help='mask_format_path')
    parser.add_argument('--save-mask-dir', type=str, help='save_mask_dir')
    parser.add_argument('--save-fig-dir', type=str, help='save_fig_dir')
    

    parser.add_argument('--fc1-neuron-dim', type=int, help='fc1_neuron_dim')
    parser.add_argument('--fc1-weight-dim', type=int, help='fc1_weight_dim')
    parser.add_argument('--en2x-enc-k', type=int, help='save_mask_dir')
    parser.add_argument('--en2x-dec-k', type=int, help='save_mask_dir')
    parser.add_argument('--x2en-enc-k', type=int, help='save_mask_dir')
    parser.add_argument('--x2en-dec-k', type=int, help='save_mask_dir')


    args = parser.parse_args()

    main(args)