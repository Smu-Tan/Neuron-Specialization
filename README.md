# Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation, EMNLP 2024

The repository of the EMNLP 2024 paper "Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation", see [paper](https://aclanthology.org/2024.emnlp-main.374/).


## Neuron Specialization Training and Evaluation Example

### 1. Preparations:

  * [Download EC30 Fairseq train-val-test set data-bin](https://drive.google.com/file/d/1qHRFNU-helRLpHkr6rspqEZs0eDj-68l/view?usp=drive_link). We provide our pre-processed training-validation-test sets in Fairseq bin format, which is the easiest way to reproduce.

  After downloading, extract the dataset to ec30/fairseq-data-bin-sharded.


  * [mT-big Neuron Specialization Model Checkpoint](https://drive.google.com/file/d/1LF8BP-5HfN9j9LfME0Jz28ULAztEUyeu/view?usp=drive_link). We provide the mT-big Neuron Specialization Checkpoint, which corresponds to "Ours" in Table 2.

  After downloading, extract the checkpoint and move to scripts/mT-big-NS/checkpoints.

  * [mT-big Baseline Model Checkpoint](https://drive.google.com/file/d/1IYOO_-lkgBh05p6XMUM1SqdYFhqmLXP3/view?usp=drive_link). We provide mT-big Baseline Model Checkpoint, which corresponds to "mT-big" in Table 2.

  After downloading, extract the checkpoint and move to scripts/mT-big-baseline/checkpoints.


### 2. Direct Evaluation using downloaded checkpoints.

  * [Evaluation scripts of mT-big Baseline Model](https://github.com/Smu-Tan/Neuron-Specialization/tree/main/scripts/mT-big-baseline/scripts/eval.sh). Scripts for evaluating mT-big Baseline Model on Flores test set.

  * [Evaluation script of mT-big Neuron Specialization Model](https://github.com/Smu-Tan/Neuron-Specialization/tree/main/scripts/mT-big-NS/scripts/eval.sh). Scripts for evaluating mT-big Neuron Specialization Model on Flores test set.
  
  
### 3. Neuron Specialization Training.

  * [Training scripts of mT-big Neuron Specialization Model](https://github.com/Smu-Tan/Neuron-Specialization/tree/main/scripts/mT-big-NS/scripts/train.sh). Scripts for training mT-big Neuron Specialization Model.


## Citation

For citation, please cite our Neuron Specialization paper (tan-etal-2024-neuron);

```
@inproceedings{tan-etal-2024-neuron,
    title = "Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation",
    author = "Tan, Shaomu  and
      Wu, Di  and
      Monz, Christof",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.374/",
}
```

If you use the EC30 dataset, please cite this as well (tan2023towards).

```
@inproceedings{tan2023towards,
  title={Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance},
  author={Tan, Shaomu and Monz, Christof},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={13553--13568},
  year={2023}
}
```