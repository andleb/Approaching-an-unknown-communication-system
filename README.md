# Approaching an unknown communication system by latent space exploration and causal inference

[This repository](https://github.com/Neurips2023Submission/Neurips2023Submission/) is the supplement submission of the above paper.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials.

<img src="figs/SchematicCDEV.png" width=50% height=50%>>



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...



## The `fiwGAN` model

The `fiwGAN` architecture implementation in `pytorch` is located in [ciwfiwgan](ciwfiwgan).  



Unfortunately, the data is not free to share; the training parameters, however, were the following:
```train
python train.py --fiw --num_categ 5 --datadir training_directory --logdir log_directory\
--num_epochs 8000 --slice_len 65536
```

Add `--cont last` to the end of the training statement to continue from the last found state  
Add `--cont epoch_number` to continue from the state corresponding to `epoch_number`


>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


### Trained Model

<!-- TODO: hyperparam / script -->

The trained model used to generate results can be obtained here and loaded with 

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Analysis

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

All content in this repository is licensed under the [MIT license](LICENSE).
