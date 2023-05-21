# Approaching an unknown communication system by latent space exploration and causal inference

[This repository](https://github.com/Neurips2023Submission/Neurips2023Submission/) is part of the supplement submission of the above paper.

<img src="figs/SchematicCDEV.png" width=50% height=50%>>

**Abstract:**
> <p align="justify"> <i> This paper proposes a methodology for discovering meaningful properties in data by exploring the latent space of unsupervised deep generative models. We combine manipulation of individual latent variables to extreme values with methods inspired by causal inference into an approach we call causal disentanglement with extreme values (CDEV) and show that this method yields insights for model interpretability. With this, we can infer what properties of unknown data the model encodes as meaningful, using it to glean insight into the communication system of sperm whales (Physeter macrocephalus), one of the most intriguing and understudied animal communication systems. The network architecture used has been shown to learn meaningful representations of speech; here, it is used as a learning mechanism to decipher the properties of another vocal communication system in which case we have no ground truth. The proposed methodology suggests that sperm whales encode information using the number of clicks in a sequence, the regularity of their timing, and audio properties such as the spectral mean and the acoustic regularity of the sequences. Some of these findings are consistent with existing hypotheses, while others are proposed for the first time. We also argue that our models uncover rules that govern the structure of  units in the communication system and apply them while generating innovative data not shown during training. This paper suggests that an interpretation of the outputs of deep neural networks with causal inference methodology can be a viable strategy for approaching data about which little is known and presents another case of how deep learning can limit the hypothesis space. Finally, the proposed approach can be extended to arbitrary architectures and datasets.</i></p>



## Requirements

To install requirements for 

```setup
pip install -r requirements.txt
```


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


## License

All content in this repository is licensed under the [MIT license](LICENSE).
