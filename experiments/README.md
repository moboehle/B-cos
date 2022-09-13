# Training 
Each of the experiment configurations follows the same structure, i.e., it is contained in a folder with the name of 
the dataset (e.g., CIFAR10 / Imagenet) and then in a subfolder to cluster types of experiments (with different architectures for example).

In order to run an experiment, you can use the following command: 
```
python experiments/Imagenet/bcos/model.py --single_epoch=true --experiment_name=resnet_34 --dataset_name=Imagenet --model_config=bcos
``` 
This would run the experiment `resnet_34`specified in `experiments/Imagenet/bcos/experiment_parameters.py` for a single epoch.
In order to pick up from a previous checkpoint, use the following:  
```
python experiments/Imagenet/bcos/model.py --continue_exp=true --experiment_name=resnet_34 --dataset_name=Imagenet --model_config=bcos
``` 
Per default, the results are saved in the current directory. In order to specify an output directory, use the `--base_path` option.

For more details, check the argument parser that is used in `experiments/Imagenet/bcos/model.py`
