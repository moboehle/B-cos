<div style="margin-left:auto; margin-right:auto; padding:0; text-align:center">
  <img class="intro_gifs"  src="media/gifs/lorikeet.gif?raw=true"/>
  <img class="intro_gifs"  src="media/gifs/drake.gif?raw=true"/>
  <img class="intro_gifs"  src="media/gifs/zebra.gif?raw=true"/>
</div>
# Introduction
 [B-cos Nets on GitHub](https://github.com/moboehle/B-cos) 
| [B-cos Nets on YouTube (Computer Vision Seminar, MIT)](https://youtu.be/0NSSzXoWa2c) 


Today's deep neural network architectures achieve impressive performances on a wide array of different tasks.
In order to pull off these feats, they are trained 'end-to-end' on large amounts of data, which allow them 
to learn the most useful features for a given task on their own
instead of relying on hand-crafted features designed by domain experts.
While the possibility for end-to-end training has significantly boosted performance, this approach comes at a cost: 
    as the feature extraction process is highly unconstrained, 
    it is difficult to follow the decision making process throughout the network and thus hard
    to understand in hindsight how a deep neural network arrived at a particular decision.

With our work on **B-cos Networks**, we show that it is possible to train neural networks
end-to-end whilst constraining _how_ they are supposed to solve a task. By knowing the _how_, we can faithfully reflect
the decision making process and produce highly detailed explanations for the model predictions.
Despite these constraints, we show that B-cos Nets constitute performant classifiers. 

Feel free to contact us if you have questions or comments: [Author contact page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/moritz-boehle)

# Copyright and license
Copyright (c) 2021 Moritz Böhle, Max-Planck-Gesellschaft

This code is licensed under the BSD License 2.0, see [license](LICENSE).

Further, if you use any of the code in this repository for your research, please cite as:
```
  @inproceedings{Boehle2022CVPR,
          author    = {Moritz Böhle and Mario Fritz and Bernt Schiele},
          title     = {B-cos Networks: Alignment is All we Need for Interpretability},
          journal   = {IEEE/CVF Conference on Computer Vision and Pattern Recognition ({CVPR})},
          year      = {2022}
      }
```

