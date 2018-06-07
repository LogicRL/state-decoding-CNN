# state-decoding-CNN
CNN network for symbolic state monitoring, applied on MonteZuma's Revenge, achieved 99.96% accuracy (PyTorch version).

An example of this: 

<p align="center">
<img src="https://pic-markdown.s3.amazonaws.com/region=us-west-2&tab=overview/2018-05-27-040049.png" width="500" height="300" />
</p>

The CNN state decoder takes in raw image input and outputs list of detected states: 
```
['actorInRoom,room_1', 'actorOnSpot,room_1,conveyor_1']
```

The key feature of this implementation is that the network could dynamically increase the number of outputs to match the increasing number of states as the reinforcement learning goes deeper into the game.

Example usage: 
At the beggining, the agent only have 15 different states to detect, so use CLASSES = \[15\], 
then as the agent goes deeper into the game (a different room), it will have more states to detect, let's say it's 3 more states, then simply by using CLASSES = \[15, 3\] in CNNModel, the network is able to detect more states.

## Authorship

Below is the authorship information for this project.

  * __Author__:  Shangwu Yao
  * __Email__:   shangwuyao@gmail.com

Copyright (C) 2018, Shangwu Yao. All rights reserved.
