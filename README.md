# eval_kitti #

The *eval_kitti* software contains tools to evaluate object detection results using the KITTI dataset.

### Tools ###

* *evaluate_object* is an improved version of the official KITTI evaluation that enables multi-class evaluation and splits of the training set for validation. It's updated according to the modifications introduced in 2017 by the KITTI authors.
* *parser* is meant to provide mAP and mAOS stats from the precision-recall curves obtained with the official evaluation script.
* *create_link* is a helper that can be used to create a link to the results obtained with [lsi-faster-rcnn](https://github.com/cguindel/lsi-faster-rcnn).

### Usage ###
Build **evaluate_object** with CMake:
```
mkdir build
cd build
cmake ..
make
```

The following folders are required to be placed in build:
* `data/object/label_2`, with the KITTI dataset labels.
* `lists`, containing the  `.txt` files with the train/validation splits.
* `results`, in which a subfolder should be created for every test, including a second-level `data` folder with the resulting `.txt` files.

*evaluate_object* should be called with the name of the results folder and the validation split; e.g.: ```./evaluate_object leaderboard valsplit ```

*parser* needs the results folder; e.g.: ```./parser.py leaderboard ```. **Note**: *parser* will only provide results for *Car*, *Pedestrian* and *Cyclist*; modify it (line 8) if you need to evaluate the rest of classes.  
