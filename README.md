# eval_kitti #

[![Build Status](https://travis-ci.org/cguindel/eval_kitti.svg?branch=master)](https://travis-ci.org/cguindel/eval_kitti)
[![License: CC BY-NC-SA](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/3.0/)

The *eval_kitti* software contains tools to evaluate object detection results using the KITTI dataset. The code is based on the [KITTI object development kit](http://www.cvlibs.net/datasets/kitti/eval_object.php). This version is useful to validate object detection approaches before submitting results to the official KITTI ranking.

### Tools ###

* *evaluate_object* is an improved version of the official KITTI evaluation that enables multi-class evaluation and splits of the training set for validation. It's updated according to the modifications introduced in 2017 by the KITTI authors.
* *parser* is meant to provide mAP and mAOS stats from the precision-recall curves obtained with the evaluation script (updated on November 2019 to reflect changes in the official evaluation)
* *create_link* is a helper that can be used to create a link to the results obtained with [lsi-faster-rcnn](https://github.com/cguindel/lsi-faster-rcnn).

### Usage ###
Build *evaluate_object* with CMake:
```
mkdir build
cd build
cmake ..
make
```

The `evaluate_object` executable will be then created inside `build`. The following folders are also required to be placed there in order to perform the evaluation:

* `data/object/label_2`, with the KITTI dataset labels.
* `lists`, containing the  `.txt` files with the train/validation splits. These files are expected to contain a list of the used image indices, one per row, [following these examples](https://github.com/cguindel/lsi-faster-rcnn/tree/master/data/kitti/lists).
* `results`, in which a subfolder should be created for every test, including a second-level `data` folder with the resulting `.txt` files to be evaluated.

### Typical evaluation pipeline ###
1. Get results using the object detector that you want to evaluate. Results must follow the KITTI format; please refer to the [KITTI object detection **devkit**](http://www.cvlibs.net/datasets/kitti/eval_object.php) for further information. You should use a different set of frames for training and validation. As a result, you will have a folder with thousands of txt files, one per validation frame, and typically with non-correlative filenames.
2. Create a folder within `eval_litti/build/results` with the name of the experiment. Create a symbolic link to the folder containing your resulting txt files inside the newly created folder and rename it as `data`; e.g., if the experiment is named `exp1`, txt files to be evaluated should be accessible at `eval_litti/build/results/exp1/data`.
3. Go to `eval_kitti/build` and run `evaluate_object` with the name of the experiment and the txt file containing the validation split as arguments. For instance, if you created the folder `eval_litti/build/results/exp1` and are using the `valsplit` validation set from [here](https://github.com/cguindel/lsi-faster-rcnn/tree/master/data/kitti/lists), you should run ```./evaluate_object exp1 valsplit```.
5. Wait until the script is completed. Modify lines 8 and 11 of `eval_kitti/parser.py` to specify the classes and parameters for which you want average results; by default, you will obtain the same stats as in the KITTI benchmark. Then, go back to `eval_kitti` and run ```./parser.py``` passing the name of the experiment as an argument. In our example, ```python parser.py exp1```. Average stats (AP for detection, AOS for orientation, etc.) will be printed.

### Copyright ###
This work is a derivative of [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php) by A. Geiger, P. Lenz, C. Stiller and R. Urtasun, used under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/). Consequently, code in this repository is published under the same [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.
