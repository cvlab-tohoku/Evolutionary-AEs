# Requirement

* Pytorch
* Ubuntu 14.04 LTS
* CUDA version 8.0
* Python version 3.6.2

---
## Usage

### Run the architecture search

If you run with the ResSet as the function set, you go to Evolutionary-AEs folder and

```shell
python exp_main.py -f ResSet
```

Or if you run with the ConvSet:

```shell
python exp_main.py -f ConvSet
```

Or if you run with the Primitive function set, you go to Evolutionary-AEs-primitive folder and

```shell
python exp_main.py -f Primitive
```
Primitive set does not contain ConvBlock or ResBlock.

When you use the multiple GPUs, please specify the `-g` option:

```shell
python exp_main.py -f ConvSet -g 2
```

After the execution, the files, `network_info.pickle` and `log_cgp.txt` will be generated. The file `network_info.pickle` contains the information for Cartegian genetic programming (CGP) and `log_cgp.txt` contains the log of the optimization and discovered CNN architecture's genotype lists.

Some parameters (e.g., # rows and columns of CGP, and # epochs) can easily change by modifying the arguments in the script `exp_main.py`.

### Re-training

The discovered architecture is re-trained by the different training scheme (500 epoch training with momentum SGD) to polish up the network parameters. All training data are used for re-training, and the accuracy for the test data set is reported.

```shell
python exp_main.py -m retrain
```
