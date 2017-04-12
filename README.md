# Classified

## Getting the dataset
On [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/FsdEHsJJ6oF7yiL) we
have put all the input CSV files. They need to be put into `./data/input/`.

Since this is about 1.2GB, we can imagine this takes a while to download.
Therefore, we have prepared two Anaconda installations on `lilo5.science.ru.nl`,
with a checkout of our working code, including the data files.

## Preparation on lilo5.science.ru.nl
Before you can run our code, please do the following on `lilo5.science.ru.nl`:

```
$ cd /scratch/gmulder-pub/
$ export ORIGINAL_PATH=${PATH}                           # backup the original PATH
$ export PATH=/scratch/gmulder-pub/anaconda/bin:${PATH}  # add Anaconda2
$ export PATH=/scratch/gmulder-pub/anaconda3/bin:${PATH} # add Anaconda3
$ cd repo/
$ python2 -V    # to check you actually run Anaconda2
Python 2.7.13 :: Anaconda 4.3.0 (64-bit)
$ python3 -V    # to check you actually run Anaconda3 with Python3.5.3
Python 3.5.3 :: Anaconda custom (64-bit)
```

After you're done with running the code, you can restore the PATH variable
using `export PATH=${ORIGINAL_PATH}`, or simply logout and login again.

## Running feature extraction
After the dataset has been put into `./data/input/`, one can simply run `python
Feature_extraction.py`.

```
$ python2 Feature_extraction.py   # takes about 1 minute  on lilo5

```

## Pipelines

### General pipeline

First run `Feature_extraction.py` (see above), then:

```
$ python2 General_Pipeline.py     # takes about 4 minutes on lilo5
```

After that, in `./submission/`, there is a submission CSV for Kaggle.

### Deep learning pipeline

We have prepared a Python 3.5.3 installation with Keras 1.2.1 on lilo5
as well. First run `Feature_extraction.py` (see above), then:

```
$ cd other_pipelines/
$ python3 PipelineDL.py
```

**Caveat**: When running this for testing purposes on lilo5.science.ru.nl,
the process ran for about ~3~ 4 CPU hours without results (see the output
of `script` here: `/scratch/gmulder-pub/repo/other_pipelines/pdl.log`. The
process might still be running when you read this. Its output will be
appended to the `pdl.log` log file). This might have to do with inconsistencies
with the machine on which the code was developed and lilo5.science.ru.nl

### XGBoost pipeline

First run `Feature_extraction.py` (see above), then:

```
$ cd other_pipelines/
$ python2 Pipeline\ XGB.py     # this takes about X minutes on lilo5
```

(initially, XGBoost missed a library, this was [fixed with
`conda install libgcc`](https://github.com/dmlc/xgboost/issues/1043#issuecomment-255603019)
that updates the needed libraries)

After it is done, the submission CSV can be found in `./submission/sparse_xgb.csv`.
At the moment of writing, it is still running. The current
output can be found in `./other_pipelines/xg.log`.

## Cleaning up

When in `/scratch/gmulder-pub/repo/`, one can clean up the generated Pickle
files and submission CSV using `make clean`.

## Old files

If you're interested in the files we used during development, please have
a look at [this point in time](https://github.com/RoelBouman/Classified/tree/60eb72c57a1b12fbd52c6205e56b363fd15a6780)
in our repository.

## Output from Python scripts
If everything goes according to plan, you will find the following
console output:

```
$ python Feature_extraction.py 
importing libraries
loading data
extracting features
Brand features: train shape (74645, 131), test shape (112071, 131)
Model features: train shape (74645, 1667), test shape (112071, 1667)
Apps data: train shape (74645, 19237), test shape (112071, 19237)
Labels data: train shape (74645, 492), test shape (112071, 492)
```

```
$ python General_Pipeline.py 
start iteration 0
end iteration 1
start iteration 1
end iteration 2
start iteration 2
end iteration 3
start iteration 3
end iteration 4
start iteration 4
end iteration 5
start iteration 5
end iteration 6
start iteration 6
end iteration 7
start iteration 7
end iteration 8
start iteration 8
end iteration 9
start iteration 9
end iteration 10
starting final prediction
```

```
$ python3 PipelineDL.py
Using TensorFlow backend.
start iteration 0
Epoch 1/3
# [..] a couple of warning regarding CPU instructions
67176/67176 [==============================] - 51s - loss: 2.3503 - acc: 0.1613 - val_loss: 2.3664 - val_acc: 0.1633  
Epoch 2/3
67176/67176 [==============================] - 47s - loss: 2.2684 - acc: 0.1951 - val_loss: 2.3609 - val_acc: 0.1591
Epoch 3/3
67176/67176 [==============================] - 46s - loss: 2.2340 - acc: 0.2082 - val_loss: 2.3621 - val_acc: 0.1580
[..]
```

```
$ python2 Pipeline\ XGB.py

```
