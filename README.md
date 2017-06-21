# Classified

* [Dataset](https://surfdrive.surf.nl/files/index.php/s/K2FYXiWVb8B9yMH),

## How to run our code

On lilo5.science.ru.nl, we have prepared a working Anaconda2 4.3.1 64-bit
installation for you. It is accessible in the following way:

```
$ cd /scratch/gmulder-pub/repo/
$ source init.sh                  # this makes sure you have the correct PATH
$ python -V
Python 2.7.13 :: Anaconda custom (64-bit)
```

After you're done, you can simply run:

```
$ cd /scratch/gmulder-pub/repo/
$ source reset.sh                 # this makes sure you reset your PATH
$ python -V
Python 2.7.12
```

The data files are located in the `data/` directory, the source code in `src/`.
We have already prepared the data set for you. If you want test our
pre-processing, please check the next section, otherwise skip to _Executing our
pipeline_.

### Pre-processing

1. Get the dataset from the link above and store it in `data/`
2. In `data/` run `./unpack.sh` (alternatively, recursively unzip the dataset)
3. In the base directory, run `source init.sh` if you have not yet done so
4. In `src/` run `python naive_data_processing.py` 
5. In `src/` run `python construct_validation_set.py`

### Executing our pipeline

In `/scratch/gmulder-pub/repo/src/` on lilo5, you can find our pipeline. You
can run it through:

```
$ time python xgboost-reweighted.py
[..]

real    4m36.191s
user    27m21.308s
sys     67m9.252s
```

### Executing different classifiers

We have a couple of classifiers that were tested, but in the end not used. They
can be found in `src/` and run through:

 ```
$ time python keras-test.py     # we used Keras==2.0.4
                                # init.sh places a correct keras.json in your
                                # home folder, and reset.sh restores the backup
[..]
submission saved to ../submission/DL_sub.csv

real    16m37.138s
user    292m41.788s
sys     7m47.812s
```

```
$ time python sgdclassifier.py
[..]

real    2m5.607s
user    2m8.776s
sys     0m0.812s
```

```
$ time python Naive\ Classifiers.py
[..]
Linear Regresion Cross Validation Score
0.524993062908
Decision Tree Classifier Cross Validation Score
0.594144853645
K Neighbors Cross Validation Score
0.931221351349
Random Forest Classifier Cross Validation Score
0.786182070125
Gradient Boosting Regressor Cross Validation Score
0.454981100656

real    0m25.136s
user    0m31.876s
sys     0m6.984s
```
