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

1. Get the dataset from the link above
2. In `data/` run `./unpack.sh` (alternatively, recursively unzip the dataset)
3. In the base directory, run `source init.sh` if you have not yet done so
4. In `src/` run `python naive_data_processing.py` 

### Executing our pipeline

In `/scratch/gmulder-pub/repo/src/` on lilo5, you can find our pipeline.

### Executing different classifiers

We have two classifiers that were tested, but in the end not used. They can be
found in `src/` and run through:

```
$ python keras-test.py          # we used Keras==2.0.4
```

and


```
$ python sgdclassifier.py
```
