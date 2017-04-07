# Classified

## Getting the dataset
On [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/FsdEHsJJ6oF7yiL) we
have put all the input CSV files. They need to be put into `./data/input/`.

## Running feature extraction
After the dataset has been put into `./data/input/`, one can simply run `python
Feature_extraction.py`. This has been tested with Anaconda 4.3.0 [64-bit
installer for Python 2.7](https://www.continuum.io/downloads).

We have prepared an Anaconda installation on `lilo5.science.ru.nl`, with a
checkout of our working code, including the data files. To use it, run:

```
$ cd /scratch/gmulder-pub/
$ export PATH=/scratch/gmulder-pub/anaconda/bin:${PATH}
$ cd repo/
$ python -V    # to check you actually run Anaconda
Python 2.7.13 :: Anaconda 4.3.0 (64-bit)
$ python Feature_extraction.py   # takes about 1 minute  on lilo5
$ python General_Pipeline.py     # takes about 4 minutes on lilo5
```

After that, in `./submission/`, there is a submission CSV for Kaggle.

## Output from Python scripts
If everything goes according to plan, you will find the following output:

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

## Cleaning up

When in `/scratch/gmulder-pub/repo/`, one can clean up the generated Pickle
files and submission CSV using `make clean`.
