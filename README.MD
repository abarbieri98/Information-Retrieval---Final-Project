# Information Retrieval - Final Project

## Project: Implementation of the BM25 IR system

## Andrea Barbieri - A.Y. 2022/2023

### Folder content

The content of the folders are the following:

- `moviesummaries`: folder containing the pre-computed indexes, tf and idf scores in pickle format, needed in the file `example.py`.
- `ProbIR.py`: file containing all the functions and classes implemented for the project.
- `example.py`: an example file based on the [Wiki Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset. 

### Usage of `example.py`

First, the CSV file downloadable at the link above must be placed inside the `moviesummaries` folder.

Then the program can be run easily using
```bash
    python example.py
```

The file will first read the documents and import the needed objects (the dictionaries inside `moviesummaries`) to initialize the IR system. Then the `query` method will be called, letting the user searching a query of choice. The retrieved documents will then be printed to the user, allowing for relevance feedback by answering "n" to the proposed question. The program will go on with the relevance feedback until the user is satisfied, then it will print the final list of retrieved documents.

Query function can be modified in various parameters, to see more please check the help for the `query` method.

### Usage of `example.py` with other corpora

To use `example.py` with other corpora one should first import it and create a list of object of type `Document` (defined inside the `ProbIR.py` file). Then, one can initialize the system by calling `IR = PIR.ProbIR.from_corpus([name of the list of Documents])`, which will automatically compute the needed objects. The program will then go on as explained above.