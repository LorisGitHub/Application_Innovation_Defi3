# Usage

First you should set your xml files into the Data folder at the source of the repository.

## SVM

To run, you should just run the `main.py` files in the SVM folder. You can use `main.py -h` to see the arguments that the program takes. The program will produce `.svm` files which can be used by the `train.py` and `predict.py` of liblinear.
When liblinear finished, it will create a file named `out.txt` in the SVM folder. YOu can then use `reparse.py` which will read from the test files to create a new `realOutput.txt` files which can be uploaded to the platform.

## Keras

To use Keras, you should first generate your models, to do so, you can use the following two files `word2vec.py` and `fasttext.py`. It will generate `.model` file in Models directory which will be imported by the other files.

To train standard neural network, you have two files, `wordEmbeddingWord2vec.py` and `wordEmbeddingFastText.py` which will create a `.h5` files which contain the best model accuracy.
TO predict, you should use the corresponding predicts python files, `word2vecpredict.py` and `fasttext.py` which will in turn create a .txt in the Results directory which can be upload to the platform.

To train CCN, you we only made a file for fasttext, `wordEmbeddingFastTextCNN.py`, who will do both train and predict in one go and stock the results in the Results directory.

We have made an improvement by stocking input data of the model into `.pickle` files which are stored in the PreparedData folder. It allow train program to run without reparsing the while corpus.
