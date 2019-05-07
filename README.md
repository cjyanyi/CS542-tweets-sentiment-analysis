# CS542-tweets-sentiment-analysis
   
### Background:
data resource: https://www.kaggle.com/kazanova/sentiment140   
This data set contains 1.6 million tweets texts extracted using the twitter API. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.
Goals:   
Design a NLP model that is able to predict the polarity (negative or positive) of the tweet.
Approach:   
Key Frameworks: Keras, TensorFlow, Flask   

### Step 1: Data Preprocessing
1: Tokenize the words in the dataset.   
2: Separate the data set into training set and testing set. 
3: Word Embedding.   
Using GLoVe https://nlp.stanford.edu/projects/glove/ to map a tokenized word to a fixed length vector.

### Step 2: Design and train a deep learning NLP model
1: General Models: TextCNN, TextRNN, TextRCNN. Design and train each model separately. 
2: Model Embedding: Optimizing the above models jointly, and use all the prediction results to make a final decision.   
TextCNN: https://arxiv.org/abs/1408.5882   
TextRNN: https://www.ijcai.org/Proceedings/16/Papers/408.pdf   

### Step 3: Demonstration
We will deploy our model as RESTful APIs, in order to make on-line prediction using the model we trained in Step2.

### Run codes
1: Create a folder 'dataset' under the same directory with 'CS542-tweets-sentiment-analysis'. Download the file 'training.1600000.processed.noemoticon.csv' from https://www.kaggle.com/kazanova/sentiment140 and put it in 'dataset'.   
2: Create a folder 'glove.6B' under the project directory, and download 'glove.6B.200d.txt' from https://nlp.stanford.edu/projects/glove/ .   
3: Run 'train.py' to train and save the model, after that run 'pred.py' to make prediction using the pre-trained model.
Run 'experiments.py' to reproduce the demo results. 
