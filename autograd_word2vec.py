import autograd
from autograd import numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

def vocab_build(corpus):
    '''
    Params:
    corpus- Text corpus containing raw text extracted from the file

    Output: vocab- Word vocabulary or list of unique words
    Builds vocabulary of unique words.
    '''
    vocab = []
    for word in corpus.split():
        word = word.replace('.', '')
        word = word.replace(' ', '')
        #placing unique words into the word vocabulary
        if word not in vocab:
            vocab.append(word)
    return vocab

def word_to_int(vocab):
    '''
    Params: Vocab - Word Vocabulary.
    Output: word2int - Dictionary having word as a key and position of word as its value.

     Builds a dictionary which maps words to its position in the vocabulary
    '''
    word2int = {}
    for i, w in enumerate(vocab):
        word2int[w] = i

    return word2int

def one_hot_encoding(index,vocab_size):
    '''
    Params: index - Position of element in the list.
            vocab_size - Size of the elements containing list
    Output: One-hot encoded list with value 1 at position of the element

    Generates one-hot vector for given words-index
    '''
    temp = np.zeros(vocab_size)
    temp[index]= 1
    return temp

def init_nn_word_embedding(dims,vocab_size,random_state=npr.RandomState(0)):
    '''
    Params: dims - Dimension of embedded representation.
            vocab_size - Size of word vocabulary.
            random_state - To initialize the weights and bais parameter of the neural network.
    Output: params - Matrix of randomly initialized weights and bais of hidden layers.

    Initializies neural network parameter for word embedding.
    '''
    #initailizing the weight and bais parameter of the neural network
    init_weight = [random_state.randn(vocab_size,dims),random_state.randn(dims,vocab_size)] # Weight Matrix
    init_bais = [random_state.randn(dims), random_state.randn(vocab_size)] # Bais Matrix
    params = [[init_weight[0],init_bais[0]], [init_weight[1],init_bais[1]]] # Initailized parameters of word embedding neural network
    return params

def softmax(input):
    '''
    Params: input - Output of the last layer of neural network
            output - Array having softmax value output.

    Softmax activation function
    '''
    temp = np.exp(input)
    softmax = temp/np.sum(temp)
    return softmax

def predict(params, input):
    '''
    Params: params - Parameters of word embedding neural network.
            input - Encoded word on which model is going to perform prediction.

    Output - Predicted vectorized value.
    '''
    hidden_layer_1 = np.sum(np.matmul(input,params[0][0]),params[0][1]) # Hidden Layer
    hidden_layer_2 = softmax(np.sum(np.matmul(hidden_layer_1,params[1][0]),params[1][1])) # Output Layer with softmax activation
    return hidden_layer_2

def loss(params,input,labels):
    '''
    Params : Params: params - Parameters of word embedding neural network.
            input - Encoded word on which model is going to perform prediction.
            labels - Label associated with the word (here neighbouring words of input word)

    Output - Computes loss between predicted value and original labelled value

    Computes loss between value predicted by the model and given label (here neighbouring words of input word)
    which can be used as the metrics to computes performance of the model during training
    '''
    prediction = predict(params, input) # input prediction
    #computing loss between actual value and predicted value
    return -np.sum(np.log(prediction)*labels)/labels.shape[0]

def accuracy(params,input,labels):
    '''
    Params : Params: params - Parameters of word embedding neural network.
             input - Encoded word on which model is going to perform prediction.
            labels - Label associated with the word (here neighbouring words of input word)

    Output - Accuracy metric value to show how model is performing on given input.

    '''
    prediction = np.argmax(predict(params,input),axis=1)
    target = np.argmax(labels, axis=1)
    #total number of prediction equal target divided total number of samples which provide accuracy metrics
    return np.mean(prediction==target)

def train(grad, epochs, learning_rate, params, X_train, Y_train, X_test, Y_test, performance=True):
    '''
    Params: grad - Function to compute gradient for given params
            epochs - Number of iteration for training the model
            learning_rate - Hyperparameter that controls weights updation with respect to the gradient loss
            params - Initailised parameter of word embedding model
            X_train - List of encoded input words which used to train the word embedding model.
            Y_train - List of encoded neighbouring words corresponds to input word or list of labels
            X_test - Cross-validation set input
            Y_test - Cross-validation set labels.
            performance (optional) - Display performance of the model after each epoch if set True.

    Output: Optimized parameters of word embedding model.

    Trains word embedding model and updates it parameters to get better output.
    '''
    for i in range(epochs):
        for j in range(len(X_train)):
            #gradient calculation
            gradient = grad(params, X_train[j], Y_train[j])
            #model parameters updation
            for w in range(len(params)):
                params[w][0] = params[w][0] - learning_rate * gradient[w][0]
                params[w][1] = params[w][1] - learning_rate * gradient[w][1]
        #Display performance
        if(performance):
            print("epoch:" + str(i)+"/"+str(epochs))
            print("loss:" + str(np.mean(loss(params, X_train, Y_train))))
            print("Training Accuracy:"+str(accuracy(params, X_train, Y_train)))
            print("Testing Accuracy:"+str(accuracy(params, X_test, Y_test)))
            print('\n')
    return params

def euclidean_distance(vec1, vec2):
    '''
    Params: vec1 - Vector of first word
            vec2 - Vector of second word
    Output: Euclidean distance between the words on the basis of word vector
    '''
    return np.sqrt(np.sum((vec1-vec2)**2))

def nearest_word(word_index, vectors):
    '''
    Params: word_index - Index of word in word vocabulary
            vectors - Set of vector associated with embedded words
    Output: Vocabulary index of the closest word
    '''
    min_dist = 1e6 # pseudo-infinity
    min_index = -1
    #vector of query word
    w_vector = vectors[word_index]
    #looping over list of vector
    for index, vector in enumerate(vectors):
        #finding Euclidean distance between query word vector and other word vector
        if euclidean_distance(vector, w_vector) < min_dist and not np.array_equal(vector, w_vector):
            min_dist = euclidean_distance(vector, w_vector)
            min_index = index
    return min_index


def word2vec_model(corpus_file_path, dims, window_size, validation_split, epochs, learning_rate, performance=True):
    '''
    Params: corpus_file_path - Path of the file containing raw text required for training the model.
            dims - Dimension of embedded representation.
            window_size - Defines neighbourhood of a word.
            validation_split - Ratio to split data into training and validation.
            epochs - Number of iteration for training the model.
            learning_rate - Hyperparameter that controls weights updation with respect to the gradient loss
            performance (optional) - Display performance of the model after each epoch if set True.

    Output: Trained Word2Vec model parameter.

    Builds word embedding model from the given corpus of the raw data.
    '''

    f = open(corpus_file_path, 'r')
    corpus = f.read()
    corpus = corpus.lower()
    #Building Word Vocabulary from corpus of raw text data
    vocab = vocab_build(corpus)

    raw_sent = corpus.split('.  ')
    sent = []
    for i in raw_sent:
        i = i.replace('.', '')
        i = i.replace('  ', ' ')
        sent.append(i.split())
    data = []
    #Preparing dataset containing encoded word with its neighbouring words which is used to train word embedding model.
    for s in sent:
        for word_index, w in enumerate(s):
            #finding neighbouring word in the reference frame of the given word
            for nb_word in s[max(word_index - window_size, 0):min(word_index + window_size, len(s))]:
                if nb_word != w:
                    data.append([w, nb_word])

    X = []
    Y = []
    vocab_size = len(vocab)
    #data encoding
    for data_word in data:
        X.append(one_hot_encoding(word_to_int[data_word[0]], vocab_size))
        Y.append(one_hot_encoding(word_to_int[data_word[1]], vocab_size))

    #split dataset into training and testing
    training = int(len(X) * float(1-validation_split))
    X_train = np.array(X[:training])
    Y_train = np.array(Y[:training])
    X_test = np.array(X[training:])
    Y_test = np.array(Y[training:])
    #intializing parameters of word embedding neural network
    init_params = init_nn_word_embedding(dims, vocab_size)
    #gradient function
    gradloss = grad(loss)
    #training word embedding model on the given dataset
    optimized_params = train(gradloss, epochs, learning_rate, init_params, X_train, X_test,
                             Y_train, Y_test,performance=performance)
    return optimized_params