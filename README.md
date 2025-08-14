# CraigNeuralNetwork
An object oriented neural network for training any number of inputs to lead to any number of outputs.  Layers can be either Sigmoid or Tanh for their activations, so experiment with what works best for your training.  A Softmax Layer is also available (with categorical cross-entropy) but should only be used for the output Layer.

Static main() function is found in the NetworkRunner class.  Also found there are the configurable setup variables.

* numberOfEpochs = number of iterations to loop while training
* epochIteration = how often to print the network details during training
* learningRate = how big the weights and biases can change each epoch, the bigger the number the more drastic the leap
* layerCount = must match the length of the nodesCount int[] (probably not necessary anymore, but it's there so make it legit - this is a holdover from when I treated hidden Layers differently than input or output Layers)
* nodesCount = the list of Neurons per Layer - the first number is the input Layer, the last number is the output Layer, all other numbers will be hidden Layers
* numberOfTrainingRows = the number of randomized training rows to generate - usedful for already known functions, like sin, cos, etc.
* trainingType = an enum to tell the main function what you are trying to teach your network
* isBatch = a flag for indicating batch processing or not.  true = iterate over a batch of training inputs before updating weights and biases, false = stochastic gradient descent (ie, 1 input at a time before updates)
* batchSize = if the isBatch flag is true, then this indicates how many training inputs to run before updates

Current set up for MNIST numeric training:

	// Number of epochs to run
	final static int numberOfEpochs = 20;
	
	// Print epoch every N iterations
	final static int epochIteration = 1;
	
	// Learning rate
	final static double learningRate = 0.01;
	
	// Number of total layers
	final static int layerCount = 4;
	
	// Number of nodes in each hidden layer
	final static int[] nodesCount = {784, 128, 64, 10};
	
	// In the scenario where we generate randomized training rows, use this
	final static int numberOfTrainingRows = 20;
	
	// Identify what type of training we're doing
	final static TrainingType trainingType = TrainingType.NumberReading;
	
	// Are we running learning in batches or stochastic?
	final static boolean isBatch = true;
	
	// How big is the batch?
	final static int batchSize = 64;
