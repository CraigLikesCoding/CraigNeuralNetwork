package craig.ai;

import java.util.Random;

public class AI_XOR 
{
	// Number of nodes in hidden layer
	final int hiddenNodeCount = 3;
	
	// Number of input nodes
	final int inputNodeCount = 2;

    // Weights between input layer (2 neurons) and hidden layer (# neurons)
    double[][] inputToHiddenWeights = new double[hiddenNodeCount][inputNodeCount]; 

    // Biases for the # hidden neurons
    double[] hiddenBiases = new double[hiddenNodeCount];

    // Weights between hidden layer (# neurons) and output neuron (1 neuron)
    double[] hiddenToOutputWeights = new double[hiddenNodeCount];

    // Bias for the output neuron
    double outputBias;

    Random random = new Random();

    public AI_XOR() {
        // Initialize weights and biases with small random values
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            hiddenBiases[i] = getRandomWeight();
            
            hiddenToOutputWeights[i] = getRandomWeight();
            
            for (int j = 0; j < inputNodeCount; j++) 
            {
                inputToHiddenWeights[i][j] = getRandomWeight();
            }
        }
        outputBias = getRandomWeight();
    }

    // Generates a small random number between -1 and 1
    private double getRandomWeight() 
    {
        return random.nextDouble() * 2 - 1;
    }

    // Sigmoid activation function
    private double sigmoid(double x) 
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

  
    // Initial forward pass: compute output given inputs
    public double feedForward(double[] inputs) 
    {
        // Step 1: Compute hidden layer outputs
        double[] hiddenOutputs = new double[hiddenNodeCount];
        
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            double sum = 0.0;

            // Weighted sum of inputs to this hidden neuron
            for (int j = 0; j < inputNodeCount; j++) 
            {
                sum += inputs[j] * inputToHiddenWeights[i][j];
            }

            // Add bias, then apply activation function
            sum += hiddenBiases[i];
            
            hiddenOutputs[i] = sigmoid(sum);
        }

        // Step 2: Compute output layer result
        double outputSum = 0.0;
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            outputSum += hiddenOutputs[i] * hiddenToOutputWeights[i];
        }

        // Add output bias and apply sigmoid
        outputSum += outputBias;
        return sigmoid(outputSum);
    }
    
    // Next level feedForward function which will return a ForwardResult class
    // in order to retain the input array, the input array to the hidden layer, 
    // the array of activation of the hidden layer, the input to the output layer, 
    // and finally the activation of the output node.
    public ForwardResult feedForwardDetailed(double[] inputs) 
    {
        ForwardResult result = new ForwardResult();
        result.inputs = inputs;
        result.inputValueOfHiddenNode = new double[hiddenNodeCount];
        result.activationValueOfHiddenNode = new double[hiddenNodeCount];

        // Hidden layer computation
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            double sum = 0.0;
            for (int j = 0; j < inputNodeCount; j++) 
            {
                sum += inputs[j] * inputToHiddenWeights[i][j];
            }
            
            sum += hiddenBiases[i];
            result.inputValueOfHiddenNode[i] = sum;
            result.activationValueOfHiddenNode[i] = sigmoid(sum);
        }

        // Output neuron
        double sumOutput = 0.0;
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            sumOutput += result.activationValueOfHiddenNode[i] * hiddenToOutputWeights[i];
        }
        sumOutput += outputBias;
        result.inputValueOfOutputNode = sumOutput;
        result.activationValueOfOutputNode = sigmoid(sumOutput);

        return result;
    }
    
    private void backpropagate(ForwardResult result, double expectedOutput, double learningRate) 
    {
        double output = result.activationValueOfOutputNode;
        double error = output - expectedOutput;

        // Derivative of output layer activation
        // Meaning the derivative of the sigmoid function is
        // = sigmoid(x) * (1 - sigmoid(x))
        double deltaOutput = error * output * (1 - output);

        // Update weights and bias from hidden to output
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            hiddenToOutputWeights[i] -= learningRate * deltaOutput * result.activationValueOfHiddenNode[i];
        }
        
        outputBias -= learningRate * deltaOutput;

        // Hidden layer deltas and updates
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            double hiddenActivation = result.activationValueOfHiddenNode[i];
            
            double deltaHidden = deltaOutput * hiddenToOutputWeights[i] * hiddenActivation * (1 - hiddenActivation);

            for (int j = 0; j < inputNodeCount; j++) 
            {
                inputToHiddenWeights[i][j] -= learningRate * deltaHidden * result.inputs[j];
            }

            hiddenBiases[i] -= learningRate * deltaHidden;
        }
    }

    public void train(double[][] inputs, double[] expectedOutputs, int epochs, double learningRate) 
    {
    	ForwardResult result = new ForwardResult();
    	
    	double[] outputs = new double[4];
    	
    	double[] input;
    	
        for (int epoch = 0; epoch < epochs; epoch++) 
        {
            for (int i = 0; i < inputs.length; i++) 
            {
                result = feedForwardDetailed(inputs[i]);
                backpropagate(result, expectedOutputs[i], learningRate);
                
                if (epoch % 1000 == 0)
                {
                	outputs[i] = result.activationValueOfOutputNode;
                }
            }

            // Optionally print error every N epochs
            if (epoch % 1000 == 0) 
            {
                System.out.printf("Epoch %d complete%n", epoch);
                
                for (int i = 0; i < inputs.length; i++) 
                {
                	input = inputs[i];
                    System.out.printf("Input: [%d %d] → Output: %.4f%n", (int)input[0], (int)input[1], outputs[i]);
                }
            }
        }
    }

    private void printResults(double[][] trainingInputs)
    {
        double output;
        
        System.out.println("Final result is:");
        
        for (double[] input : trainingInputs) 
        {
            output = feedForward(input);
            System.out.printf("Input: [%d %d] → Output: %.4f%n", (int)input[0], (int)input[1], output);
        }

        for (int i = 0; i < hiddenNodeCount; i++)
        {
        	for (int j = 0; j < inputNodeCount; j++)
        	{
            	System.out.printf("Input Node %d -> Hidden Node %d:%n", j, i);
            	System.out.printf("Input Weight To Hidden Node = %f%n", inputToHiddenWeights[i][j]);
        	}
        	
        	System.out.printf("Bias = %f%n%n", hiddenBiases[i]);
        }
        
        System.out.println();
        
        for (int i = 0; i < hiddenNodeCount; i++)
        {
        	System.out.printf("Hidden Node %d -> Output Node:%n", i);
        	System.out.printf("Input Weight To Output Node = %f%n", hiddenToOutputWeights[i]);
        }
        
        System.out.printf("Bias = %f%n%n", outputBias);
    }


    public static void main(String[] args) 
    {
        AI_XOR ai = new AI_XOR();

        // Test inputs for XOR 
        double[][] trainingInputs = 
        {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] trainingOutputs = {0, 1, 1, 0}; // XOR targets

        ai.train(trainingInputs, trainingOutputs, 300000, 0.05);

        
        
        ai.printResults(trainingInputs);
    }
}

class ForwardResult 
{
	double[] inputs;
    double[] inputValueOfHiddenNode;
    double[] activationValueOfHiddenNode;
    double inputValueOfOutputNode;
    double activationValueOfOutputNode;
}

