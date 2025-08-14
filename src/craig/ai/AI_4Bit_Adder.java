package craig.ai;

import java.util.Random;

public class AI_4Bit_Adder 
{
	// Number of nodes in hidden layer
	final int hiddenNodeCount = 75;
	
	// Number of input nodes
	final int inputNodeCount = 4;

	// Number of output nodes
	final int outputNodeCount = 5;
	
	// Number of epochs to run
	final int numberOfEpochs = 500000;
	
	// Print epoch every N iterations
	final int epochIteration = 100000;
	
	// Learning rate
	final double learningRate = 0.01;
	
    // Weights between input layer (2 neurons) and hidden layer (# neurons)
    double[][] inputToHiddenWeights = new double[hiddenNodeCount][inputNodeCount]; 

    // Biases for the # hidden neurons
    double[] hiddenBiases = new double[hiddenNodeCount];

    // Weights between hidden layer (# neurons) and output neurons (# neurons)
    double[][] hiddenToOutputWeights = new double[hiddenNodeCount][outputNodeCount];

    // Bias for the output neurons
    double outputBias[] = new double[outputNodeCount];

    Random random = new Random();

    public AI_4Bit_Adder() {
        // Initialize weights and biases with small random values
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            hiddenBiases[i] = getRandomWeight();
            
            for (int j = 0; j < outputNodeCount; j++)
            {
            	hiddenToOutputWeights[i][j] = getRandomWeight();
                outputBias[j] = getRandomWeight();
            }
            
            for (int j = 0; j < inputNodeCount; j++) 
            {
                inputToHiddenWeights[i][j] = getRandomWeight();
            }
        }
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
    
    // Next level feedForward function which will return a ForwardResult class
    // in order to retain the input array, the input array to the hidden layer, 
    // the array of activation of the hidden layer, the input to the output layer, 
    // and finally the activation of the output node.
    public ForwardResult4BitAdder feedForwardDetailed(double[] inputs) 
    {
        ForwardResult4BitAdder result = new ForwardResult4BitAdder(inputNodeCount, hiddenNodeCount, outputNodeCount);
        result.inputs = inputs;
        result.inputValuesOfHiddenNode = new double[hiddenNodeCount];
        result.activationValuesOfHiddenNode = new double[hiddenNodeCount];

        // Hidden layer computation
        for (int i = 0; i < hiddenNodeCount; i++) 
        {
            double sum = 0.0;
            for (int j = 0; j < inputNodeCount; j++) 
            {
                for (int k = 0; k < inputs.length; k++)
                {
                	sum += inputs[k] * inputToHiddenWeights[i][j];
                }
            }
            
            sum += hiddenBiases[i];
            result.inputValuesOfHiddenNode[i] = sum;
            result.activationValuesOfHiddenNode[i] = sigmoid(sum);
        }

        // Output neurons
        double sumOutput = 0.0;
        
        for (int i = 0; i < outputNodeCount; i++)
        {
        	sumOutput = 0.0;
        	
        	for (int j = 0; j < hiddenNodeCount; j++)
        	{
        		sumOutput += result.activationValuesOfHiddenNode[j] * hiddenToOutputWeights[j][i];
        	}
        	
            sumOutput += outputBias[i];
            result.inputValuesOfOutputNode[i] = sumOutput;
            result.activationValuesOfOutputNode[i] = sigmoid(sumOutput);
        }
        
        return result;
    }
    
    private void backpropagate(ForwardResult4BitAdder result, double expectedOutput, double learningRate, int outputNodeNumber) 
    {
        double output = result.activationValuesOfOutputNode[outputNodeNumber];;
        double error = output - expectedOutput;

        // Derivative of output layer activation
        // Meaning the derivative of the sigmoid function is
        // = sigmoid(x) * (1 - sigmoid(x))
        double deltaOutput = error * output * (1 - output);

        // Update weights and bias from hidden to output
        for (int j = 0; j < outputNodeCount; j++)
        {
	        for (int i = 0; i < hiddenNodeCount; i++) 
	        {
	            hiddenToOutputWeights[i][j] -= learningRate * deltaOutput * result.activationValuesOfHiddenNode[i];
	        }
        }
        
        outputBias[outputNodeNumber] -= learningRate * deltaOutput;

        // Hidden layer deltas and updates
        for (int k = 0; k < outputNodeCount; k++)
        {
	        for (int i = 0; i < hiddenNodeCount; i++) 
	        {
	            double hiddenActivation = result.activationValuesOfHiddenNode[i];
	            
	            double deltaHidden = deltaOutput * hiddenToOutputWeights[i][k] * hiddenActivation * (1 - hiddenActivation);
	
	            for (int j = 0; j < inputNodeCount; j++) 
	            {
	                inputToHiddenWeights[i][j] -= learningRate * deltaHidden * result.inputs[j];
	            }
	
	            hiddenBiases[i] -= learningRate * deltaHidden;
	        }
        }
    }

    public void train(double[][] inputs, double[][] expectedOutputs) 
    {
    	ForwardResult4BitAdder result = new ForwardResult4BitAdder(inputNodeCount, hiddenNodeCount, outputNodeCount);
    	
    	double[][] outputs = new double[outputNodeCount][inputs.length];
    	
    	double[] input;
    	
        for (int epoch = 0; epoch < numberOfEpochs; epoch++) 
        {
            for (int i = 0; i < inputs.length; i++) 
            {
                result = feedForwardDetailed(inputs[i]);
                
                for (int j = 0; j < outputNodeCount; j++)
                {
                	backpropagate(result, expectedOutputs[i][j], learningRate, j);
                    
                    if (epoch % epochIteration == 0)
                    {
                    	outputs[j][i] = result.activationValuesOfOutputNode[j];
                    }
                }
            }

            // Print data every N epochs
            if (epoch % epochIteration == 0) 
            {
            	System.out.println();
                System.out.printf("Epoch %d complete%n", epoch);
                
                for (int i = 0; i < inputs.length; i++) 
                {
                	input = inputs[i];
                	
                	System.out.printf("Input: [%d %d %d %d]%n", (int)input[0], (int)input[1], (int)input[2], (int)input[3]);
                	
                	for (int j = 0; j < outputNodeCount; j++)
                	{
                		System.out.printf(" → Output Node %d: Actual is: %.4f → expected is %d", j, outputs[j][i], (int)expectedOutputs[i][j]);
                		System.out.println();
                	}
                }
            }
        }
    }

    private void printResults(double[][] trainingInputs, double[][] trainingOutputs)
    {
        //double[] output = new double[outputNodeCount];
        
        double[] input = new double[trainingInputs[0].length];
        
        for (int i = 0; i < hiddenNodeCount; i++)
        {
        	for (int j = 0; j < inputNodeCount; j++)
        	{
            	System.out.printf("Input Node %d -> Hidden Node %d:%n", j, i);
            	System.out.printf("Input Weight To Hidden Node = %f%n", inputToHiddenWeights[i][j]);
        	}
        	
        	System.out.printf("Bias Hidden Node %d = %f%n%n", i, hiddenBiases[i]);
        }
        
        System.out.println();
        
        for (int i = 0; i < hiddenNodeCount; i++)
        {
        	System.out.printf("Hidden Node %d -> Output Node:%n", i);
        	
        	for (int j = 0; j < outputNodeCount; j++)
        	{
        		System.out.printf("Input Weight To Output Node %d = %f%n", j, hiddenToOutputWeights[i][j]);
        	}
        	
        	System.out.println();
        }
        
        for (int j = 0; j < outputNodeCount; j++)
        {
        	System.out.printf("Bias Output Node %d = %f%n%n", j, outputBias[j]);
        }
        
        System.out.println("Final result is:");
        
        for (int i = 0; i < trainingInputs.length; i++) 
        {
        	input = trainingInputs[i];
        	
        	ForwardResult4BitAdder results = feedForwardDetailed(input); 
            		
            System.out.printf("Input: [%d %d %d %d]%n", (int)input[0], (int)input[1], (int)input[2], (int)input[3]);
            
            System.out.printf("Wish 4: [");
            
            for (int j = 0; j < trainingOutputs[i].length; j++)
            {
            	System.out.printf("%.4f ", trainingOutputs[i][j]);
            }
            
            System.out.printf("]%n");
            
            System.out.printf("Actual: [");
            
            for (int j = 0; j < outputNodeCount; j++)
            {
            	System.out.printf("%.4f ", results.activationValuesOfOutputNode[j]);
            }
            
            System.out.printf("]%n%n");
        }


    }


    public static void main(String[] args) 
    {
        AI_4Bit_Adder ai = new AI_4Bit_Adder();

        // Test inputs for adding 4 bits together 
        double[][] trainingInputs = 
        {
            {0, 0, 0, 0}, // 0+0+0+0 = 0
            {0, 0, 0, 1}, // 0+0+0+1 = 1
            {0, 0, 1, 0}, // 0+0+1+0 = 1
            {0, 0, 1, 1}, // 0+0+1+1 = 2
            
            {0, 1, 0, 0}, // 0+1+0+0 = 1
            {0, 1, 0, 1}, // 0+1+0+1 = 2
            {0, 1, 1, 0}, // 0+1+1+0 = 2
            {0, 1, 1, 1}, // 0+1+1+1 = 3
            
            {1, 0, 0, 0}, // 1+0+0+0 = 1
            {1, 0, 0, 1}, // 1+0+0+1 = 2
            {1, 0, 1, 0}, // 1+0+1+0 = 2
            {1, 0, 1, 1}, // 1+0+1+1 = 3
            
            {1, 1, 0, 0}, // 1+1+0+0 = 2
            {1, 1, 0, 1}, // 1+1+0+1 = 3
            {1, 1, 1, 0}, // 1+1+1+0 = 3
            {1, 1, 1, 1}  // 1+1+1+1 = 4
        };

        // Results from adding
        // Switching the results to 5 bits, treated as lightbulbs.  Only one bit should be
        // "on" in the result.  So [01000] is 1, while [00001] is 4, etc.  
        double[][] trainingOutputs = 
        {
        	{1, 0, 0, 0, 0}, // 0
        	{0, 1, 0, 0, 0}, // 1
        	{0, 1, 0, 0, 0}, // 1
        	{0, 0, 1, 0, 0}, // 2
        	
        	{0, 1, 0, 0, 0}, // 1
        	{0, 0, 1, 0, 0}, // 2
        	{0, 0, 1, 0, 0}, // 2
        	{0, 0, 0, 1, 0}, // 3
        	
        	{0, 1, 0, 0, 0}, // 1
        	{0, 0, 1, 0, 0}, // 2
        	{0, 0, 1, 0, 0}, // 2
        	{0, 0, 0, 1, 0}, // 3
        	
        	{0, 0, 1, 0, 0}, // 2
        	{0, 0, 0, 1, 0}, // 3
        	{0, 0, 0, 1, 0}, // 3
        	{0, 0, 0, 0, 1}  // 4
        }; 

        ai.train(trainingInputs, trainingOutputs);

        ai.printResults(trainingInputs, trainingOutputs);
    }
}

class ForwardResult4BitAdder 
{
	double[] inputs;
    double[] inputValuesOfHiddenNode;
    double[] activationValuesOfHiddenNode;
    double[] inputValuesOfOutputNode;
    double[] activationValuesOfOutputNode;
    
    public ForwardResult4BitAdder(int numberInputs, int numberHiddenNodes, int numberOutputNodes)
    {
    	inputs = new double[numberInputs];
    	inputValuesOfHiddenNode = new double[numberHiddenNodes];
    	activationValuesOfHiddenNode = new double[numberHiddenNodes];
    	inputValuesOfOutputNode = new double[numberOutputNodes];
    	activationValuesOfOutputNode = new double[numberOutputNodes];
    }
}

