package craig.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork_2Bit_XOR 
{
	// Number of hidden layers
	final int hiddenLayerCount = 2;
	
	// Number of nodes in each hidden layer
	final int[] hiddenNodeCount = {2, 8};
	
	// Number of input nodes
	final int inputNodeCount = 2;

	// Number of output nodes
	final int outputNodeCount = 1;
	
	// Number of epochs to run
	final int numberOfEpochs = 100001;
	
	// Print epoch every N iterations
	final int epochIteration = 10000;
	
	// Learning rate
	final double learningRate = 0.01;
	
    // Weights between input layer (2 neurons) and 1st hidden layer (# neurons)
    double[][] inputToHiddenWeights1stLayer = new double[hiddenNodeCount[0]][inputNodeCount]; 

    // Biases for the hidden neurons in all hidden layers
    // The index for this array will be the hidden layer number
    // Each item in this array will contain a double array for the actual biases for that layer
    // Will be initialized in this class's constructor
    HiddenBias[] hiddenBiasesList = new HiddenBias[hiddenLayerCount]; 
    
    // This ArrayList will really only be used if there are at least 2 hidden layers.
    // If there's only 1, then the existing variables will suffice.  They cover inputs
    // to 1st hidden and then the final hidden to the outputs.  In that case, the 1st 
    // and final hidden layers are the same layer.  
    List<HiddenLayer> hiddenLayersArray = new ArrayList<>();

    // Weights between final hidden layer (# neurons) and output neurons (# neurons)
    double[][] hiddenLastLayerToOutputWeights = new double[hiddenNodeCount[hiddenLayerCount - 1]][outputNodeCount];

    // Bias for the output neurons
    double outputBias[] = new double[outputNodeCount];

    Random random = new Random();

    @SuppressWarnings("unused")
	public NeuralNetwork_2Bit_XOR() 
    {
       	for (int k = 0; k < hiddenLayerCount; k++)
    	{
    		hiddenBiasesList[k] = new HiddenBias(hiddenNodeCount[k]);
    		
    		hiddenBiasesList[k].hiddenBiases = new double[hiddenNodeCount[k]];
    		
	        // Initialize weights and biases with small random values
	        for (int i = 0; i < hiddenNodeCount[k]; i++) 
	        {
	        	// We'll always have a 1st hidden layer
	        	hiddenBiasesList[k].hiddenBiases[i] = getRandomWeight();
	            
	            // We'll always have an output layer, but we'll populate it 
	        	// once we're at the final hidden layer
	        	if ((k + 1) == hiddenLayerCount)
	        	{
		            for (int j = 0; j < outputNodeCount; j++)
		            {
		            	hiddenLastLayerToOutputWeights[i][j] = getRandomWeight();
		                outputBias[j] = getRandomWeight();
		            }
	        	}
	            
	            // We'll always have an input to 1st hidden layer, so only 
	        	// populate this while we're looking at the first hidden layer
	        	if (k == 0)
	        	{
		            for (int j = 0; j < inputNodeCount; j++) 
		            {
		                inputToHiddenWeights1stLayer[i][j] = getRandomWeight();
		            }
	        	}
	        }
        }
    	
        // If we have more than 1 hidden layer, then we have to initialize
        // our ArrayList to hold the weights and biases between these layers
    	// If there are 2 layers, then this will hold the 1st HiddenLayer, which maps 
    	// the weights between this layer and right.  
    	// If there are 3 layers, then this will hold the 1st layer and the 2nd layer.
    	// But this will never hold the final layer, since the inputs from the final hidden layer are
    	// handled by the hiddenLastLayerToOutputWeights double list.
        if (hiddenLayerCount > 1)
        {
        	for (int i = 0; i < (hiddenLayerCount - 1); i++)
        	{
        		hiddenLayersArray.add(new HiddenLayer(hiddenNodeCount[i], hiddenNodeCount[i + 1]));
        		
        		for (int j = 0; j < hiddenNodeCount[i]; j++)
        		{
        			for (int k = 0; k < hiddenNodeCount[i+1]; k++)
        			{
        				hiddenLayersArray.get(i).hiddenLayerWeightsToNextLayer[j][k] = getRandomWeight();
        			}
        		}
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
    public ForwardResult4BitAdder2HiddenLayers feedForwardDetailed(double[] inputs) 
    {
        ForwardResult4BitAdder2HiddenLayers result = new ForwardResult4BitAdder2HiddenLayers(inputNodeCount, hiddenNodeCount, outputNodeCount, hiddenLayerCount);
        result.inputs = inputs;
        
        double sum = 0.0;
        double sumHidden = 0.0;
        double sumOutput = 0.0;
        
        // Calculate input/first hidden layer values first
        for (int i = 0; i < hiddenNodeCount[0]; i++)
        {
            // We're in the first hidden layer, then the weights are coming
            // into the first layer, so treat this like we did before expanding 
            // the number of layers
            for (int j = 0; j < inputNodeCount; j++) 
            {
               	sum += inputs[j] * inputToHiddenWeights1stLayer[i][j];
            }
            
            sum += hiddenBiasesList[0].hiddenBiases[i];
            
            result.hiddenLayer.get(0).inputValuesOfHiddenNode[i] = sum;
            result.hiddenLayer.get(0).activationValuesOfHiddenNode[i] = sigmoid(sum);
        }
        
        // Now let's go through the next N layers and sum up the inputs between them.  The main outer loop will
        // iterate over the number of hidden layers we've got, starting with 1 (since 0 was covered in the input level.
        // Like with the previous section, our first "real" outer loop will count the nodes in the next layer to the right.
        // The inner loop will count the nodes in the layer to the left.  We'll then sum up the result of multiplying
        // the activation node from the left layer by the weight coming from that node.  After applying the bias,
        // we'll assign that to the input value of h-layer hidden node and then assign the sigmoid of that to 
        // the activation value of the same layer.  
        for (int h = 1; h < hiddenLayerCount; h++)
        {
            sumHidden = 0.0;
            
	        for (int i = 0; i < hiddenNodeCount[h]; i++) 
	        {
            	for (int j = 0; j < hiddenNodeCount[h - 1]; j++)
            	{
            		// At this point, h is the hidden layer we're concerned with.  We know h is at least 1, meaning
            		// that we're at least at the 2nd hidden layer.  Because of that, we know that the result object
            		// will need to pull the activation values from h-1 and multiply that against the nodes
            		// found in hiddenLayersArray, also at h-1
            		
            		sumHidden += result.hiddenLayer.get(h - 1).activationValuesOfHiddenNode[j] * hiddenLayersArray.get(h - 1).hiddenLayerWeightsToNextLayer[j][i];
            	}
	            
            	sumHidden += hiddenBiasesList[h].hiddenBiases[i];
	            
	            result.hiddenLayer.get(h).inputValuesOfHiddenNode[i] = sumHidden;
	            result.hiddenLayer.get(h).activationValuesOfHiddenNode[i] = sigmoid(sumHidden);
	        }
        }

        // Output neurons
        for (int i = 0; i < outputNodeCount; i++)
        {
        	sumOutput = 0.0;
        	
        	for (int j = 0; j < hiddenNodeCount[hiddenLayerCount - 1]; j++)
        	{
        		sumOutput += result.hiddenLayer.get(hiddenLayerCount - 1).activationValuesOfHiddenNode[j] * hiddenLastLayerToOutputWeights[j][i];
        	}
        	
            sumOutput += outputBias[i];
            result.inputValuesOfOutputNode[i] = sumOutput;
            result.activationValuesOfOutputNode[i] = sigmoid(sumOutput);
        }
        
        return result;
    }
    
    private void backpropagate(ForwardResult4BitAdder2HiddenLayers result, double expectedOutput, double learningRate, int outputNodeNumber) 
    {
        double output = result.activationValuesOfOutputNode[outputNodeNumber];;
        double error = output - expectedOutput;
        
        double hiddenActivation;
        
        double deltaHidden;

        // Derivative of output layer activation
        // Meaning the derivative of the sigmoid function is
        // = sigmoid(x) * (1 - sigmoid(x))
        double deltaOutput = error * output * (1 - output);

        // Update weights and bias from hidden to output
        // We're going to focus on the final hidden layer first, as that layer is the one that 
        // touches the output layer
        for (int j = 0; j < outputNodeCount; j++)
        {
	        for (int i = 0; i < hiddenNodeCount[hiddenLayerCount - 1]; i++) 
	        {
	            hiddenLastLayerToOutputWeights[i][j] -= learningRate * deltaOutput * result.hiddenLayer.get(hiddenLayerCount - 1).activationValuesOfHiddenNode[i];
	        }
        }
        
        outputBias[outputNodeNumber] -= learningRate * deltaOutput;
        
        // Before this point, our hiddenLayersArray contained all hidden layers except the rightmost one.
        // The intent originally was to have the final hidden layer be kept elsewhere.  But now I'm realizing
        // that with the backprop looping, it'll be more flexible code-wise to be able to pull that same info
        // out of the this Array with all the other hidden layers in it.  So that's why now I'm adding the 
        // results from the rightmost backprop into a HiddenLayer object which then gets added to the end of the 
        // already existing Array.  Eventually I'll have to re-think this approach, but for now I just want it working.
        HiddenLayer hl = new HiddenLayer();
        hl.setLayerFromBackProp(hiddenLastLayerToOutputWeights);
        hiddenLayersArray.add(hl);
        
        // If there are multiple hidden layers, it's time to calculate the deltas between each of them.
        // If there is only one hidden layer, then there's nothing to do here and the loop should just be skipped.
        // result.hiddenLayers has all the hidden layers
        // hiddenLayersArray has N-1 hidden layers (since the 1st layer's weights are stored separately)
        // So for backproping through the hidden layers, we want to start on the "right" most hidden layer,
        // which is indexed (i - 1) for hiddenLayersArray, but as i for results.hiddenLayer.  Good luck! 
        // We need to calculate the weights coming from the right-1 layer (pulling actual activation values 
        // from result.hiddenLayer) and then storing the new weights into the hiddenLayersArray.  Go! 
        for (int i = hiddenLayerCount - 1; i > 0; i--)
        {
        	// Like above, this first for loop is for the rightmost layer (imitating the output layer above)
        	// The inner for loop will be for the layer to the left of that (copying the final hidden layer above)
        	for (int j = 0; j < hiddenNodeCount[i]; j++)
        	{
        		for (int k = 0; k < hiddenNodeCount[i - 1]; k++)
        		{
        			// In this section of the loop, j is the index for the "right" most hidden node list.
        			// k is the index for the hidden node list to the left of it.
        			// Here we need to update the hiddenLayersArray left list to update its hiddenLayerWeightsToNextLayer 
        			// values by pulling activation values from the right list.  
        			
        			// Get the activation value of the nodes of the rightmost hidden layer
        			hiddenActivation = result.hiddenLayer.get(i).activationValuesOfHiddenNode[j];
        			
        			deltaHidden = deltaOutput * hiddenLayersArray.get(i - 1).hiddenLayerWeightsToNextLayer[k][j] * hiddenActivation * (1 - hiddenActivation);
        			
        			for (int h = 0; h < hiddenNodeCount[i]; h++)
        			{
        				hiddenLayersArray.get(i - 1).hiddenLayerWeightsToNextLayer[k][j] -= learningRate * deltaHidden * result.hiddenLayer.get(i).inputValuesOfHiddenNode[j];
        			}
        			
        			hiddenBiasesList[i].hiddenBiases[j] -= learningRate * deltaHidden;
        		}
        	}
        }        

        // Now it's time to backprop the input nodes to the 1st hidden layer  
        for (int k = 0; k < inputNodeCount; k++)
        {
	        for (int i = 0; i < hiddenNodeCount[0]; i++) 
	        {
	            hiddenActivation = result.hiddenLayer.get(0).activationValuesOfHiddenNode[i];
	            
	            deltaHidden = deltaOutput * inputToHiddenWeights1stLayer[i][k] * hiddenActivation * (1 - hiddenActivation);
	
	            for (int j = 0; j < inputNodeCount; j++) 
	            {
	                inputToHiddenWeights1stLayer[i][j] -= learningRate * deltaHidden * result.inputs[j];
	            }
	
	            hiddenBiasesList[0].hiddenBiases[i] -= learningRate * deltaHidden;
	        }
        }
    }

    public void train(double[][] inputs, double[][] expectedOutputs) 
    {
    																						
    	ForwardResult4BitAdder2HiddenLayers result = new ForwardResult4BitAdder2HiddenLayers(inputNodeCount, hiddenNodeCount, outputNodeCount, hiddenLayerCount);
    	
    	double[][] outputs = new double[outputNodeCount][inputs.length];
    	
    	double[] input;
    	
        for (int epoch = 0; epoch < numberOfEpochs; epoch++) 
        {
            for (int i = 0; i < inputs.length; i++) 
            {
/*            	System.out.println("************* NON-OOP Version *************");
            	System.out.println("************* DEBUGGING DATA BEGIN *************");
            	System.out.printf("Epoch number %d%n", epoch);
            	System.out.printf("%n$$$$$$$$$$$$$$$%nAbout to run the feedforward for input [%f, %f]%n", inputs[i][0], inputs[i][1]);
            	printNetwork();
*/            	
                result = feedForwardDetailed(inputs[i]);
                
/*                System.out.println("Here are the weighted sums PRIOR to activation");
                
                printWeightedSums(result);
                
                System.out.printf("%n$$$$$$$$$$$$$$$%nResults from feedforward for input [%f, %f]%n", inputs[i][0], inputs[i][1]);
            	printNetwork();
*/                
                for (int j = 0; j < outputNodeCount; j++)
                {
                	backpropagate(result, expectedOutputs[i][j], learningRate, j);
                	
/*                	System.out.printf("%n$$$$$$$$$$$$$$$%nResults from backprop for input [%f, %f]%n", inputs[i][0], inputs[i][1]);
                	printNetwork();
*/                    
                    if (epoch % epochIteration == 0)
                    {
                    	outputs[j][i] = result.activationValuesOfOutputNode[j];
                    }
                }
                
/*                System.out.println("Printing results data (which will include all 4 inputs)");
                printOutput(inputs, expectedOutputs);
                
            	System.out.println("************* DEBUGGING DATA END *************");*/
            }

            // Print data every N epochs
            if (epoch % epochIteration == 0) 
            {
            	System.out.println();
                System.out.printf("Epoch %d complete%n", epoch);
                
                for (int i = 0; i < inputs.length; i++) 
                {
                	input = inputs[i];
                	
                	// Printing for XOR
                	System.out.printf("Input: [%d %d]%n", (int)input[0], (int)input[1]);
                	
                	for (int j = 0; j < outputNodeCount; j++)
                	{
                		System.out.printf(" → Output Node %d: Actual is: %.4f → expected is %d", j, outputs[j][i], (int)expectedOutputs[i][j]);
                		System.out.println();
                	}
                }
            }
        }
    }
    
    private void printWeightedSums(ForwardResult4BitAdder2HiddenLayers result)
    {	
    	for (int j = 0; j < hiddenLayerCount; j++)
    	{
    		for (int i = 0; i < hiddenNodeCount[j]; i++)
    		{
    			System.out.printf("Hidden layer %d hidden node %d weighted sum = %f%n", j, i, result.hiddenLayer.get(j).inputValuesOfHiddenNode[i]);
    		}
    	}
    	
    	  // Output neurons
        for (int i = 0; i < outputNodeCount; i++)
        {
        	System.out.printf("Output layer output node %d weighted sum = %f%n", i, result.inputValuesOfOutputNode[i]);
        }
    }

    private void printOutput(double[][] trainingInputs, double[][] trainingOutputs)
    {
    	double[] input = new double[trainingInputs[0].length];
    	
        System.out.println("Final result is:");
        
        for (int i = 0; i < trainingInputs.length; i++) 
        {
        	input = trainingInputs[i];
        	
        	ForwardResult4BitAdder2HiddenLayers results = feedForwardDetailed(input); 
            		
        	// Printing for XOR
        	System.out.printf("Input: [%d %d]%n", (int)input[0], (int)input[1]);
            
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
    
    private void printResults(double[][] trainingInputs, double[][] trainingOutputs)
    {  
    	System.out.println();
    	System.out.printf("The results from the run of %d epochs is:%n", numberOfEpochs);
        double[] input = new double[trainingInputs[0].length];
        
        // First up, let's print out some info between the input nodes and the first hidden node layer
        for (int i = 0; i < hiddenNodeCount[0]; i++)
        {
        	for (int j = 0; j < inputNodeCount; j++)
        	{
            	System.out.printf("Input Node %d -> Hidden Node %d:%n", j, i);
            	System.out.printf("Input Weight To Hidden Node = %f%n", inputToHiddenWeights1stLayer[i][j]);
        	}
        	
        	System.out.printf("Bias Hidden Node %d = %f%n%n", i, hiddenBiasesList[0].hiddenBiases[i]);
        }
        
        System.out.println();
        
        // Now we have to cycle through the hidden input layers to show weights and biases between them.
        // If there's only one hidden layer, then this will just be skipped.
        for (int i = 0; i < hiddenLayerCount - 1; i++)
        {
        	System.out.printf("The details between hidden layer %d and %d:%n", i, i + 1);
        	
        	for (int j = 0; j < hiddenNodeCount[i]; j++)
        	{
        		for (int k = 0; k < hiddenNodeCount[i + 1]; k++)
        		{
        			System.out.printf("Hidden Node %d -> Hidden Node %d:%n", j, k);
        			System.out.printf("Hidden Weight To Hidden Node = %f%n", hiddenLayersArray.get(i).hiddenLayerWeightsToNextLayer[j][k]);
        			
            		System.out.printf("Bias Hidden Node %d = %f%n%n", k, hiddenBiasesList[i + 1].hiddenBiases[k]);
        		}   
        	}     	
        }
        
        System.out.println();
        
        // Finally, we'll print the details between the final hidden node layer and the output layer
        
        for (int i = 0; i < hiddenNodeCount[hiddenLayerCount - 1]; i++)
        {
        	System.out.printf("Hidden Node %d -> Output Node:%n", i);
        	
        	for (int j = 0; j < outputNodeCount; j++)
        	{
        		System.out.printf("Input Weight To Output Node %d = %f%n", j, hiddenLastLayerToOutputWeights[i][j]);
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
        	
        	ForwardResult4BitAdder2HiddenLayers results = feedForwardDetailed(input); 
            		
        	// Printing for XOR
        	System.out.printf("Input: [%d %d]%n", (int)input[0], (int)input[1]);
            
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
    
    // These values were extracted from a working XOR run.  I used these to test the updated 
    // version of this Network logic, refactoring it to OOP form.  If you need a consistent
    // flow from input to output, you can call this function after setup and random initialization
    // of weights and biases. 
    public void seedNetwork()
    {
    	// From Input (Layer 0) to Hidden Layer 0 (Layer 1)
    	inputToHiddenWeights1stLayer[0][0] = -0.49374066664118454;
    	inputToHiddenWeights1stLayer[1][0] = 0.21387611998075173;
    	inputToHiddenWeights1stLayer[2][0] = -0.09941186582161810;
    	inputToHiddenWeights1stLayer[3][0] = 0.32897686226310420;
    	inputToHiddenWeights1stLayer[0][1] = 0.60240370691785890;
    	inputToHiddenWeights1stLayer[1][1] = 0.09450286710529077;
    	inputToHiddenWeights1stLayer[2][1] = -0.55748926858078860;
    	inputToHiddenWeights1stLayer[3][1] = 0.10304643409175185;

    	// From Hidden Layer 0 (Layer 1) to Hidden Layer 1 (Layer 2)
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[0][0] = -0.41118280924008930;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[0][1] = 0.61872656885185020;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[1][0] = 0.31686833046069050;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[1][1] = -0.38653245202219066;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[2][0] = 0.31265739503449486;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[2][1] = 0.80426453911883720;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[3][0] = -0.61292677330547750;
    	hiddenLayersArray.get(0).hiddenLayerWeightsToNextLayer[3][1] = -0.47900854555162420;

    	// From Hidden Layer 1 (Layer 2) to Output Layer (Layer 3)
    	hiddenLastLayerToOutputWeights[0][0] = -0.70007024458959140;
    	hiddenLastLayerToOutputWeights[1][0] = 0.76015925454438230;
    	
    	// Hidden Layer 0 biases  (HIDDEN LAYER NUMBER = 0)
    	hiddenBiasesList[0].hiddenBiases[0] =  0.78874876894524130;
    	hiddenBiasesList[0].hiddenBiases[1] =  0.98130160389828250;
    	hiddenBiasesList[0].hiddenBiases[2] = -0.91000699720043880;
    	hiddenBiasesList[0].hiddenBiases[3] = -0.76130048884048930;

    	// Hidden Layer 1 biases  (HIDDEN LAYER NUMBER = 1)
    	hiddenBiasesList[1].hiddenBiases[0] =  0.72773747552058250;
    	hiddenBiasesList[1].hiddenBiases[1] = -0.37898926599938830;

    	// Output layer biases
    	outputBias[0] = 0.26178720340242423;
    }
    
    public void printNetwork()
    {
    	System.out.println("Weights From Input To Hidden Layer 0:");
    	for (int i = 0; i < inputNodeCount; i++)
    	{
    		for (int j = 0; j < hiddenNodeCount[0]; j++)
    		{
    			System.out.printf("Input %d to Hidden Node %d Weight = %.17f%n", i, j, inputToHiddenWeights1stLayer[j][i]);
    		}
    	}
    	
    	for (int i = 0; i < hiddenLayerCount - 1; i++)
    	{
    		System.out.printf("Weights From Hidden Layer %d to Hidden Layer %d:%n", i, i + 1);
	    	
    		for (int j = 0; j < hiddenNodeCount[i]; j++)
	    	{
	    		for (int k = 0; k < hiddenNodeCount[i + 1]; k++)
	    		{
	    			System.out.printf("Hidden Node %d to Hidden Node %d Weight = %.17f%n", i, j, hiddenLayersArray.get(i).hiddenLayerWeightsToNextLayer[j][k]);
	    		}
	    	}
    	}
    	
    	
    	for (int j = 0; j < hiddenLayerCount; j++)
    	{
    		
    		System.out.printf("\nHidden Layer %d:%n", j);
    		
	    	for (int i = 0; i < hiddenNodeCount[j]; i++)
	    	{
	    		System.out.printf("Hidden Node %d Bias = %.17f%n", i, hiddenBiasesList[j].hiddenBiases[i]);
	    	}
    	}    	
    	
    	System.out.println("Output Layer:");
    	
    	for (int i = 0; i < outputNodeCount; i++)
    	{
    		System.out.printf("Output Node %d Bias = %.17f%n", i, outputBias[i]);
    	}
    }


    public static void main(String[] args) 
    {
        NeuralNetwork_2Bit_XOR ai = new NeuralNetwork_2Bit_XOR();

        ai.printNetwork();
        
        // Test inputs for XOR 
        double[][] trainingInputs = 
        {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] trainingOutputs = {{0}, {1}, {1}, {0}}; // XOR targets

        ai.train(trainingInputs, trainingOutputs);

        ai.printResults(trainingInputs, trainingOutputs);
    }
}


/*        // Test inputs for adding 4 bits together 
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
// Switching the results to 5 bits, treated as light bulbs.  Only one bit should be
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
*/

// This helper class will set up the architecture of the network:
// The input nodes
// An ArrayList of Hidden Layers based on the number of hidden layers (each with a count of nodes)
// The output nodes (with input values and activation values)
class ForwardResult4BitAdder2HiddenLayers 
{
	double[] inputs;
	
	List<ForwardResultHiddenLayer> hiddenLayer;
    
    double[] inputValuesOfOutputNode;
    double[] activationValuesOfOutputNode;
    
    public ForwardResult4BitAdder2HiddenLayers(int numberInputs, int[] numberHiddenNodes, int numberOutputNodes, int numberHiddenLayers)
    {
    	inputs = new double[numberInputs];
    	
    	hiddenLayer = new ArrayList<>();
    	
    	for (int i = 0; i < numberHiddenLayers; i++)
    	{
    		hiddenLayer.add(new ForwardResultHiddenLayer(numberHiddenNodes[i]));
    	}

    	inputValuesOfOutputNode = new double[numberOutputNodes];
    	activationValuesOfOutputNode = new double[numberOutputNodes];
    }
}

class ForwardResultHiddenLayer
{
	double[] inputValuesOfHiddenNode;
	double[] activationValuesOfHiddenNode;
	
	public ForwardResultHiddenLayer(int numberHiddenNodes)
	{
		inputValuesOfHiddenNode = new double[numberHiddenNodes];
		activationValuesOfHiddenNode = new double[numberHiddenNodes];
	}
}

class HiddenLayer
{
    // Weights between this input layer and the next layer
	// We need to know how many nodes are in this layer and how many are in the next one
    double[][] hiddenLayerWeightsToNextLayer; 

    public HiddenLayer(int numberHiddenNodesThisLayer, int numberHiddenNodesNextLayer)
    {
    	hiddenLayerWeightsToNextLayer = new double[numberHiddenNodesThisLayer][numberHiddenNodesNextLayer];
    }
    
    public HiddenLayer()
    {
    	// Just here for backprop setup.
    }
    
    public void setLayerFromBackProp(double[][] hiddenToOutputWeights)
    {
    	// This will be called during backprop in order to add the hidden output weights into the Array
    	// that holds all hidden layers (since we don't normally need the final hidden layer in the Array).
    	
    	hiddenLayerWeightsToNextLayer = new double[hiddenToOutputWeights.length][hiddenToOutputWeights[0].length];
    	
    	for (int i = 0; i < hiddenToOutputWeights.length; i++)
    	{
    		for (int j = 0; j < hiddenToOutputWeights[0].length; j++)
    		{
    			hiddenLayerWeightsToNextLayer[i][j] = hiddenToOutputWeights[i][j];
    		}
    	}
    }
}

class HiddenBias
{
	double[] hiddenBiases;
	
	public HiddenBias(int numberOfNodes)
	{
		hiddenBiases = new double[numberOfNodes];
	}
}