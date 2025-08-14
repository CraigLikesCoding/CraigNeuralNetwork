package craig.ai;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import craig.ai.external.MNISTLoader;
import craig.ai.external.NetworkReaderWriter;
import craig.ai.layers.Layer;
import craig.ai.layers.SigmoidLayer;
import craig.ai.layers.SoftmaxLayer;
import craig.ai.layers.TanhLayer;

public class NeuralNetwork 
{
	List<Layer> layers = new ArrayList<>();
	
	// Number of epochs to run
	private int numberOfEpochs;
	
	// Print epoch every N iterations
	private int epochIteration;
	
	// Learning rate
	private double learningRate;

	// Total number of network layers, including input, hidden, and output
	private int layerCount;
	
	// Number of nodes in each layer (input first, then hidden, then output)
	// nodeCounts.length should equal layerCount
	private int[] nodeCounts;
	
	// Set this to true if you want to see a ton of debugging info - like printing the contents of 
	// the entire network out 4 times per input per epoch.  It is verbose.  Not recommended to do this 
	// unless your numberOfEpochs is under 5.
	private boolean printDebug = false;
	
	// After each epoch, calculate the MSE and store it here for future reference or graphing
	ArrayList<HashMap<Integer, Double>> mseList = new ArrayList<>();
	
	// Helper class for generating training values
	Helper helper;
	
	// Identifies if the forwardFeed and backprop should be done in batch (meaning updated
	// accumulated deltas and gradients) or stochastically
	private boolean isBatch = false;
	
	// If we're running in batch, this will store the number of training pairs to run before
	// actually updating weights and biases
	private int batchSize;
	
	public NeuralNetwork(int numberOfEpochs, int epochIteration, double learningRate, int layerCount, int[] nodeCounts)
	{
		setNumberOfEpochs(numberOfEpochs);
		
		setEpochIteration(epochIteration);
		
		setLearningRate(learningRate);
		
		Layer layer;
		
		// First we'll instantiate each Layer (as a Sigmoid to start, then maybe modify them later)
		// And now at this point we have Tanh Layers as well.  So our hidden Layers will be Tanh, while
		// the output Layer will be Sigmoid.
		for (int i = 0; i < layerCount; i++)
		{
			if (i < layerCount - 1)
			{
				layer = (Layer) new TanhLayer(nodeCounts[i]);
			}
			else
			{
				layer = (Layer) new SoftmaxLayer(nodeCounts[i]);
			}
			
			if (i == 0)
			{
				for (Neuron neuron: layer.getNeurons())
				{
					// Input Neurons don't need a bias.
					neuron.setBias(0);
				}
			}
			
			// For all Layers but the output Layer, add a Bias Neuron
			if (i < (layerCount - 1))
			{
				//layer.addNeuron(new Neuron(true));
			}
			
			addLayer(layer);
		}
		
		setLayerCount(layerCount);
		
		setNodeCounts(nodeCounts);
	}
	
	// The original way I made Connections was broken, to the extreme.  I created a list of Connections
	// between a left Neuron and several right Neurons and then I would add that list to the Neuron 
	// as an outbound Connection list.  Then I repeated the process for incoming Connections.
	// This meant that Connections which should have been references to a single Connection were not, 
	// causing all tons of math problems later.  Time to re-do the logic by iterating over both sets of 
	// existing Neurons, creating a Connection between the two, and then adding that Connection
	// to each Neuron using as a new function I just created for the Neuron class.
	public void initiateConnections()
	{
		Connection tempConnection;
		Layer currentLayer;
		Layer otherLayer;
		
		for (int i = 0; i < layers.size() - 1; i++)
		{
			currentLayer = layers.get(i);
			
			otherLayer = layers.get(i+1);
			
			for (Neuron leftNeuron: currentLayer.getNeurons())
			{
				for (Neuron rightNeuron: otherLayer.getNeurons())
				{
					// Create a new Connection between two Neurons.
					// This constructor initializes the weight to a randomized value
					tempConnection = new Connection(leftNeuron, rightNeuron);
					
					// Is it really this easy?  Don't mean to jinx it before running it,
					// but hopefully this is it.  Creating a Connection between these 
					// two Neurons and then adding it to each?  We'll see if this works.
					// And if it solves my 4 day problem!
					leftNeuron.addOutgoingConnection(tempConnection);
					rightNeuron.addIncomingConnection(tempConnection);
				}
			}
		}
	}
	
	// This train function is called when we will use the Helper class to generate
	// our inputs/expectedOutputs training data.  Each epoch, we'll generate x 
	// number of rows of training data.  And at this point, we're going to train the network
	// on 4 concurrent functions: sin(x), cos(x), x^2, and abs(x).  
	public void train(int numberOfTrainingRows)
	{
		double[][] inputs = new double[numberOfTrainingRows][1];
		double[][] expectedOutputs = new double[numberOfTrainingRows][4];
		
		helper = Helper.getInstance();
		
        // Iterate over the number of epochs
		for (int epoch = 0; epoch < numberOfEpochs; epoch++) 
        {
			// Iterate over the number of input neurons
            for (int inputIndex = 0; inputIndex < numberOfTrainingRows; inputIndex++) 
            {
            	for (int i = 0; i < numberOfTrainingRows; i++)
            	{
            		inputs[i][0] = new Random().nextDouble() * 2 - 1;
            		
            		expectedOutputs[i] = helper.generate4FunctionOutputs(inputs[i][0]);
            	}
            	
            	if (printDebug)
            	{
            		printDebug(0, epoch, inputs, inputIndex, expectedOutputs);
            	}
            	
            	// This will iterate over all the Layers
            	forwardFeed(inputs[inputIndex]);
            	
            	if (printDebug)
            	{
            		printDebug(1, epoch, inputs, inputIndex, expectedOutputs);
            	}
                
            	// Now that all the Neurons have activation values saved, it's time to compare
            	// the final layer to the expected outputs (from the input to this function). 
            	// This call will update the delta values of all Neurons in all Layers except the input
            	// Layer (where the deltas don't matter).  The delta value is calculated by the Layer 
            	// itself using whatever strategy that Layer implements.
            	backPropagate(expectedOutputs[inputIndex]);
            	
            	if (printDebug)
            	{
            		printDebug(2, epoch, inputs, inputIndex, expectedOutputs);
            	}
            	
            	// Now that the deltas are updated across the board, it's time to use those deltas
            	// to adjust the weights of Connections and the biases of Neurons.
            	updateWeightsAndBiases(learningRate);
            	
            	if (printDebug)
            	{
            		printDebug(3, epoch, inputs, inputIndex, expectedOutputs);
            	}
            }
            
            calculateAndSaveMSE(inputs, expectedOutputs);
            
            if (epoch % epochIteration == 0)
            {
            	printInfoAtEpochIteration(epoch, inputs, expectedOutputs);
            }
        }	
	}
	
	
	// This train function is called when the inputs/expectedOutputs are 
	// passed in from the calling main() function.  Useful for small training sets.
	// Now adding logic for batch processing.
	public void train(double[][] inputs, double[][] expectedOutputs)
	{
        // Iterate over the number of epochs
		for (int epoch = 0; epoch < numberOfEpochs; epoch++) 
        {
			ArrayList<Integer> shuffledInputs = new ArrayList<>();
			
			for (int i = 0; i < inputs.length; i++) 
            {
				shuffledInputs.add(i);
            }
			
			Collections.shuffle(shuffledInputs);
			
			int inputIndex;
			
			int actualBatchSize;
			
			// Iterate over the number of input neurons
            for (int j = 0; j < shuffledInputs.size(); j += getBatchSize()) 
            {
            	if (isBatch())
            	{
            		actualBatchSize = Math.min(getBatchSize(), shuffledInputs.size() - j);
            		
            		for (int i = 0; i < actualBatchSize; i++)
            		{
    	            	// There's no guarantee we'll have an exact number of batches per input collection, so 
    	            	// check here if we're at the end.  If so, end the loop (and also set the actualBatchSize
    	            	// to the number of training pairs we actually iterated over here.
            			inputIndex = shuffledInputs.get(j + i);
                		
    	            	// This will iterate over all the Layers
    	            	forwardFeed(inputs[inputIndex]);
    	            	
    	            	// Now that all the Neurons have activation values saved, it's time to compare
    	            	// the final layer to the expected outputs (from the input to this function). 
    	            	// This call will update the delta values of all Neurons in all Layers except the input
    	            	// Layer (where the deltas don't matter).  The delta value is calculated by the Layer 
    	            	// itself using whatever strategy that Layer implements.
    	            	backPropagate(expectedOutputs[inputIndex]);    
    	            	
    	            	// Now that the deltas are updated across the board, since this is batch, it's time
    	            	// to accumulate the values.  Those accumulations will be used after this iteration
    	            	// of batch to then update the weights and biases.
    	            	accumulateGradients();
    	            }
	            	
	            	// Now that the accumulated deltas and gradients are updated across the board, 
	            	// it's time to use those deltas to adjust the weights of Connections and the biases of Neurons.
	            	updateWeightsAndBiases(learningRate, actualBatchSize);
            	}
            	else
            	{
	            	inputIndex = shuffledInputs.get(j);
	            	
	            	if (printDebug)
	            	{
	            		printDebug(0, epoch, inputs, inputIndex, expectedOutputs);
	            	}
	            	
	            	// This will iterate over all the Layers
	            	forwardFeed(inputs[inputIndex]);
	            	
	            	if (printDebug)
	            	{
	            		printDebug(1, epoch, inputs, inputIndex, expectedOutputs);
	            	}
	                
	            	// Now that all the Neurons have activation values saved, it's time to compare
	            	// the final layer to the expected outputs (from the input to this function). 
	            	// This call will update the delta values of all Neurons in all Layers except the input
	            	// Layer (where the deltas don't matter).  The delta value is calculated by the Layer 
	            	// itself using whatever strategy that Layer implements.
	            	backPropagate(expectedOutputs[inputIndex]);
	            	
	            	if (printDebug)
	            	{
	            		printDebug(2, epoch, inputs, inputIndex, expectedOutputs);
	            	}
	            	
	            	// Now that the deltas are updated across the board, it's time to use those deltas
	            	// to adjust the weights of Connections and the biases of Neurons.
	            	updateWeightsAndBiases(learningRate);
	            	
	            	if (printDebug)
	            	{
	            		printDebug(3, epoch, inputs, inputIndex, expectedOutputs);
	            	}
            	}
            }
            
            calculateAndSaveMSE(inputs, expectedOutputs);
            
            if (epoch % epochIteration == 0)
            {
            	printInfoAtEpochIteration(epoch, inputs, expectedOutputs);
            }
        }
	}

	private void calculateAndSaveMSE(double[][] inputs, double[][] expectedOutputs)
	{
	    int outputCount = layers.get(layers.size() - 1).getNeurons().size();
	    double[] totalLossPerOutput = new double[outputCount];
	    
	    HashMap<Integer, Double> msePerOutput = new HashMap<>();
	    
	    for (int j = 0; j < inputs.length; j++) 
	    {
	        forwardFeed(inputs[j]);
	        
	        for (int k = 0; k < outputCount; k++) 
	        {
	        	double actual = layers.get(layers.size() - 1).getNeurons().get(k).getOutputValue();
	            double expected = expectedOutputs[j][k];
	            totalLossPerOutput[k] += Math.pow(actual - expected, 2);
	        }
	    }
	    
	    for (int k = 0; k < outputCount; k++) 
	    {
	    	// Average the total loss per output by inputs
	        totalLossPerOutput[k] /= inputs.length; 
	        
	        msePerOutput.put(k, totalLossPerOutput[k]);
	    }
	   
	    mseList.add(msePerOutput);
	}

	
    private void forwardFeed(double[] inputs) 
    { 
    	// First set the output values of the first layer of neurons to match the inputs, 
    	// since the first layer is really the input layer.
    	for (int i = 0; i < layers.get(0).getNeurons().size(); i++)
    	{
    		if (!layers.get(0).getNeurons().get(i).isBias())
    		{
    			layers.get(0).getNeurons().get(i).setOutputValue(inputs[i]);
    		}
    	}
    	
    	// Now iterate over the 2nd layer and beyond to forward the activation values
    	// to the next layer.
    	for (int i = 1; i < layers.size(); i++)
    	{
			layers.get(i).feedForward();
    	}    	
    }
    
    private void backPropagate(double[] expected) 
    {
    	// Let's start with the final layer and calculate (and store) the delta values based on the expected output.
    	layers.get(layers.size() - 1).backwardDeltaCalculateOutputLayer(expected);
    	
    	// Now that those Neurons have deltas calculated and stored, I can iterate over the rest of the 
    	// Layers and call each layer's backward function to calculate the delta.
    	for (int i = layers.size() - 2; i > 0; i--)
    	{
    		layers.get(i).backwardDeltaCalculateOtherLayers();
    	}
    	
    	// Now calculate and store the gradients for all the Connections.  These gradients will be pulled
    	// later during batch processing.  
        calculateGradients();    	
    }
    
    public void accumulateGradients()
    {
    	// Since we're in batch mode, we're going to iterate over the Layers as if we were updating
    	// weights and biases. Except instead of updating the actual values, we're going to add current
    	// gradients to the gradient accumulators for both weights (in Connections) and biases (in Neurons).
    	for (int i = 0; i < layers.size(); i++)
    	{
    		layers.get(i).accumulateGradients();
    	}
    }

    public void calculateGradients()
    {
    	for (Layer layer : layers) 
        {
            for (Neuron neuron : layer.getNeurons()) 
            {
                for (Connection connection : neuron.getOutgoing()) 
                {
                    double grad = neuron.getOutputValue() * connection.getTo().getDelta();
                    connection.setGradient(grad);
                }
            }
        }   	
    }
    
    public void updateWeightsAndBiases(double learningRate) 
    { 
    	// Let's iterate over the Layers going forward (starting at the first Layer).
    	// We only needed to go backwards to calculate deltas, and that's done already.  So now
    	// we can go forward and call each Layer's update function for weights and biases, passing in the 
    	// learning value.  I need to start at the first Layer (input Layer) as the Layer logic
    	// will iterate over the Connections between this and the next Layer and pull the "to" Neuron's
    	// delta value for the calculation.  
    	for (int i = 0; i < layers.size(); i++)
    	{
    		layers.get(i).updateWeightsAndBiases(learningRate);
    	}
    }
    
    public void updateWeightsAndBiases(double learningRate, int actualBatchSize) 
    { 
    	// Let's iterate over the Layers going forward (starting at the first Layer).
    	// We only needed to go backwards to calculate deltas, and that's done already.  So now
    	// we can go forward and call each Layer's update function for weights and biases, passing in the 
    	// learning value.  I need to start at the first Layer (input Layer) as the Layer logic
    	// will iterate over the Connections between this and the next Layer and pull the "to" Neuron's
    	// delta value for the calculation.  
    	for (int i = 0; i < layers.size(); i++)
    	{
    		layers.get(i).updateWeightsAndBiasesBatch(learningRate, actualBatchSize);
    	}
    }
    
    public void exportWeightsAndBiases()
    {
    	NetworkReaderWriter nrw = new NetworkReaderWriter();
    	
    	try
    	{
    		LocalDateTime currentDateTime = LocalDateTime.now();
    		
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd-HH.mm.ss");
            
            String formattedDateTime = currentDateTime.format(formatter);
    		
            nrw.saveNetwork("/Users/craigadelhardt/Documents/NeuralNetworkExport/" + formattedDateTime + ".json", nodeCounts, layers);
    	}
    	catch(Exception e)
    	{
    		System.out.println("Problem writing to file");
    		System.out.println(e);
    	}
    }
    
    public boolean loadExistingNetwork(String filename)
    {
    	NetworkReaderWriter nrw = new NetworkReaderWriter();
    	
    	try
    	{
    		nrw.loadNetwork(filename, nodeCounts, layers);
    	}
    	catch(IllegalArgumentException e)
    	{
    		System.out.println("Network in file doesn't match network in code");
    		System.out.println(e);
    		
    		return false;
    	}
    	catch(IOException e)
    	{
    		System.out.println("Problem reading file");
    		System.out.println(e);
    		
    		return false;
    	}
    	
    	return true;
    }
    
    // Block of printing functions
	private void printDebug(int step, int epoch, double[][] inputs, int inputIndex, double[][] expectedOutputs)
	{
		if (step == 0)
		{
        	System.out.println("************* OOP Version *************");
        	System.out.println("************* DEBUGGING DATA BEGIN *************");
        	System.out.printf("Epoch number %d%n", epoch);
        	System.out.printf("%n$$$$$$$$$$$$$$$%nAbout to run the feedforward for input [%f, %f]%n", inputs[inputIndex][0], inputs[inputIndex][1]);
        	printNetwork();

		}
		else if (step == 1)
		{
    		System.out.printf("%n$$$$$$$$$$$$$$$%nResults from feedforward for input [%f, %f]%n", inputs[inputIndex][0], inputs[inputIndex][1]);
    		printNetwork();
		}
		else if (step == 3)
		{
        	System.out.printf("%n$$$$$$$$$$$$$$$%nResults from backprop for input [%f, %f]%n", inputs[inputIndex][0], inputs[inputIndex][1]);
        	printNetwork();
		}
		else if (step == 4)
		{
        	System.out.printf("%n$$$$$$$$$$$$$$$%nResults from updating weights and biases for input [%f, %f]%n", inputs[inputIndex][0], inputs[inputIndex][1]);
        	printNetwork();
        	System.out.println("Printing results data (which will include all 4 inputs)");
        	printResults(inputs,  expectedOutputs);
        	System.out.println("************* DEBUGGING DATA END *************");	
		}
	}
	
	public void printNetwork()
	{
		System.out.println("\n*****************************************");
		System.out.printf("This network has %d layers.%n", layerCount);
		int layerCounter = 0;
		
		for (Layer layer: layers)
		{
			System.out.printf("%nLayer number %d:%n", layerCounter++);
			layer.printLayer();
		}
	}
	
	public void printMNISTResults(double[][] trainingInputs, double[][] trainingOutputs)
	{
		double[] input;
		
		System.out.println();
		
        for (int i = 0; i < trainingInputs.length; i++) 
        {
        	input = trainingInputs[i];
        	
        	forwardFeed(input);
        	
        	System.out.print("Input :");
        	MNISTLoader.printImageASCII(input);
        	System.out.print("\n");
        	
            System.out.printf("Expect: [");
        	for (double output: trainingOutputs[i])
        	{
        		System.out.printf("%.6f ", output);
        	}
        	System.out.print("]\n");
            
            System.out.printf("Actual: [");
            for (Neuron n: layers.get(layers.size() - 1).getNeurons())
            {
            	System.out.printf("%.6f ", n.getOutputValue());
            }
            System.out.printf("]%n%n");
        }		
	}
	
	public void printResults(double[][] trainingInputs, double[][] trainingOutputs)
	{
		double[] input;
		
		System.out.println();
		
        for (int i = 0; i < trainingInputs.length; i++) 
        {
        	input = trainingInputs[i];
        	
        	forwardFeed(input);
        	
        	System.out.print("Input : [");
        	for (double inputItem: input)
        	{
        		System.out.print(inputItem + " ");
        	}
        	System.out.print("]\n");
        	
            System.out.printf("Expect: [");
        	for (double output: trainingOutputs[i])
        	{
        		System.out.printf("%.6f ", output);
        	}
        	System.out.print("]\n");
            
            System.out.printf("Actual: [");
            for (Neuron n: layers.get(layers.size() - 1).getNeurons())
            {
            	System.out.printf("%.6f ", n.getOutputValue());
            }
            System.out.printf("]%n%n");
        }
	}	
	
	private void printInfoAtEpochIteration(int epoch, double[][] inputs, double[][] expectedOutputs)
	{
    	
    	//System.out.printf("%n%nDetails for Epoch %d:%n", epoch);
    	
    	//printNetwork();

    	//printResults(inputs,  expectedOutputs);
    	
    	System.out.println("Epoch " + epoch + ", MSE: " + mseList.get(epoch));
    	
    	//System.out.println("--------------------------");
	}
    
	// This was used to initiate a Network that could be deterministic and comparable to the 
	// non-OOP version of my code that I created and worked already.  This was helpful for stepping
	// through the system as it ran in parallel with watching the non-OOP system run.  This is 
	// useful only for XOR learning with inputs of {[0,0],[0,1],[1,0],[1,1]} and will produce
	// a successful output.  
	private void seedNetwork()
	{
		// Weights From Input (Layer 0) To Hidden Layer 0 (Layer 1)
	    layers.get(0).getNeurons().get(0).getOutgoing().get(0).setWeight(-0.49374066664118454);
	    layers.get(0).getNeurons().get(0).getOutgoing().get(1).setWeight( 0.21387611998075173);
	    layers.get(0).getNeurons().get(0).getOutgoing().get(2).setWeight(-0.09941186582161810);
	    layers.get(0).getNeurons().get(0).getOutgoing().get(3).setWeight( 0.32897686226310420);

	    layers.get(0).getNeurons().get(1).getOutgoing().get(0).setWeight( 0.60240370691785890);
	    layers.get(0).getNeurons().get(1).getOutgoing().get(1).setWeight( 0.09450286710529077);
	    layers.get(0).getNeurons().get(1).getOutgoing().get(2).setWeight(-0.55748926858078860);
	    layers.get(0).getNeurons().get(1).getOutgoing().get(3).setWeight( 0.10304643409175185);

	    // Weights From Hidden Layer 0 (Layer 1) To Hidden Layer 1 (Layer 2)
	    layers.get(1).getNeurons().get(0).getOutgoing().get(0).setWeight(-0.41118280924008930);
	    layers.get(1).getNeurons().get(0).getOutgoing().get(1).setWeight( 0.61872656885185020);

	    layers.get(1).getNeurons().get(1).getOutgoing().get(0).setWeight( 0.31686833046069050);
	    layers.get(1).getNeurons().get(1).getOutgoing().get(1).setWeight(-0.38653245202219066);

	    layers.get(1).getNeurons().get(2).getOutgoing().get(0).setWeight( 0.31265739503449486);
	    layers.get(1).getNeurons().get(2).getOutgoing().get(1).setWeight( 0.80426453911883720);

	    layers.get(1).getNeurons().get(3).getOutgoing().get(0).setWeight(-0.61292677330547750);
	    layers.get(1).getNeurons().get(3).getOutgoing().get(1).setWeight(-0.47900854555162420);

	    // Weights From Hidden Layer 1 (Layer 2) To Output (Layer 3)
	    layers.get(2).getNeurons().get(0).getOutgoing().get(0).setWeight(-0.70007024458959140);
	    layers.get(2).getNeurons().get(1).getOutgoing().get(0).setWeight( 0.76015925454438230);
	    
	    // Hidden Layer 0 biases
	    layers.get(1).getNeurons().get(0).setBias(0.78874876894524130);
	    layers.get(1).getNeurons().get(1).setBias(0.98130160389828250);
	    layers.get(1).getNeurons().get(2).setBias(-0.91000699720043880);
	    layers.get(1).getNeurons().get(3).setBias(-0.76130048884048930);

	    // Hidden Layer 1 biases
	    layers.get(2).getNeurons().get(0).setBias(0.72773747552058250);
	    layers.get(2).getNeurons().get(1).setBias(-0.37898926599938830);

	    // Output Layer biases
	    layers.get(3).getNeurons().get(0).setBias(0.26178720340242423);
	}
	
	public int getNumberOfEpochs() 
	{
		return numberOfEpochs;
	}

	public void setNumberOfEpochs(int numberOfEpochs) 
	{
		this.numberOfEpochs = numberOfEpochs;
	}

	public int getEpochIteration() 
	{
		return epochIteration;
	}

	public void setEpochIteration(int epochIteration) 
	{
		this.epochIteration = epochIteration;
	}

	public double getLearningRate() 
	{
		return learningRate;
	}

	public void setLearningRate(double learningRate) 
	{
		this.learningRate = learningRate;
	}

	public List<Layer> getLayers() 
	{
		return layers;
	}

	public void setLayers(List<Layer> layers) 
	{
		this.layers = layers;
	}

	public void addLayer(Layer layer)
	{
		layers.add(layer);
	}

	public int getLayerCount() {
		return layerCount;
	}

	public void setLayerCount(int layerCount) {
		this.layerCount = layerCount;
	}

	public int[] getNodeCounts() 
	{
		return nodeCounts;
	}

	public void setNodeCounts(int[] nodeCounts) 
	{
		this.nodeCounts = nodeCounts;
	}
	
	public ArrayList<HashMap<Integer, Double>> getMseList()
	{
		return mseList;
	}
	
	public boolean isBatch() {
		return isBatch;
	}

	public void setBatch(boolean isBatch) {
		this.isBatch = isBatch;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}
}
