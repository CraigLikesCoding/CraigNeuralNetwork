package craig.ai.layers;

import java.util.ArrayList;
import java.util.List;

import craig.ai.Connection;
import craig.ai.Neuron;
import craig.ai.helpers.Helper;
import craig.ai.loss.Loss;

public abstract class GenericLayer implements Layer 
{
	private Helper helper;
	
	protected Loss loss;
	
	protected int id;
	
	// This class implements the generic functions for a Layer.  A class that extends
	// this abstract class will implement the activation and delta methods that are 
	// specific to the type of activation strategies for it.
	
	protected List<Neuron> neurons = new ArrayList<Neuron>();

	public GenericLayer(int neuronCount)
	{
		Neuron neuron;
		
		for (int i = 0; i < neuronCount; i++)
		{
			neuron = new Neuron();
			
			neurons.add(neuron);
		}
		
		helper = Helper.getInstance();
		
		id = helper.getCounter();
	}
	
	public void feedForward()
	{
		// Here we'll iterated over each Neuron in this Layer.  For each Neuron, call its 
		// function to calculate the weighted sum, which will iterate over the Connections between 
		// the prior layer and this one.  
		for (int i = 0; i < neurons.size(); i++) 
		{
			// Only do this if it's a regular Neuron.  A Bias Neuron doesn't change its output value.
			if (!neurons.get(i).isBias())
			{
				neurons.get(i).setOutputValue(applyActivation(neurons.get(i).calculateWeightedSum()));
			}
        }
	}
	
	public void backwardDeltaCalculateOtherLayers() 
	{
		double sum = 0.0;
		
		// Here we'll iterate over all Neurons in this Layer.  For each Neuron, grab the outbound Connection.
		// With that Connection, iterate over all Neurons and pull their delta values.  Multiply the delta
		// by the weight of the connection and then add that to a running sum.
		for (Neuron neuron : neurons) 
		{
			sum = 0.0;
			
			for (Connection outgoingConnection: neuron.getOutgoing())
			{
				sum += outgoingConnection.getWeight() * outgoingConnection.getTo().getDelta();
			}
			
			// Now that we've got a sum of weights times deltas, we can apply the derivative of whatever 
			// activation methods are for this Layer to calculate the delta for this Neuron.  
			neuron.setDelta(calculateHiddenDelta(sum, neuron.getOutputValue()));
		}
	}
	
	public void backwardDeltaCalculateOutputLayer(double[] expectedOutput) 
	{
      	// First thing is to iterate through the Neurons in this layer and calculate and set the 
      	// delta value (which is basically the derivative of the the activation) for these Neurons.  
    	for (int i = 0; i < neurons.size(); i++)
    	{    		
    		neurons.get(i).setDelta(calculateOutputDelta(expectedOutput[i], neurons.get(i).getOutputValue()));
    	}
	}
	
	public void updateWeightsAndBiases(double learningRate) 
	{	 
    	// Let's iterate over each Neuron in this Layer.  For each Neuron, I'll first update its bias with the 
    	// formula bias -= learningRate * delta;.  Then also for each Neuron, I'll iterate over 
    	// its outgoingConnection array and modify the weight of each Connection based on the left 
    	// Neuron's output and the delta of the right Neuron (weight -= learningRate * delta_B * output_A;).  
    	double bias = 0.0;
    	
    	double weight;
    	
		for (Neuron neuron: neurons)
    	{
    		bias = neuron.getBias();
    		
    		bias -= learningRate * neuron.getDelta();
    		
    		neuron.setBias(bias);
    		
    		for (Connection connection: neuron.getOutgoing())
    		{
    			weight = connection.getWeight();
    			
    			weight -= learningRate * neuron.getOutputValue() * connection.getTo().getDelta();
    			
    			connection.setWeight(weight);
    		}
    	}
	}
	
    public void updateWeightsAndBiasesBatch(double learningRate, int actualBatchSize)
    {
    	// This function is called after a batch of training is complete.  At this point, all
    	// Connections and all Neurons have accumulatedGradient values stored inside.  Now 
    	// we have to take those accumulations and average them out before applying them
    	// to the Neuron bias value and Connection weight value
    	for (Neuron neuron: neurons)
    	{
    		double averageGradient = neuron.getAccumulatedGradient() / actualBatchSize;
    		neuron.setBias(neuron.getBias() - learningRate * averageGradient);
    		
    		neuron.resetAccumulatedGradient();
    		
    		for (Connection connection: neuron.getOutgoing())
    		{
    			double averageWeight = connection.getAccumulatedGradient() / actualBatchSize;
    			connection.setWeight(connection.getWeight() - learningRate * averageWeight);
    			
    			connection.resetAccumulatedGradient();
    		}
    	}
    }
	
	public void accumulateGradients()
	{
		// This function is called during batch processing mode.  In batch mode, instead of immediately 
	    // updating weights and biases after each training sample, we accumulate their gradients 
	    // across the entire batch. These accumulated values will later be averaged (or summed) 
		// and applied during the weight/bias update step.  To achieve this, we'll iterate over each Neuron
		// in this Layer and accumulate the bias gradient from the Neuron's delta value.  
		for (Neuron neuron: neurons)
		{
			neuron.addToAccumulatedGradient();
			
			for (Connection connection: neuron.getOutgoing()) 
			{
				connection.addToAccumulatedGradient();
			}
		}
	}
	
	public void printLayer()
	{
		System.out.printf("This %s layer has %d Neurons:%n", this.getClass(), neurons.size());
		
		int neuronCounter = 0;
		
		for (Neuron neuron: neurons)
		{
			System.out.printf("%nNeuron number %d:%n", neuronCounter++);
			
			neuron.printNeuron();
		}
	}
	
	public void setNeurons(List<Neuron> neuronsInput)
	{
		neurons = neuronsInput;
	}
	
	public List<Neuron> getNeurons()
	{
		return neurons;
	}

	public void addNeuron(Neuron neuron)
	{
		neurons.add(neuron);
	}

	public Loss getLoss() 
	{
		return loss;
	}

	public void setLoss(Loss loss) 
	{
		this.loss = loss;
	}

}
