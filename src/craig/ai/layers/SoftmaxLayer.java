package craig.ai.layers;

import java.util.Arrays;

import craig.ai.GenericLayer;

public class SoftmaxLayer extends GenericLayer implements Layer
{

	public SoftmaxLayer(int neuronCount) 
	{
		super(neuronCount);
	}

	@Override
	public void feedForward()
	{
		// In the GenericLayer, this function iterates over all the Neurons in this Layer and calls
		// the applyActivation for each weighted sum, one at a time.  However in SoftmaxLayer, that's not
		// the case.  Instead, we iterate over the Neurons and store each weighted sum in an array.
		// That array then gets passed to SoftmaxLayer's new applyActivation method, which requires
		// the weighted sums of every Neuron at this Layer.  The output will be an array of activated
		// values, which we'll then place in each Neuron at this Layer.  

		double[] weightedSums = new double[neurons.size()];
		
		double[] activationValues;
				
		for (int i = 0; i < neurons.size(); i++) 
		{
			// Only do this if it's a regular Neuron.  A Bias Neuron doesn't change its output value.
			if (!neurons.get(i).isBias())
			{
				weightedSums[i] = neurons.get(i).calculateWeightedSum();
			}
        }
		
		activationValues = applyActivation(weightedSums);
		
		for (int i = 0; i < neurons.size(); i++) 
		{
			// Only do this if it's a regular Neuron.  A Bias Neuron doesn't change its output value.
			if (!neurons.get(i).isBias())
			{
				neurons.get(i).setOutputValue(activationValues[i]);
			}
        }
	}
	
	@Override
	public double applyActivation(double x) 
	{
		throw new UnsupportedOperationException("SoftmaxLayer uses vector-level activation; use applyActivation(double[] x) instead.");
	}

	public double[] applyActivation(double[] weightedSums)
	{
		// SoftmaxLayer is different than Tanh or Sigmoid.  Those methods activate at a per Neuron basis.  However,
		// SoftmaxLayer does it's calculation based on all the Neurons in this Layer.  As of this implementation,
		// this will only be for the output Layer.  
		
		// First up, get the maximum value of the weighted sums passed in
		double maxWeightedSum = Arrays.stream(weightedSums).max().getAsDouble();
		
		// Now we need our variables to store values as we process:
		double sumExponentials = 0;
		double[] exponentials = new double[weightedSums.length];
		
		// Now we'll loop and exponentiate:
		for (int i = 0; i < weightedSums.length; i++) 
		{
			// We subtract out the biggest weighted sum from the current weighted sum to prevent 
			// large exponents from causing overflow.  We're raising e to the power of
			// (weightedSums[i] - maxWeightedSum).
	        exponentials[i] = Math.exp(weightedSums[i] - maxWeightedSum);
	        
	        // Now we keep a running sum of all the exponentials.
	        sumExponentials += exponentials[i];
	    }
		
		// Now we'll loop through again and divide each exponential by the sum of all exponentials.  
		// These values are what will be returned from this function to be assigned to the 
		// output Neuron output values.
		for (int i = 0; i < weightedSums.length; i++)
		{
			exponentials[i] /= sumExponentials;
		}
		
		return exponentials;
	}
	
	@Override
	public double calculateHiddenDelta(double sumOfDeltas, double actualOutput) 
	{
		// This is called for hidden Layers.  But for Softmax, we very rarely use it outside of the 
		// output Layer.  So we'll just have this throw an exception to avoid it being called.
		
		throw new UnsupportedOperationException("SoftmaxLayer should not be called from a hidden Layer (at this time).");
	}

	@Override
	public double calculateOutputDelta(double expectedOutput, double actualOutput) 
	{
		// This calculates the error and multiplies it by the derivative of the activation function.
		// Used only for output Layers.
		// Believe it or not, there's no derivative needed here.  We just subtract the expecteOutput
		// from the actualOutput and return that.  For Softmax + categorical cross-entropy, derivative 
		// simplifies to (y_pred - y_true)
		return (actualOutput - expectedOutput);
	}

	public String getId()
	{
		return "SoftmaxLayer_" + id;
	}	
}
