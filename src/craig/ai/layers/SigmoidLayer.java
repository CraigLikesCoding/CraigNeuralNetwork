package craig.ai.layers;

import craig.ai.helpers.Helper;
import craig.ai.helpers.Helper.LossMode;
import craig.ai.loss.ScalarLoss;

public class SigmoidLayer extends GenericLayer implements Layer 
{
	// This implementation of Layer is for sigmoid activation
	
	public SigmoidLayer(int neuronCount)
	{
		super(neuronCount);
	}

	public double applyActivation(double input) 
	{
		// This is the sigmoid function
		
		return 1.0 / (1.0 + Math.exp(-input));
	}
	
	private double sigmoidDerivativeFromOutput(double sigmoidOutput)
	{
		// This is the derivative of sigmoid, which is called during back propagation
		return sigmoidOutput * (1 - sigmoidOutput);
		
	}
	
	// This is defined at the GenericLayer level, but I'm overriding it here because SigmoidLayer
	// has the option of 
	@Override
	public void backwardDeltaCalculateOutputLayer(double[] expectedOutput) 
	{
      	// First thing is to iterate through the Neurons in this layer and calculate and set the 
      	// delta value (which is basically the derivative of the the activation) for these Neurons.  
    	for (int i = 0; i < neurons.size(); i++)
    	{    		
    		neurons.get(i).setDelta(calculateOutputDelta(expectedOutput[i], neurons.get(i).getOutputValue()));
    	}
	}
	
	public double calculateOutputDelta(double expectedOutput, double actualOutput)
	{
		double lossDerivative = 0;
		
		// Output Layers can be scalar or vector based.  Sigmoid will only ever be scalar, so if any 
		// other mode is set, then throw an exception.
		switch(getLossMode()) 
		{
			case Helper.LossMode.SCALAR:
			{
				// This calculates the error and multiplies it by the derivative of the sigmoid function
				// Used only for output Layers
				// Calculate the Loss derivative first, with respect to the expected value (dL/dy)
				lossDerivative = ((ScalarLoss)loss).lossDerivative(expectedOutput, actualOutput);
				
				break;
			}
			default:
			{
				throw new IllegalStateException("Unexpected loss mode: " + getLossMode());
			}
				
		}
		
		// Now calculate the activation derivative (dy/dz)
		double activationDerivative = sigmoidDerivativeFromOutput(actualOutput);
		
		// Now apply the chain rule by multiplying the two derivatives together
		return (lossDerivative * activationDerivative);
	}
	
	public double calculateHiddenDelta(double sumOfDeltas, double actualOutput)
	{
		// This is called for hidden Layers.  It takes the sum of weighted deltas
		// and multiplies that by the derivative of the sigmoid.
		
		return sigmoidDerivativeFromOutput(actualOutput) * sumOfDeltas;
	}

	public String getId()
	{
		return "SigmoidLayer_" + id;
	}
	
	public LossMode getLossMode()
	{
		return loss.getMode();
	}
}
