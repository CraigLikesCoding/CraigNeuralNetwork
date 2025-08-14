package craig.ai.layers;

import craig.ai.GenericLayer;

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
	
	public double calculateOutputDelta(double expectedOutput, double actualOutput)
	{
		// This calculates the error and multiplies it by the derivative of the sigmoid function
		// Used only for output Layers
		double error = actualOutput - expectedOutput;
		
		return (error * sigmoidDerivativeFromOutput(actualOutput));
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
}
