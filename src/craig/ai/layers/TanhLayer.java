package craig.ai.layers;

import craig.ai.GenericLayer;

public class TanhLayer extends GenericLayer implements Layer
{
	public TanhLayer(int neuronCount)
	{
		super(neuronCount);
	}
	
	
	@Override
	public double applyActivation(double x) 
	{
		return Math.tanh(x);
	}
	
	private double tanhDerivativeFromOutput(double tanhOutput) 
	{
        return 1.0 - (tanhOutput * tanhOutput);
    }
	
	public double calculateHiddenDelta(double sumOfDeltas, double actualOutput)
	{
		// This is called for hidden Layers.  It takes the weighted sum (which is summed up elsewhere)
		// and multiplies that by the derivative of the Tanh.
		
		return tanhDerivativeFromOutput(actualOutput) * sumOfDeltas;
	}
	
	public double calculateOutputDelta(double expectedOutput, double actualOutput)
	{
		// This calculates the error and multiplies it by the derivative of the Tanh function
		// Used only for output Layers
		double error = actualOutput - expectedOutput;
		
        return error * tanhDerivativeFromOutput(actualOutput);
	}
	
	public String getId()
	{
		return "TanhLayer_" + id;
	}
}
