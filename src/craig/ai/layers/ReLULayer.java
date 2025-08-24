package craig.ai.layers;

import craig.ai.helpers.Helper.LossMode;

public class ReLULayer extends GenericLayer implements Layer
{
	public ReLULayer(int neuronCount)
	{
		super(neuronCount);
	}
	
	
	@Override
	public double applyActivation(double x) 
	{
		return Math.max(0, x);
	}
	
	private double reLUDerivativeFromOutput(double reLUOutput) 
	{
		return reLUOutput > 0 ? 1 : 0;
    }
	
	public double calculateHiddenDelta(double sumOfDeltas, double actualOutput)
	{
		// This is called for hidden Layers.  It takes the weighted sum (which is summed up elsewhere)
		// and multiplies that by the derivative of the ReLU.
		
		return reLUDerivativeFromOutput(actualOutput) * sumOfDeltas;
	}
	
	public double calculateOutputDelta(double expectedOutput, double actualOutput)
	{
		// This calculates the error and multiplies it by the derivative of the ReLU function
		// Used only for output Layers
		double error = actualOutput - expectedOutput;
		
        return error * reLUDerivativeFromOutput(actualOutput);
	}
	
	public String getId()
	{
		return "ReLULayer_" + id;
	}

	@Override
	public LossMode getLossMode() 
	{
		return getLoss().getMode();
	}
}
