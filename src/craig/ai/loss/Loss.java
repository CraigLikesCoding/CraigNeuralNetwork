package craig.ai.loss;

import craig.ai.helpers.Helper.LossMode;

public interface Loss 
{
	LossMode getMode();
	
	default double loss(double expected, double actual) 
	{
        throw new UnsupportedOperationException("Scalar loss not implemented");
    }
	
	default double loss(double[] expected, double[] actual) 
	{
        throw new UnsupportedOperationException("Vector loss not implemented");
    }
}
