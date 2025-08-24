package craig.ai.helpers;

import java.util.Random;

// Singleton helper class to retrieve random numbers for initialization 
// as well as to provide unique ID's for all objects to make debugging easier
// When you need it, just call Helper.getInstance().  No need to instantiate it yourself.

public class Helper 
{
	private static volatile Helper instance;
	
	private Random random = new Random();
	
	private int counter = 0;
	
	private Helper()
	{
		if (instance != null)
		{
			throw new RuntimeException("Use getInstance to get the singleton of this Helper class.");
		}
	}
	
	public static Helper getInstance()
	{
		if (instance == null)
		{
			synchronized(Helper.class)
			{
				if (instance == null)
				{
					instance = new Helper();
				}
			}
		}
		
		return instance;
	}
	
	public double getRandomWeight()
	{
		return random.nextDouble() * 2 - 1;
	}
	
	public int getCounter()
	{
		return counter++;
	}
	
	// This function generates the outputs for 4 functions from a single output:
	// 1) sin(x)
	// 2) cos(x)
	// 3) x^2
	// 4) abs(x)
	public double[] generate4FunctionOutputs(double input)
	{
		double sin = Math.sin(input);
		double cos = Math.cos(input);
		double square = Math.pow(input, 2);
		double abs = Math.abs(input);
		
		double[] outputValues = {sin, cos, square, abs};
		 
		
		return outputValues;
	}
	
	public enum TrainingType
	{
		XOR,
		FourFunctions,
		NumberReading,
		TurtleTraining,
		TurtleTesting
	}
	
	public enum LossMode
	{
		SCALAR,  
		VECTOR
	}
}
