package craig.ai.helpers;

import java.util.List;

import craig.ai.Neuron;

public class ArgMaxAnalyzer 
{
	private final int numberOfClasses;
	
	private int[] truePositiveCounter;
	private int[] falsePositiveCounter;
	private int[] falseNegativeCounter;
	
	
	
	public ArgMaxAnalyzer(int numberOfClasses)
	{
		this.numberOfClasses = numberOfClasses;
		
		truePositiveCounter = new int[numberOfClasses];
		falsePositiveCounter = new int[numberOfClasses];
		falseNegativeCounter = new int[numberOfClasses];
	}
	
	public void addPrediction(int predictedIndex, int trueIndex)
	{
		if (predictedIndex == trueIndex) 
		{
		    truePositiveCounter[predictedIndex]++;
		} 
		else 
		{
		    falsePositiveCounter[predictedIndex]++;
		    falseNegativeCounter[trueIndex]++;
		}
	}
	
	public static int getArgMax(List<Neuron> outputs)
	{
		double[] outputsArray = new double[outputs.size()];
		
		for (int i = 0; i < outputs.size(); i++)
		{
			outputsArray[i] = outputs.get(i).getOutputValue(); 
		}
		
		return getArgMax(outputsArray);
	}
	
	public static int getArgMax(double[] outputs)
	{
		int index = 0;
		double maxValue = Double.NEGATIVE_INFINITY;;

		for (int i = 0; i < outputs.length; i++)
		{
			if (outputs[i] > maxValue)
			{
				index = i;
				maxValue = outputs[i];
			}
		}
		
		return index;
	}
	
	public double getAccuracy() 
	{
	    int correct = 0, total = 0;
	    
	    for (int i = 0; i < numberOfClasses; i++) 
	    {
	        correct += truePositiveCounter[i];
	        total += truePositiveCounter[i] + falsePositiveCounter[i] + falseNegativeCounter[i];
	    }
	    
	    return (total == 0) ? 0.0 : (double) correct / total;
	}

	
	public void printF1Stats()
	{
		// First print the per class stats:
		for (int i = 0; i < numberOfClasses; i++)
		{
	        // Precision = TP / (TP+FP), Recall = TP / (TP+FN)
	        double precision = (truePositiveCounter[i] + falsePositiveCounter[i]) == 0 ? 0 : (double) truePositiveCounter[i] / (truePositiveCounter[i] + falsePositiveCounter[i]);
	        double recall = (truePositiveCounter[i] + falseNegativeCounter[i]) == 0 ? 0 : (double) truePositiveCounter[i] / (truePositiveCounter[i] + falseNegativeCounter[i]);
	
	        // F1 = harmonic mean
	        double f1 = ((precision + recall) == 0) ? 0 : 2 * precision * recall / (precision + recall);
	        
	        System.out.printf("Class %d: F1 Score: %.3f (Precision: %.3f, Recall: %.3f)%n", i, f1, precision, recall);
		}
		
		// Then print the micro stats for all classes:
		int totalTP = 0, totalFP = 0, totalFN = 0;
		
		for (int i = 0; i < numberOfClasses; i++) 
		{			
		    totalTP += truePositiveCounter[i];
		    totalFP += falsePositiveCounter[i];
		    totalFN += falseNegativeCounter[i];
		}
		
		double precision = (totalTP + totalFP) == 0 ? 0 : (double) totalTP / (totalTP + totalFP);
		double recall = (totalTP + totalFN) == 0 ? 0 : (double) totalTP / (totalTP + totalFN);
		double f1 = ((precision + recall) == 0) ? 0 : 2 * precision * recall / (precision + recall);

		System.out.printf("Overall (micro) F1 Score: %.3f (Precision: %.3f, Recall: %.3f)%n", f1, precision, recall);
		System.out.printf("Accuracy for this epoch: %f%n", getAccuracy());		
		System.out.println("-------------------");
	}
}
