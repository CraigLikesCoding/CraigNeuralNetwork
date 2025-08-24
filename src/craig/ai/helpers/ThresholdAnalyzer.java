package craig.ai.helpers;

import java.util.List;

public class ThresholdAnalyzer 
{
	private double expectedOutput; 
	private double actualOutput;
	
	public ThresholdAnalyzer(double actual, double expected)
	{
		expectedOutput = expected;
		actualOutput = actual;
	}
	
	public static double chooseBestThreshold(List<ThresholdAnalyzer> thresholds) 
	{
	    double bestThreshold = 0.0;
	    double bestF1 = -1.0; 
	    double bestPrecision = 0.0;
	    double bestRecall = 0.0;

	    for (int i = 0; i <= 100; i++) 
	    {
	        double threshold = i / 100.0;
	        
	        int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

	        for (ThresholdAnalyzer currentThreshold : thresholds) 
	        {
	            boolean prediction = (currentThreshold.getActualOutput() >= threshold) ? true : false;
	            
	            boolean actual = currentThreshold.getExpectedOutput() >= 0.5; // treat >=0.5 as true
	            
	            if (prediction && actual)
	            {
	            	truePositive++;
	            }
	            else if (!prediction && !actual)
	            {
	            	trueNegative++;
	            }
	            else if (prediction && !actual) 
	            { 
	            	falsePositive++;
	            }
	            else 
	            {
	            	falseNegative++;
	            }
	        }

	        // Precision = TP / (TP+FP), Recall = TP / (TP+FN)
	        double precision = (truePositive + falsePositive) == 0 ? 0 : (double) truePositive / (truePositive + falsePositive);
	        double recall = (truePositive + falseNegative) == 0 ? 0 : (double) truePositive / (truePositive + falseNegative);

	        // F1 = harmonic mean
	        double f1 = ((precision + recall) == 0) ? 0 : 2 * precision * recall / (precision + recall);
	        
	        if (f1 > bestF1) 
	        {
/*		        System.out.println("Hey, f1 > bestF1...let's see some data");
		        System.out.printf("f1 = %f%n", f1);
		        System.out.printf("bestF1 = %f%n", bestF1);
	        	System.out.printf("truePositive = %d%n", truePositive);
		        System.out.printf("trueNegative = %d%n", trueNegative);
		        System.out.printf("falsePositive = %d%n", falsePositive);
		        System.out.printf("falseNegative = %d%n", falseNegative);
		        System.out.printf("precision = %f%n", precision);
		        System.out.printf("recall = %f%n", recall);
		        System.out.println();*/
		        
	            bestF1 = f1;
	            bestThreshold = threshold;
	            bestPrecision = precision;
	            bestRecall = recall;
	        }
	    }

	    System.out.printf("Best threshold=%.2f%n", bestThreshold);
	    System.out.printf("F1 Score: %.3f (Precision: %.3f, Recall: %.3f)%n", bestF1, bestPrecision, bestRecall);
	    return bestThreshold;
	}

	public double getExpectedOutput() {
		return expectedOutput;
	}

	public void setExpectedOutput(double expectedOutput) {
		this.expectedOutput = expectedOutput;
	}

	public double getActualOutput() {
		return actualOutput;
	}

	public void setActualOutput(int actualOutput) {
		this.actualOutput = actualOutput;
	}
}
