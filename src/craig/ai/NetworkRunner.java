package craig.ai;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import craig.ai.Helper.TrainingType;

public class NetworkRunner 
{
	// Number of epochs to run
	final static int numberOfEpochs = 2;
	
	// Print epoch every N iterations
	final static int epochIteration = 1;
	
	// Learning rate
	final static double learningRate = 0.02;
	
	// Number of total layers
	final static int layerCount = 4;
	
	// Number of nodes in each hidden layer
	final static int[] nodesCount = {784, 128, 64, 10};
	
	// In the scenario where we generate randomized training rows, use this
	final static int numberOfTrainingRows = 20;
	
	// Identify what type of training we're doing
	final static TrainingType trainingType = TrainingType.NumberReading;
	
	// Are we running learning in batches or stochastic?
	final static boolean isBatch = true;
	
	// How big is the batch?
	final static int batchSize = 64;

    @SuppressWarnings("unused")
	public static void main(String[] args) 
    {
        NeuralNetwork nn = new NeuralNetwork(numberOfEpochs, epochIteration, learningRate, layerCount, nodesCount);
        
        nn.initiateConnections();
        
        nn.setBatch(isBatch);
        nn.setBatchSize(batchSize);
        
        switch (trainingType)
        {
        	case TrainingType.NumberReading:
        	{        		
        		try 
        		{
					double[][] trainingInputs = MNISTLoader.loadImages("/Users/craigadelhardt/Documents/NeuralNetworkTraining/train-images-idx3-ubyte");
					int[] outputValues = MNISTLoader.loadLabels("/Users/craigadelhardt/Documents/NeuralNetworkTraining/train-labels-idx1-ubyte");
					
					double[][] trainingOutputs = new double[outputValues.length][10];
					
					for (int i = 0; i < outputValues.length; i++)
					{
						trainingOutputs[i][outputValues[i]] = 1;
					}
					
					nn.train(trainingInputs, trainingOutputs);
					
					System.out.println("Network learning is complete:");
					//nn.printNetwork();
					
					double[][] fullTestInputs = MNISTLoader.loadImages("/Users/craigadelhardt/Documents/NeuralNetworkTraining/t10k-images-idx3-ubyte");
					int[] fullTestOutputs = MNISTLoader.loadLabels("/Users/craigadelhardt/Documents/NeuralNetworkTraining/t10k-labels-idx1-ubyte");
					
					int numberOfTests = 25;
					
					double[][] actualTestOutputs = new double[numberOfTests][10];
					
					int[] testingIndexes = new int[numberOfTests];
					
					for (int i = 0; i < numberOfTests; i++)
					{
						testingIndexes[i] = new Random().nextInt(fullTestOutputs.length);
					}
					
					for (int i = 0; i < testingIndexes.length; i++)
					{
						
						actualTestOutputs[i][fullTestOutputs[testingIndexes[i]]] = 1;
					}
					
					// Need to reduce the testInputs array to just 10 images, since we're 
					// only pulling 10 outputs for our test after training.
					double[][] actualTestInputs = new double[numberOfTests][fullTestInputs[0].length];

					for (int i = 0; i < numberOfTests; i++)
					{
						for (int j = 0; j < actualTestInputs[0].length; j++)
						{
							actualTestInputs[i][j] = fullTestInputs[testingIndexes[i]][j];
						}
					}
					
					nn.printMNISTResults(actualTestInputs, actualTestOutputs);
					
					nn.exportWeightsAndBiases();
					
	                ArrayList<String> titles = new ArrayList<>();
	                titles.add("Output 0");
	                titles.add("Output 1");
	                titles.add("Output 2");
	                titles.add("Output 3");
	                titles.add("Output 4");
	                titles.add("Output 5");
	                titles.add("Output 6");
	                titles.add("Output 7");
	                titles.add("Output 8");
	                titles.add("Output 9");

	                Charting.plotMSE(nn.getMseList(), titles);

				} 
        		catch (IOException e) 
        		{
        			System.out.println("Some file I/O issue");
					e.printStackTrace();
				}
        		
        		break;
        	}
        	case TrainingType.FourFunctions:
        	{
                // Self created training nodes:	
                //nn.seedNetwork();
            	
            	// This train function is called when we will use the Helper class to generate
            	// our inputs/expectedOutputs training data.  Each epoch, we'll generate x 
            	// number of rows of training data.  And at this point, we're going to train the network
            	// on 4 concurrent functions: sin(x), cos(x), x^2, and abs(x).  

                System.out.println("Initialization:");
                //nn.printNetwork();

                nn.train(numberOfTrainingRows);
                
                System.out.println("Network learning is complete:");
                //nn.printNetwork();
                
                // Now generate some random inputs and expectedOutputs and let's see how they 
                // work in our trained network.
            	double[][] trainingInputs = new double[30][1];
            	double[][] trainingOutputs = new double[30][4];
            	
            	Helper helper = Helper.getInstance();
            	
            	for (int i = 0; i < 30; i++)
            	{
            		trainingInputs[i][0] = new Random().nextDouble() * 2 - 1;
            		
            		trainingOutputs[i] = helper.generate4FunctionOutputs(trainingInputs[i][0]);
            	}
                
                nn.printResults(trainingInputs, trainingOutputs);
                
                nn.exportWeightsAndBiases();

      /*        if (nn.loadExistingNetwork("/Users/craigadelhardt/Documents/NeuralNetworkExport/2025.08.10-22.45.38.json"))
                {
                	System.out.println("Network is loaded:");
                    nn.printNetwork();
                    nn.printResults(trainingInputs, trainingOutputs);
                }*/
                
                ArrayList<String> titles = new ArrayList<>();
                titles.add("sin(x)");
                titles.add("cos(x)");
                titles.add("x squared");
                titles.add("absolute value of x");
                Charting.plotMSE(nn.getMseList(), titles);
                
                break;
        	}
        	case TrainingType.XOR:
        	{
            	// Hardcoded training nodes:
                //nn.seedNetwork();

                // Test inputs for 4 logic gates 
                double[][] trainingInputs = 
                {
                    {0, 0},
                    {0, 1},
                    {1, 0},
                    {1, 1}
                };

                // Outputs will represent 4 different logic gates:
                // AND, OR, NAND, XOR
                // The goal is to train the network to take a 2 bit input and produce the expected
                // output for each of the logic gates
                double[][] trainingOutputs = 
                {
                	{0, 0, 1, 0}, // AND=0, OR=0, NAND=1, XOR=0
                	{0, 1, 1, 1}, // AND=0, OR=1, NAND=1, XOR=1
                	{0, 1, 1, 1}, // AND=0, OR=1, NAND=1, XOR=1
                	{1, 1, 0, 0}  // AND=1, OR=1, NAND=0, XOR=0
                };
               
                System.out.println("Initialization:");
                nn.printNetwork();

                nn.train(trainingInputs, trainingOutputs);
                
                System.out.println("Network learning is complete:");
                nn.printNetwork();
                
                nn.printResults(trainingInputs, trainingOutputs);
                
                nn.exportWeightsAndBiases();

        /*        if (nn.loadExistingNetwork("/Users/craigadelhardt/Documents/NeuralNetworkExport/2025.08.10-22.45.38.json"))
                {
                	System.out.println("Network is loaded:");
                    nn.printNetwork();
                    nn.printResults(trainingInputs, trainingOutputs);
                }*/
            	
                break;
        	}
        	default:
        		System.out.println("Didn't identify a proper training routine");
        }
    }
}


