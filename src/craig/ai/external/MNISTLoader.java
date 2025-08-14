package craig.ai.external;

import java.io.*;

public class MNISTLoader 
{
	public static double[][] loadImages(String filePath) throws IOException 
	{
        try (DataInputStream imageFile = new DataInputStream(new FileInputStream(filePath))) 
        {
        	// Check the first byte to read the magic number of in the file, as this number
        	// tells us if the MNIST file contains the images, the labels, or is not a valid
        	// file.  In this function, we're looking for the images file (2051) and nothing else.
            int magic = imageFile.readInt();
            
            if (magic != 2051) 
            {
            	throw new IOException("Invalid MNIST image file magic number");
            }

            // Pull out the descriptive data from the image file
            // These 4 bytes indicate the count of images:
            int numImages = imageFile.readInt();
            
            // These 4 bytes indicate the number of rows of pixels per image 
            // (28 in the the MNIST dataset)
            int numRows = imageFile.readInt();
            
            // These 4 bytes indicate the number of columns of pixels per image
            // (28 in the the MNIST dataset)
            int numCols = imageFile.readInt();

            // This is going to go straight into the input neurons for learning
            // We'll need numRows * numCols input neurons
            double[][] images = new double[numImages][numRows * numCols];

            for (int i = 0; i < numImages; i++) 
            {
                for (int j = 0; j < numRows * numCols; j++) 
                {
                    int pixel = imageFile.readUnsignedByte();
                    images[i][j] = pixel / 255.0;
                }
            }
            
            return images;
        }
    }
	
    public static int[] loadLabels(String filePath) throws IOException 
    {
        try (DataInputStream labelFile = new DataInputStream(new FileInputStream(filePath))) 
        {
        	// Check the first byte to read the magic number of in the file, as this number
        	// tells us if the MNIST file contains the images, the labels, or is not a valid
        	// file.  In this function, we're looking for the label file (2049) and nothing else.
            int magic = labelFile.readInt();
            
            if (magic != 2049) 
            {
            	throw new IOException("Invalid MNIST label file magic number");
            }

            // First we grab the 4 bytes that tell us how many labels are in this file.
            // It better match the number of images in the other file!  And I assume they 
            // are going to be lined up.
            int numLabels = labelFile.readInt();
            
            int[] labels = new int[numLabels];

            for (int i = 0; i < numLabels; i++) 
            {
                labels[i] = labelFile.readUnsignedByte();
            }
            
            return labels;
        }
    }
    
    // This function can be used as a verifier.  It will take the 28x28 grayscale 
    // values that were pulled from the images file and display them ASCII style
    public static void printImageASCII(double[] image) 
    {
        int width = 28;
        int height = 28;

        for (int row = 0; row < height; row++) 
        {
            StringBuilder line = new StringBuilder();
            
            for (int col = 0; col < width; col++) 
            {
                double val = image[row * width + col];
            
                if (val > 0.75) 
                {
                	line.append('#');
                }
                else if (val > 0.5)
                {
                	line.append('O');
                }
                else if (val > 0.25)
                {	
                	line.append('o');
                }
                else if (val > 0.1) 
                {
                	line.append('.');
                }
                else 
                {
                	line.append(' ');
                }
            }
            System.out.println(line);
        }
    }
    
    // Quick tester for file validity.
    public static void main(String[] args) throws IOException 
    {
        try (DataInputStream dis = new DataInputStream(new FileInputStream("/Users/craigadelhardt/Documents/NeuralNetworkTraining/train-labels-idx1-ubyte"))) 
        {
            int magic = dis.readInt();
            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            System.out.printf("Magic: %d, Images: %d, Rows: %d, Cols: %d%n",
                              magic, numImages, numRows, numCols);
        }
    }
    
}
