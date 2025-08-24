package craig.ai.external;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

public class ImageLoader {

    /**
     * Loads images from a base folder. Each subfolder is a label.
     * Returns an array of inputs (flattened 32x32 grayscale) and an array of labels.
     */
	// This image loader grabs every sub-folder in a folder.  The name of the sub-folder needs to match the 
	// label of the picture in the sub-folder.  For this first iteration, we'll have turtles in one sub-folder and
	// everything else in another sub-folder.  
    public static void loadImages(String baseFolderPath, List<double[]> inputs, List<int[]> labels, List<String> fileNames) throws IOException 
    {
        // Create a File out of the folder path passed in
    	File baseFolder = new File(baseFolderPath);
        
        // If for whatever reason the path passed in isn't a folder...
    	if (!baseFolder.isDirectory()) 
        {
            throw new IOException("Base folder path is not a directory: " + baseFolderPath);
        }

    	// Parse through the sub-folders and generate an array of Files of those sub-folders.
        File[] subFolders = baseFolder.listFiles(File::isDirectory);
        
        // If there aren't any sub-folders, then there's nothing to do but complain.
        if (subFolders == null)
        {
        	System.out.println("There aren't any sub-folders in your directory.");
        	return;
        }

        // Iterate over the subFolders
        for (File subFolder : subFolders) 
        {
        	// Does this subFolder contain turtles?
            int[] label = new int[2];
            
            label[0] = subFolder.getName().equalsIgnoreCase("turtle") ? 1 : 0;
            label[1] = subFolder.getName().equalsIgnoreCase("turtle") ? 0 : 1;

            // Get all the files within the subFolder.
            File[] files = subFolder.listFiles();
            
            // If there aren't any files, move along.
            if (files == null)
            {
            	continue;
            }

            // Iterate over the files in the subFolder
            for (File file : files) 
            {
            	// Is this file actually a file?  What else could it be?  Just in case...
                if (!file.isFile())
                {
                	continue;
                }

                // Let's turn that file into a BufferedImage
                BufferedImage img = ImageIO.read(file);
                
                // If the file wasn't actually an image file that was readable, then img 
                // will be null.  Move along!
                if (img == null)
                {
                	continue;  
                }

                // Standardize the background
                img = standardizeBackground(img);
                
                // Call the internal function here to resize the BufferedImage, to grayscale it,
                // and to turn it into an array of doubles
                double[] input = imageToInputArray(img);
                
                // Save the image array and labels
                inputs.add(input);
                labels.add(label);
                fileNames.add(file.getName());
            }
        }
    }

    // This function takes a BufferedImage, resizes it to 32x32, turns it gray, and then
    // turns it into an array of double for 1024 numeric representations of the pixels
    public static double[] imageToInputArray(BufferedImage img) 
    {
    	int width = 64;
    	int height = 64;
    	
        // Set up a new BufferedImage that will be bound by a 32x32 pixel box and is only gray.
    	BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        
    	// This is all it takes to change the size and color of the BufferedImage.  Neat.
        resized.getGraphics().drawImage(img, 0, 0, width, height, null);

        // Set up our input array
        double[] input = new double[width * height];
        
        // Iterate over the y values
        for (int y = 0; y < height; y++) 
        {
        	// Now iterate over the x values
            for (int x = 0; x < width; x++) 
            {
            	// Get the RGB of the pixel we're at
                int rgb = resized.getRGB(x, y);
                
                // Turn that pixel grayscale
                int gray = rgb & 0xFF; 
                
                // Save the grayscale value normalized to between 0 and 1
                // "1 -" in front to normalize the background to 1 (white) and the foreground to 0 (black) 
                input[y * width + x] = 1 - gray / 255.0;
            }
        }
        
        return input;
    }
    
    // This standardizes the background of the image - basically if it's transparent, make it white.
    // This will help with image processing later since we don't want the image routine to see the 
    // background as anything to focus on.
    public static BufferedImage standardizeBackground(BufferedImage img) 
    {
        // Create a new image of the same size with white background
        BufferedImage standardized = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
    
        Graphics2D g = standardized.createGraphics();
        
        g.setColor(Color.WHITE);  // consistent background
        
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        
        g.drawImage(img, 0, 0, null);
        
        g.dispose();
        
        return standardized;
    }


    // This function can be used as a verifier.  It will take the 28x28 grayscale 
    // values that were pulled from the images file and display them ASCII style
    public static void printImageASCII(double[] image) 
    {
        int width = 64;
        int height = 64;
        
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

    // Quick tester to make sure an image can be read and loaded
    public static void main(String[] args) throws IOException 
    {
        List<double[]> inputs = new ArrayList<>();
        List<int[]> labels = new ArrayList<>();
        List<String> fileNames = new ArrayList<>();
    
        String folderPath = "/Users/craigadelhardt/Documents/neural network learning images"; 

        loadImages(folderPath, inputs, labels, fileNames);

        System.out.println("Loaded " + inputs.size() + " images.");
        
        if (!inputs.isEmpty()) 
        {
            System.out.println("First image label: " + labels.get(0)[0]);
            printImageASCII(inputs.get(0));
            
            System.out.println();
            
            System.out.println("Second image label: " + labels.get(1)[0]);
            printImageASCII(inputs.get(1));
            
            System.out.println();
            
            System.out.println("Third image label: " + labels.get(300)[0]);
            printImageASCII(inputs.get(300));
        }
    }
}
