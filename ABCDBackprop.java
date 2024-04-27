import java.io.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;
/*
 * @author Isha Kotalwar
 * 
 * @version 04/15/2024
 * 
 * This A-B-C-D perceptron is based on the previous A-B-C backpropagation project and the document
 * "4-Three Layer Network" It can train a network based on set weights provided by the
 * user or randomized weights. The parameters are given to the network through an external configuration file. 
 * The network trains through backpropagation.
 */

public class ABCDBackprop
{
   public int numInputs;
   public int numHiddens1;
   public int numHiddens2;
   public int numLayers;
   public int numOutputs;
   public int maxNumIterations;
   public int numTestCases;
   public int iterations;
   public static final int INPUT_LAYER = 0;
   public static final int HIDDEN1_LAYER = 1;
   public static final int HIDDEN2_LAYER = 2;
   public static final int OUTPUT_LAYER = 3;

   public double lambda;
   public double randomMin;
   public double randomMax;
   public double avgError;
   public double errorThreshold;
   
   public String trainingOrRunning;
   public String randomOrSetWeights;
   public String reasonForStopping;
   public String weightsFile;
   public String testcasesFile;
   public String parametersFile;

   public double [][] activations;
   public double[] bigOmega;
   public double[] bigPsiJ;
   public double[][] theta;
   public double[] smallPsiI;
   public double[][][] weights;

   public double[][] truthTable;
   public double[][] expectedTruthTable;
   public double[][] actualTruthTable;

   public long totalRunTime;

   /*
    * The trainOrRun() method determines whether the network is going to train or run.
    */
   public void trainOrRun()
   {
      if (trainingOrRunning.equals("training"))
      {
         trainNetwork();
      }

      else
      {
         runNetwork();
      }
   } // public void trainOrRun()

   /*
    * The init() method allocates memory and initializes all of the arrays that the network will
    * use.
    */
   public void init()
   {
      int maxNodes;

      maxNodes = Math.max(numInputs, numHiddens1);
      maxNodes = Math.max(maxNodes, numHiddens2);
      maxNodes = Math.max(maxNodes, numOutputs);

      if (trainingOrRunning.equals("training"))
      {
         bigOmega = new double[numOutputs];
         bigPsiJ = new double[numHiddens2];
         theta = new double[numLayers - 1][maxNodes];
         smallPsiI = new double[numOutputs];
      } // if(trainingOrRunning.equals("training"))

      activations = new double[numLayers][maxNodes];
      
      weights = new double[numLayers][maxNodes][maxNodes];

      truthTable = new double[numTestCases][numInputs];
      expectedTruthTable = new double[numTestCases][numOutputs];
      actualTruthTable = new double[numTestCases][numOutputs];
   } // public void init()

   /*
    * The echoConfigParameters() method prints out all of the parameters and the network
    * configuration.
    */
   public void echoConfigParameters()
   {
      System.out
            .println("Network Configuration: " + numInputs + "-" + numHiddens1 + "-" + numHiddens2 + "-" + numOutputs);
      System.out.println("Training or running? " + trainingOrRunning);
      System.out.println("Randomize or set weights? " + randomOrSetWeights);

      if (trainingOrRunning.equals("training"))
      {
         System.out.println("Random number range: " + "(" + randomMin + ", " + randomMax + ")");
         System.out.println("Maximum iterations: " + maxNumIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
      } // if(trainingOrRunning.equals("training"))
   } // public void echoConfigParameters()

   /*
   * 
   */
   public int scanInteger(String inputString)
   {
      int number = 0;
      String[] parts = inputString.split("#");
      String numberPart = parts[0].trim(); // Get the first part and trim whitespace

      Scanner numberScanner = new Scanner(numberPart);
      if (numberScanner.hasNextInt())
      {
         number = numberScanner.nextInt();
      }
      else
      {
         System.out.println("No valid integer found!" + inputString);
      }

      numberScanner.close();
      return number;
   } // public double scanInteger(String inputString)

   /*
   * 
   */
   public double scanDouble(String inputString)
   {
      double number = 0.0;
      String[] parts = inputString.split("#");
      String numberPart = parts[0].trim(); // Get the first part and trim whitespace

      Scanner numberScanner = new Scanner(numberPart);
      if (numberScanner.hasNextDouble())
      {
         number = numberScanner.nextDouble();
      }
      else
      {
         System.out.println("No valid double found!");
      }

      numberScanner.close();
      return number;
   } // public double scanDouble(String inputString)

   /*
   * 
   */
   public String scanString(String inputString)
   {
      String[] parts = inputString.split("#");
      String stringPart = parts[0].trim(); // Get the first part and trim whitespace
      return stringPart;
   } // public String scanString(String inputString)

   /*
    * The readFile() method reads in everything that is needed from a specified text file.
    */
   public void readParametersFile(String inputFileName)
   {
      parametersFile = inputFileName;
      File pFile = new File(parametersFile);
      Scanner sc = null;
      try
      {
         sc = new Scanner(pFile);
      } 
      catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      numInputs = scanInteger(sc.nextLine());
      numHiddens1 = scanInteger(sc.nextLine());
      numHiddens2 = scanInteger(sc.nextLine());
      numOutputs = scanInteger(sc.nextLine());
      numLayers = scanInteger(sc.nextLine());
      lambda = scanDouble(sc.nextLine());
      maxNumIterations = scanInteger(sc.nextLine());
      trainingOrRunning = scanString(sc.nextLine());
      randomOrSetWeights = scanString(sc.nextLine());
      randomMin = scanDouble(sc.nextLine());
      randomMax = scanDouble(sc.nextLine());
      errorThreshold = scanDouble(sc.nextLine());
      numTestCases = scanInteger(sc.nextLine());
      init();
      testcasesFile = scanString(sc.nextLine());
      weightsFile = scanString(sc.nextLine());
      sc.close();
   } // public void readFile(File pFile)


   /*
   * The saveWeightsFile() method saves weights to a file.
   */
  public void saveWeightsFile()
  {
      try 
      {
         PrintWriter pw = new PrintWriter(new FileOutputStream(new File(weightsFile), false));

         for (int k = 0; k < numLayers; k++)
         {
            for (int j = 0; j < numInputs && k == 0; j++)
            {
               for (int i = 0; i < numHiddens1; i++)
               {
                  pw.println(weights[k][j][i]);
               }
            }

            for (int j = 0; j < numHiddens1 && k == 1; j++)
            {
               for (int i = 0; i < numHiddens2; i++)
               {
                  pw.println(weights[k][j][i]);
               }
            }

            for (int j = 0; j < numHiddens2 && k == 2; j++)
            {
               for (int i = 0; i < numOutputs; i++)
               {
                  pw.println(weights[k][j][i]);
               }
            }
         } // for (int k = 0; k < numLayers; k++)
         pw.close();
      } // try
      catch (Exception e) 
      {
         System.out.println("Exception" + e.toString());
      }
  } // public void saveWeightsFile()

   /*
    * The populate() method fills all of the weight arrays with either random or given values by the
    * user. It also fills in the truth table test cases.
    */
   public void populate()
   {
      Scanner sc = null;
      File pFile = new File(parametersFile);
      try
      {
         sc = new Scanner(pFile);
      }
      catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      for (int i = 0; i < 15; i++)
      {
         sc.nextLine();
      }
      for (int i = 0; i < numTestCases; i++)
      {
         String expectedString = sc.nextLine();
         String[] parts = expectedString.split("\\s+");
         for (int j = 0; j < parts.length; j++)
         {
            expectedTruthTable[i][j] = scanInteger(parts[j]);
         }
      }
      sc.close();

      File tFile = new File(testcasesFile);
      try
      {
         sc = new Scanner(tFile);
      }
      catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      for (int i = 0; i < numTestCases; i++)
      {
         String expectedString = sc.nextLine();
         String[] parts = expectedString.split("\\s+");
         for (int j = 0; j < parts.length; j++)
         {
            truthTable[i][j] = scanInteger(parts[j]);
         }
      }

      sc.close();

      if (randomOrSetWeights.equals("set"))
      {
         File wFile = new File(weightsFile);
         try
         {
            sc = new Scanner(wFile);
         } 
         catch (FileNotFoundException e)
         {
            e.printStackTrace();
         }

         for (int k = 0; k < numLayers; k++)
         {
            for (int j = 0; j < numInputs && k == 0; j++)
            {
               for (int i = 0; i < numHiddens1; i++)
               {
                  weights[k][j][i] = scanDouble(sc.nextLine());
               }
            }

            for (int j = 0; j < numHiddens1 && k == 1; j++)
            {
               for (int i = 0; i < numHiddens2; i++)
               {
                  weights[k][j][i] = scanDouble(sc.nextLine());
               }
            }

            for (int j = 0; j < numHiddens2 && k == 2; j++)
            {
               for (int i = 0; i < numOutputs; i++)
               {
                  weights[k][j][i] = scanDouble(sc.nextLine());
               }
            }
         } // for (int k = 0; k < numLayers; k++)
      } // if (randomOrSetWeights.equals("set"))

      if (randomOrSetWeights.equals("random")) // randomize weights
      {
         for (int k = 0; k < numLayers; k++)
         {
            for (int j = 0; j < numInputs && k == 0; j++)
            {
               for (int i = 0; i < numHiddens1; i++)
               {
                  weights[k][j][i] = randomize(randomMin, randomMax);
               }
            }

            for (int j = 0; j < numHiddens1 && k == 1; j++)
            {
               for (int i = 0; i < numHiddens2; i++)
               {
                  weights[k][j][i] = randomize(randomMin, randomMax);
               }
            }

            for (int j = 0; j < numHiddens2 && k == 2; j++)
            {
               for (int i = 0; i < numOutputs; i++)
               {
                  weights[k][j][i] = randomize(randomMin, randomMax);
               }
            }
         } // for (int k = 0; k < numLayers; k++)
      } // if (randomOrSetWeights.equals("random"))
   } // public void populate()

   /*
    * The method setInputs() initialized the input values for a given index.
    */
   public void setInputs(int inputIndex)
   {
      for (int k = 0; k < numInputs; k++)
      {
         activations[INPUT_LAYER][k] = truthTable[inputIndex][k];
      }
   }

   /*
    * The trainNetwork() method runs through the entire training of the network. It sets the
    * inputs, runs the forward evaluation training, backpropagation, and calculates weights. Then, 
    * it calculates toatl error and average error by comparing the network's result to the desired result.
    * Once the max number of iterations has been reached, or the average error is less than the error
    * threshold, training is finished. trainNetwork() is called by the method trainOrRun().
    */
   public void trainNetwork()
   {
      long startTime = System.currentTimeMillis();
      totalRunTime = 0;
      double totalError = 0.0;
      boolean finished = false;

      while (!finished)
      {
         avgError = 0.0;
         totalError = 0.0;

         for (int index = 0; index < numTestCases; index++) // iterates over every test case
         {
            setInputs(index);
            forwardEvaluationTraining(index);
            backpropagation();
            runNetworkCalculations();

            for (int i = 0; i < numOutputs; i++)
            {
               double omega = (expectedTruthTable[index][i] - activations[numLayers - 1][i]);
               totalError += 0.5 * omega * omega;
               actualTruthTable[index][i] = activations[numLayers - 1][i];
            }
         } // for (int index = 0; index < numTestCases; index++)
         avgError = totalError / ((double) numTestCases);
         if (avgError <= errorThreshold)
         {
            finished = true;
         }

         iterations++;
         if (iterations >= maxNumIterations)
         {
            finished = true;
         }
      } // while (!finished)
      totalRunTime = System.currentTimeMillis() - startTime;
   } // public void trainNetwork()

   /*
    * The runNetworkCalculations() method calculates the hidden neurons and the output neurons.
    */
   public void runNetworkCalculations()
   {
      double tempTheta;

      for (int k = 0; k < numHiddens1; k++)
      {
         tempTheta = 0.0;
         int n = INPUT_LAYER;

         for (int m = 0; m < numInputs; m++)
         {
            tempTheta += weights[n][m][k] * activations[n][m];
         }
         activations[n + 1][k] = sigmoidFunction(tempTheta);
      } // for (int k = 0; k < numInputs; k++)

      for (int j = 0; j < numHiddens2; j++)
      {
         tempTheta = 0.0;
         int n = HIDDEN1_LAYER; 

         for (int k = 0; k < numHiddens1; k++)
         {
            tempTheta += weights[n][k][j] * activations[n][k];
         }
         activations[n + 1][j] = sigmoidFunction(tempTheta);
      } // for (int j = 0; j < numHiddens; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         tempTheta = 0.0;
         int n = HIDDEN2_LAYER; 

         for (int j = 0; j < numHiddens2; j++)
         {
            tempTheta += weights[n][j][i] * activations[n][j];
         }
         activations[n + 1][i] = sigmoidFunction(tempTheta);
      } // for (int i = 0; i < numOutputs; i++)
   } // public void runNetworkCalculations()

   /*
    * The forwardEvaluationTraining() method calculates the hidden neurons.
    */
   public void forwardEvaluationTraining(int index)
   {
      double tempTheta;

      for (int k = 0; k < numHiddens1; k++)
      {
         tempTheta = 0.0;
         int n = INPUT_LAYER;

         for (int m = 0; m < numInputs; m++)
         {
            tempTheta += weights[n][m][k] * activations[n][m];
         }
         activations[n + 1][k] = sigmoidFunction(tempTheta);
         theta[n][k] = tempTheta;
      } // for (int k = 0; k < numInputs; k++)

      for (int j = 0; j < numHiddens2; j++)
      {
         tempTheta = 0.0;
         int n = HIDDEN1_LAYER; 

         for (int k = 0; k < numHiddens1; k++)
         {
            tempTheta += weights[n][k][j] * activations[n][k];
         }
         activations[n + 1][j] = sigmoidFunction(tempTheta);
         theta[n][j] = tempTheta;
      } // for (int j = 0; j < numHiddens; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         tempTheta = 0.0;
         int n = HIDDEN2_LAYER; 

         for (int j = 0; j < numHiddens2; j++)
         {
            tempTheta += weights[n][j][i] * activations[n][j];
         }
         activations[n + 1][i] = sigmoidFunction(tempTheta);
         bigOmega[i] = expectedTruthTable[index][i] - activations[n + 1][i];
         smallPsiI[i] = bigOmega[i] * sigmoidDerivative(tempTheta);
      } // for (int i = 0; i < numOutputs; i++)
   } // public void forwardEvaluationTraining()

   /*
    * The backpropagation() method calculates the adjustment of weights using the backpropagation
    * algorithm.
    */
   public void backpropagation()
   {
      double omegaJ, omegaK;

      for (int j = 0; j < numHiddens2; j++)
      {
         omegaJ = 0.0;
         int n = HIDDEN2_LAYER;

         for (int i = 0; i < numOutputs; i++)
         {
            omegaJ += smallPsiI[i] * weights[n][j][i];
            weights[n][j][i] += lambda * activations[n][j] * smallPsiI[i];
         } // for (int i = 0; i < numOutputs; i++)

         bigPsiJ[j] = omegaJ * sigmoidDerivative(theta[n - 1][j]);
      } // for (int j = 0; j < numHiddens2; j++)

      for (int k = 0; k < numHiddens1; k++)
      {
         double bigPsiK;
         omegaK = 0.0;
         int n = HIDDEN1_LAYER;

         for (int j = 0; j < numHiddens2; j++)
         {
            omegaK += bigPsiJ[j] * weights[n][k][j];
            weights[n][k][j] += lambda * activations[n][k] * bigPsiJ[j];
         } // for (int j = 0; j < numHiddens2; j++)

         bigPsiK = omegaK * sigmoidDerivative(theta[n - 1][k]);

         for (int m = 0; m < numInputs; m++)
         {
            n = INPUT_LAYER;
            weights[n][m][k] += lambda * activations[n][m] * bigPsiK; 
         }
      } // for (int k = 0; k < numHiddens1; k++)
   } // public void backpropagation(int index)

   /*
    * The runNetwork() method runs on all of the test cases. It is called when the status 
    * of trainingOrRunning is "running."
    */
   public void runNetwork()
   {
      for (int index = 0; index < numTestCases; index++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            activations[index][k] = truthTable[index][k];
         }
         runNetworkCalculations();
         for (int i = 0; i < numOutputs; i++)
         {
            actualTruthTable[index][i] = activations[index][i];
         }
      } // for (int index = 0; index < numTestCases; index++)
   } // public void runNetwork();

   /*
    * The reportResults() method prints out the truth table and the output of the network. If the
    * network was training, it also reports training information, like reason for stopping, number
    * of iterations, and error reached.
    */
   public void reportResults()
   {
      System.out.println("*************************************************");
      System.out.println("Truth Table");

      for (int row = 0; row < numTestCases; row++)
      {
         String output = "";
         for (int k = 0; k < numInputs; k++)
         {
            output += truthTable[row][k] + "\t";
         }
         output += "|\t";
         for (int i = 0; i < numOutputs; i++)
         {
            output += expectedTruthTable[row][i] + "\t";
         }
         output += "|\t";
         for (int i = 0; i < numOutputs; i++)
         {
            output += actualTruthTable[row][i] + "\t  ";
         }
         System.out.println(output);
      } // for (row = 0; row < numTestCases; row++)

      if (trainingOrRunning.equals("training"))
      {
         System.out.println("*************************************************");
         System.out.println("Training Info:");
         if (avgError <= errorThreshold)
         {
            reasonForStopping = "the average error is below the error threshold.";
         }
         if (iterations >= maxNumIterations)
         {
            reasonForStopping = "the maximum amount of iterations has been reached.";
         }
         System.out.println("Training ended because " + reasonForStopping);
         System.out.println("Number of iterations reached: " + iterations);
         System.out.println("Error reached: " + avgError);
         System.out.println("Total execution time: " + totalRunTime + " ms");
         System.out.println();
      } // if (trainingOrRunning.equals("training"))
      saveWeightsFile();
   } // public void reportResults()

   /*
    * The sigmoidFunction() method returns the value of a given x plugged into the sigmoid
    * activation function.
    */
   public double sigmoidFunction(double x)
   {
      return 1.0 / (1 + Math.exp(-x));
   } // public double sigmoidFunction(double x)

   /*
    * The sigmoidDerivative() method returns the value of a given x plugged into the derivative of
    * the sigmoid activation function.
    */
   public double sigmoidDerivative(double x)
   {
      double s = sigmoidFunction(x);
      return s * (1 - s);
   } // public double sigmoidFunction(double x)

   /*
    * The randomize() method returns a random value between the given min and max values.
    */
   public double randomize(double min, double max)
   {
      return min + (max - min) * Math.random();
   } // public double randomize(double min, double max)

   /*
    * The main() method runs the network's 6 main methods.
    */
   public static void main(String[] args)
   {
      String inputFileString;
      if (args.length == 1)
      {
         inputFileString = args[0];
      }
      else
      {
         inputFileString = "default.txt";
      }
      ABCDBackprop perceptron = new ABCDBackprop();
      perceptron.readParametersFile(inputFileString);
      perceptron.echoConfigParameters();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults();
   } // public static void main(String[] args)
} // public class ABCDBackprop