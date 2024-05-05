import java.io.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;
/*
 * @author Isha Kotalwar
 * 
 * @version 04/28/2024
 * 
 * This A-B-C-D perceptron is based on the previous A-B-C backpropagation project and the document
 * "4-Three Layer Network." There are two hidden layers. It can train a network based on set weights
 * provided by the user or randomized weights. The parameters are given to the network through an
 * external configuration file. The network trains through backpropagation.
 * 
 * Table of Contents: - trainOrRun() determines whether the network is training or simply running. -
 * init() allocates space for aand initializes all of the arrays the network uses. -
 * echoConfigParameters() prints out the network configuration and other important parameters. -
 * scanInteger() takes in a String from an external file and returns an integer if present. -
 * scanDouble() takes in a String from an external file and returns a double if present. -
 * scanString() takes in a String from an external file, trims the whitespace and returns the
 * desired part of the String. - readParametersFile() goes through the external configuration file
 * and stores everything in variables. - saveWeightsFile() saves the final weights from training
 * into an external file. - populate() fills up the truth table and the weights array with either
 * random or set weights. - setInputs() sets the input activations. - trainNetwork() sets the
 * inputs, runs forward evaluation training, backpropagation, calculates weights, and calculates
 * error. If training hits the max number of iterations or the average error is less than the error
 * threshold, training ends. - runNetworkCalculations() calculates hidden and output neurons. -
 * forwardEvaluationTraining() calculates hidden neurons in forward evaluation. - backpropagation()
 * calculates the change of weights using the backpropagation algorithm. - runNetwork() runs over
 * all test cases. - reportResults() prints out training information if training, like error, and
 * number of iterations reached. It also prints out the truth table, the network's results, and time
 * for both running and training. - sigmoidFunction() calculates the value of a given input plugged
 * into the sigmoid function. - sigmoidDerivative() calculates the value of a given input plugged
 * into the derivative of the sigmoid function. - randomize() returns a random double between a
 * given min and max. - main() calls readParametersFile() if there is a file passed in at the
 * command line. Otherwise, it uses a default configuration file. It also calls
 * echoConfigParameters(), init(), populate(), trainOrRun(), and reportResults().
 */

public class NLayer
{
   public int numLayers;
   public int numInputs;
   public int[] numNodesInLayer;
   public int numOutputs;
   public int maxNumIterations;
   public int numTestCases;
   public int iterations;
   public int maxNodes;
   public int offsetInFile;
   public static final int INPUT_LAYER = 0;

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
   public String networkConfiguration;
   public static final String DEFAULT_FILE_STRING = "default.txt";

   public double[][] activations;
   public double[][] theta;
   public double[][] psi;
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
         runNetwork();
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
      if (trainingOrRunning.equals("training"))
      {
         psi = new double[numLayers][maxNodes];
         theta = new double[numLayers][maxNodes];
      } // if (trainingOrRunning.equals("training"))

      activations = new double[numLayers][maxNodes];
      numNodesInLayer = new int[numLayers];
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
      networkConfiguration =
            numInputs + "-" + numNodesInLayer[0] + "-" + numNodesInLayer[1] + "-" + numOutputs;
      System.out.println("Network Configuration: " + networkConfiguration);
      System.out.println("Training or running? " + trainingOrRunning);
      System.out.println("Randomize or set weights? " + randomOrSetWeights);

      if (trainingOrRunning.equals("training"))
      {
         System.out.println("Random number range: " + "(" + randomMin + ", " + randomMax + ")");
         System.out.println("Maximum iterations: " + maxNumIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
      } // if (trainingOrRunning.equals("training"))
   } // public void echoConfigParameters()

   /*
    * The scanInteger() method takes in a String from an external file and returns an integer if
    * present.
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
    * The scanDouble() method takes in a String from an external file and returns a double if
    * present.
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
    * The scanString() method trims the whitespace and returns the desired part of a string from an
    * external file.
    */
   public String scanString(String inputString)
   {
      String[] parts = inputString.split("#");
      String stringPart = parts[0].trim();
      return stringPart;
   } // public String scanString(String inputString)

   /*
    * The readParametersFile() method reads in everything that is needed from a specified text file.
    */
   public void readParametersFile(String inputFileName)
   {
      int hiddens = 0;
      parametersFile = inputFileName;

      File pFile = new File(parametersFile);
      Scanner sc = null;
      try
      {
         sc = new Scanner(pFile);
      } catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      trainingOrRunning = scanString(sc.nextLine());
      randomOrSetWeights = scanString(sc.nextLine());
      numTestCases = scanInteger(sc.nextLine());
      numLayers = scanInteger(sc.nextLine());
      numInputs = scanInteger(sc.nextLine());
      numOutputs = scanInteger(sc.nextLine());
      maxNodes = 0;
      maxNodes = Math.max(maxNodes, numOutputs);

      for (int i = 0; i < numLayers - 2; i++)
      {
         hiddens = scanInteger(sc.nextLine());
         maxNodes = Math.max(maxNodes, hiddens);
      }

      init();
      sc.close();

      sc = null;
      try
      {
         sc = new Scanner(pFile);
      } catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      for (int i = 0; i < 6; i++)
      {
         sc.nextLine();
      }

      numNodesInLayer[0] = numInputs;
      numNodesInLayer[numLayers - 1] = numOutputs;

      for (int i = 1; i < numLayers - 1; i++)
      {
         numNodesInLayer[i] = scanInteger(sc.nextLine());
      }

      lambda = scanDouble(sc.nextLine());
      maxNumIterations = scanInteger(sc.nextLine());
      randomMin = scanDouble(sc.nextLine());
      randomMax = scanDouble(sc.nextLine());
      errorThreshold = scanDouble(sc.nextLine());
      testcasesFile = scanString(sc.nextLine());
      weightsFile = scanString(sc.nextLine());
      sc.close();
      offsetInFile = numLayers + 11;
   } // public void readParametersFile(String inputFileName)


   /*
    * The saveWeightsFile() method saves weights to a file.
    */
  public void saveWeightsFile()
  {
      try 
      {
         PrintWriter pw = new PrintWriter(new FileOutputStream(new File(weightsFile), false));
         pw.println(networkConfiguration);
         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < numInputs && n == 0; k++)
            {
               for (int j = 0; j < numNodesInLayer[n + 1]; j++)
               {
                  pw.println(weights[n][k][j]);
               }
            }

            for (int k = 0; k < numNodesInLayer[n] && n > 0 && n < numLayers - 2; k++)
            {
               for (int j = 0; j < numNodesInLayer[n + 1]; j++)
               {
                  pw.println(weights[n][k][j]);
               }
            }

            for (int k = 0; k < numNodesInLayer[n] && n == numLayers - 2; k++)
            {
               for (int j = 0; j < numOutputs; j++)
               {
                  pw.println(weights[n][k][j]);
               }
            }
         } // for (int k = 0; k < numLayers; k++)
         pw.close();
      } // try
      catch(Exception e)
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
      } catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }

      for (int i = 0; i < offsetInFile; i++)
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
      } catch (FileNotFoundException e)
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
         } catch (FileNotFoundException e)
         {
            e.printStackTrace();
         }

         if (sc.nextLine().equals(networkConfiguration))
         {
            for (int n = 0; n < numLayers; n++)
            {
               for (int k = 0; k < numInputs && n == 0; k++)
               {
                  for (int j = 0; j < numNodesInLayer[n + 1]; j++)
                  {
                     weights[n][k][j] = scanDouble(sc.nextLine());
                  }
               }
   
               for (int k = 0; k < numNodesInLayer[n] && n > 0 && n < numLayers - 2; k++)
               {
                  for (int j = 0; j < numNodesInLayer[n + 1]; j++)
                  {
                     weights[n][k][j] = scanDouble(sc.nextLine());
                  }
               }
   
               for (int k = 0; k < numNodesInLayer[n] && n == numLayers - 2; k++)
               {
                  for (int j = 0; j < numOutputs; j++)
                  {
                     weights[n][k][j] = scanDouble(sc.nextLine());
                  }
               }
            } // for (int k = 0; k < numLayers; k++)
         } // if (sc.nextLine().equals(networkConfiguration))
         else
         {
            System.out.println("Weights file does not match network configuration!");
         }
      } // if (randomOrSetWeights.equals("set"))

      if (randomOrSetWeights.equals("random")) // randomize weights
      {
         for (int n = 0; n < numLayers; n++)
         {
            for (int k = 0; k < numInputs && n == 0; k++)
            {
               for (int j = 0; j < numNodesInLayer[n + 1]; j++)
               {
                  weights[n][k][j] = randomize(randomMin, randomMax);
               }
            }

            for (int k = 0; k < numNodesInLayer[n] && n > 0 && n < numLayers - 2; k++)
            {
               for (int j = 0; j < numNodesInLayer[n + 1]; j++)
               {
                  weights[n][k][j] = randomize(randomMin, randomMax);
               }
            }

            for (int k = 0; k < numNodesInLayer[n] && n == numLayers - 2; k++)
            {
               for (int j = 0; j < numOutputs; j++)
               {
                  weights[n][k][j] = randomize(randomMin, randomMax);
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
      int n = INPUT_LAYER;
      for (int k = 0; k < numInputs; k++)
      {
         activations[n][k] = truthTable[inputIndex][k];
      }
   } // public void setInputs(int inputIndex)

   /*
    * The trainNetwork() method runs through the entire training of the network. It sets the inputs,
    * runs the forward evaluation training, backpropagation, and calculates weights. Then, it
    * calculates total error and average error by comparing the network's result to the desired
    * result. Once the max number of iterations has been reached, or the average error is less than
    * the error threshold, training is finished. trainNetwork() is called by the method
    * trainOrRun().
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

      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int k = 0; k < numNodesInLayer[n + 1]; k++)
         {
            tempTheta = 0.0;

            for (int m = 0; m < numNodesInLayer[n]; m++)
            {
               tempTheta += weights[n][m][k] * activations[n][m];
            }
            activations[n + 1][k] = sigmoidFunction(tempTheta);
         }
      }
   } // public void runNetworkCalculations()

   /*
    * The forwardEvaluationTraining() method calculates the hidden neurons on the forward pass.
    */
   public void forwardEvaluationTraining(int index)
   {
      double tempTheta, tempOmega;

      for (int n = 1; n < numLayers - 1; n++)
      {
         for (int j = 0; j < numNodesInLayer[n]; j++)
         {
            tempTheta = 0.0;

            for (int i = 0; i < numNodesInLayer[n - 1]; i++)
            {
               tempTheta += weights[n - 1][i][j] * activations[n - 1][i];
            }
            activations[n][j] = sigmoidFunction(tempTheta);
            theta[n][j] = tempTheta;
         } // for (int j = 0; j < numNodesInLayer[n]; j++)
      } // for (int n = 1; n < numLayers; n++)

      int n = numLayers - 1;
      for (int j = 0; j < numNodesInLayer[n]; j++)
      {
         tempTheta = 0.0;
         tempOmega = 0.0;

         for (int i = 0; i < numNodesInLayer[n - 1]; i++)
         {
            tempTheta += weights[n - 1][i][j] * activations[n - 1][i];
         }
         activations[n][j] = sigmoidFunction(tempTheta);
         tempOmega = expectedTruthTable[index][j] - activations[n][j];
         psi[n][j] = tempOmega * sigmoidDerivative(tempTheta);
      } // for (int j = 0; j < numNodesInLayer[n]; j++)
   } // public void forwardEvaluationTraining()

   /*
    * The backpropagation() method calculates the adjustment of weights using the backpropagation
    * algorithm.
    */
   public void backpropagation()
   {
      double tempOmega;

      for (int n = numLayers - 2; n > 1; n--)
      {
         tempOmega = 0.0;
         for (int j = 0; j < numNodesInLayer[n]; j++)
         {
            for (int i = 0; i < numNodesInLayer[n + 1]; i++)
            {
               tempOmega += psi[n + 1][i] * weights[n][j][i];
               weights[n][j][i] += lambda * activations[n][j] * psi[n + 1][i];
            }
            psi[n][j] = tempOmega * sigmoidDerivative(theta[n][j]);
         }
      } // for (int n = numLayers - 2; n > 1; n--)

      int x = INPUT_LAYER;
      for (int m = 0; m < numNodesInLayer[x + 1]; m++)
      {
         tempOmega = 0.0;
         for (int k = 0; k < numNodesInLayer[x + 2]; k++)
         {
            tempOmega += psi[x + 2][k] * weights[x + 1][m][k];
            weights[x + 1][m][k] += lambda * activations[x + 1][m] * psi[x + 2][k];
         }

         psi[x + 1][m] = tempOmega * sigmoidDerivative(theta[x + 1][m]);

         for (int k = 0; k < numInputs; k++)
         {
            weights[x][k][m] += lambda * activations[x][m] * psi[x + 1][m];
         }
      }
   } // public void backpropagation(int index)

   /*
    * The runNetwork() method runs on all of the test cases. It is called when the status of
    * trainingOrRunning is "running."
    */
   public void runNetwork()
   {
      for (int index = 0; index < numTestCases; index++)
      {
         setInputs(index);
         runNetworkCalculations();

         int n = numLayers - 1;
         for (int i = 0; i < numOutputs; i++)
         {
            actualTruthTable[index][i] = activations[n][i];
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
      return 1.0 / (1.0 + Math.exp(-x));
   } // public double sigmoidFunction(double x)

   /*
    * The sigmoidDerivative() method returns the value of a given x plugged into the derivative of
    * the sigmoid activation function.
    */
   public double sigmoidDerivative(double x)
   {
      double s = sigmoidFunction(x);
      return s * (1.0 - s);
   } // public double sigmoidFunction(double x)

   /*
    * The randomize() method returns a random value between the given min and max values.
    */
   public double randomize(double min, double max)
   {
      return min + (max - min) * Math.random();
   } // public double randomize(double min, double max)

   /*
    * The main() method runs the network's main methods. If no configuration file is passed in at
    * the command line, a default configuration file is used.
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
         inputFileString = DEFAULT_FILE_STRING;
      }
      NLayer perceptron = new NLayer();
      perceptron.readParametersFile(inputFileString);
      perceptron.echoConfigParameters();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults();
   } // public static void main(String[] args)
} // public class NLayer
