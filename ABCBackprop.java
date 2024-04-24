import java.io.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;
/*
 * @author Isha Kotalwar
 * 
 * @version 04/22/2024
 * 
 * This A-B-C perceptron is based on the A-B-1 project, previous A-B-C project and the document
 * "3-Minimizing and Optimizing the Error Function." It can train a network based on set weights provided by the
 * user or randomized weights. The parameters are given to the network through an external configuration file. 
 * The network trains through backpropagation.
 */

public class ABCBackprop
{
   public int numInputs;
   public int numHiddens;
   public int numOutputs;
   public int maxNumIterations;
   public int numTestCases;
   public int iterations;

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
   public static final String DEFAULT_FILE_STRING = "default.txt";


   public double[] a;
   public double[] h;
   public double[] F;
   public double[] bigOmega;
   public double[] bigPsi;
   public double[] thetaJ;
   public double[] smallPsi;

   public double[][] weightskj;
   public double[][] weightsji;
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
         bigOmega = new double[numHiddens];
         bigPsi = new double[numHiddens];
         thetaJ = new double[numHiddens];
         smallPsi = new double[numOutputs];
      } // if (trainingOrRunning.equals("training"))

      a = new double[numInputs];
      h = new double[numHiddens];
      F = new double[numOutputs];
      
      weightskj = new double[numInputs][numHiddens];
      weightsji = new double[numHiddens][numOutputs];

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
            .println("Network Configuration: " + numInputs + "-" + numHiddens + "-" + numOutputs);
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
   * The scanInteger() method takes in a String from an external file
   * and returns an integer if present.
   */
   public int scanInteger(String inputString)
   {
      int number = 0;
      String[] parts = inputString.split("#");
      String numberPart = parts[0].trim(); 

      Scanner numberScanner = new Scanner(numberPart);
      if (numberScanner.hasNextInt())
      {
         number = numberScanner.nextInt();
      }
      else
      {
         System.out.println("No valid integer found!");
      }

      numberScanner.close();
      return number;
   } // public double scanInteger(String inputString)

   /*
   * The scanDouble() method takes in a String from an external file
   * and returns a double if present.
   */
   public double scanDouble(String inputString)
   {
      double number = 0.0;
      String[] parts = inputString.split("#");
      String numberPart = parts[0].trim(); 

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
   * The scanString() method trims the whitespace and returns the desired part of a string
   * from an external file.
   */
   public String scanString(String inputString)
   {
      String[] parts = inputString.split("#");
      String stringPart = parts[0].trim(); 
      return stringPart;
   } // public String scanString(String inputString)

   /*
   * The readParametersFile() method reads in each parameter from a specified configuration file.
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
      numHiddens = scanInteger(sc.nextLine());
      numOutputs = scanInteger(sc.nextLine());
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
   * The saveWeightsFile() method saves weights to an external file.
   */
  public void saveWeightsFile()
  {
      try 
      {
         PrintWriter pw = new PrintWriter(new FileOutputStream(new File(weightsFile), false));

         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHiddens; j++)
            {
               pw.println(weightskj[k][j]);
            }
         } // for (k = 0; k < nInputs; k++)

         for (int j = 0; j < numHiddens; j++)
         {
            for (int i = 0; i < numOutputs; i++)
            {
               pw.println(weightsji[j][i]);
            }
         } // for (j = 0; j < nHidden; j++)
         pw.close();
      } // try
      catch (Exception e) 
      {
         System.out.println("Exception" + e.toString());
      }
  } // public void saveWeightsFile()

   /*
   * The populate() method fills all of the weight arrays with either random values or values from
   * an external file. It also fills in the truth table test cases from an external file.
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

      for (int i = 0; i < 13; i++)
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
      } // for (int i = 0; i < numTestCases; i++)
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
      } // for (int i = 0; i < numTestCases; i++)

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

         for (int i = 0; i < numInputs; i++) 
         {
            for (int j = 0; j < numHiddens; j++) 
            {
               weightskj[i][j] = scanDouble(sc.nextLine());
            }
         } // for (int i = 0; i < numInputs; i++)

         for (int i = 0; i < numHiddens; i++) 
         {
            for (int j = 0; j < numOutputs; j++) 
            {
               weightsji[i][j] = scanDouble(sc.nextLine());
            }
         } // for (int i = 0; i < numHiddens; i++)
      } // if (randomOrSetWeights.equals("set"))

      if (randomOrSetWeights.equals("random")) 
      {
         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHiddens; j++)
            {
               weightskj[k][j] = randomize(randomMin, randomMax);
            }
         }

         for (int k = 0; k < numHiddens; k++)
         {
            for (int j = 0; j < numOutputs; j++)
            {
               weightsji[k][j] = randomize(randomMin, randomMax);
            }
         }
      } // if (randomOrSetWeights.equals("random"))
   } // public void populate()

   /*
   * The method setInputs() initialized the input values for a given index.
   */
   public void setInputs(int inputIndex)
   {
      for (int k = 0; k < numInputs; k++)
      {
         a[k] = truthTable[inputIndex][k];
      }
   } // public void setInputs(int inputIndex)

   /*
   * The trainNetwork() method runs through the entire training of the network. It sets the
   * inputs, runs the forward evaluation training, backpropagation, and calculates weights. Then, 
   * it calculates total error and average error by comparing the network's result to the desired result.
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
               double omega = (expectedTruthTable[index][i] - F[i]);
               totalError += 0.5 * omega * omega;
               actualTruthTable[index][i] = F[i];
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
   * The runNetworkCalculations() method calculates the value of hidden activations
   * and the output activations.
   */
   public void runNetworkCalculations()
   {
      double thetaj;
      double thetai;

      for (int j = 0; j < numHiddens; j++)
      {
         thetaj = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            thetaj += weightskj[k][j] * a[k];
         }
         h[j] = sigmoidFunction(thetaj);
      } // for (int j = 0; j < numHiddens; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         thetai = 0.0;
         for (int j = 0; j < numHiddens; j++)
         {
            thetai += weightsji[j][i] * h[j];
         }
         F[i] = sigmoidFunction(thetai);
      } // for (int i = 0; i < numOutputs; i++)
   } // public void runNetworkCalculations()

   /*
   * The forwardEvaluationTraining() method calculates the value of hidden activations 
   * based on the input activations.
   */
   public void forwardEvaluationTraining(int index)
   {
      double tempTheta, smallOmega;

      for (int j = 0; j < numHiddens; j++)
      {
         tempTheta = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            tempTheta += weightskj[k][j] * a[k];
         }
         h[j] = sigmoidFunction(tempTheta);
         thetaJ[j] = tempTheta;
      } // for (int j = 0; j < numHiddens; j++)

      for (int i = 0; i < numOutputs; i++)
      {
         tempTheta = 0.0;
         for (int j = 0; j < numHiddens; j++)
         {
            tempTheta += weightsji[j][i] * h[j];
         }
         F[i] = sigmoidFunction(tempTheta);
         smallOmega = expectedTruthTable[index][i] - F[i];
         smallPsi[i] = smallOmega * sigmoidDerivative(tempTheta);
      } // for (int i = 0; i < numOutputs; i++)
   } // public void forwardEvaluationTraining()

   /*
   * The backpropagation() method calculates the adjustment of weights using the backpropagation
   * algorithm.
   */
   public void backpropagation()
   {
      double omegaJ, bigPsiJ;

      for (int j = 0; j < numHiddens; j++)
      {
         omegaJ = 0.0;
         for (int i = 0; i < numOutputs; i++)
         {
            omegaJ += smallPsi[i] * weightsji[j][i];
            weightsji[j][i] += lambda * h[j] * smallPsi[i];
         } // for (int i = 0; i < numOutputs; i++)

         bigPsiJ = omegaJ * sigmoidDerivative(thetaJ[j]);

         for (int k = 0; k < numInputs; k++)
         {
            weightskj[k][j] += lambda * a[k] * bigPsiJ;
         } // for (int k = 0; k < numInputs; k++)
      } // for (int j = 0; j < numHiddens; j++)
   } // public void backpropagation()

   /*
   * The runNetwork() method runs on all of the test cases. 
   */
   public void runNetwork()
   {
      long startTime = System.currentTimeMillis();
      totalRunTime = 0;

      for (int index = 0; index < numTestCases; index++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            a[k] = truthTable[index][k];
         }

         runNetworkCalculations();

         for (int i = 0; i < numOutputs; i++)
         {
            actualTruthTable[index][i] = F[i];
         }
      } // for (int index = 0; index < numTestCases; index++)
      totalRunTime = System.currentTimeMillis() - startTime;
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
      else
      {
         System.out.println("Total execution time: " + totalRunTime + " ms");
      }
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
   * The main() method runs the network's main methods.
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
      ABCBackprop perceptron = new ABCBackprop();
      perceptron.readParametersFile(inputFileString);
      perceptron.echoConfigParameters();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults();
   } // public static void main(String[] args)
} // public class ABCBackProp