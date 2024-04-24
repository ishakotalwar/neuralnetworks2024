import java.io.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;
/*
 * @author Isha Kotalwar
 * @version 04/15/2024
 * 
 * The A-B-C perceptron is based on the A-B-1 project and the document "2-Minimizing the Error Function." It can train 
 * a network based on set weights provided by the user or randomized weights. It can also just run if a user provides weights.
 * Training happens through steepest descent.
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
   public double totalError;
   public double avgError;
   public double errorThreshold;

   public String trainingOrRunning;
   public String loadOrRandomOrSetWeights;
   public String reasonForStopping;
   public File inputFile;

   public double[] a;
   public double[] h;
   public double[] F;
   public double[] bigOmega;
   public double[] bigPsi;
   public double[] thetaJ;
   //public double[] smallOmega;
   public double[] smallPsi;
   public double[] thetaI;

   public double[][] weightskj;
   public double[][] weightsji;
   public double[][] truthTable;
   public double[][] expectedTruthTable;
   public double[][] actualTruthTable;

   public long totalRunTime;

   /*
   * The trainOrRun() method determines whether the network is going to train or
   * run.
   */
   public void trainOrRun() 
   {
      if (trainingOrRunning.equals("training")) 
      {
         trainNetwork();
      }

      else {
         runNetwork();
      }
   } // public void trainOrRun()

   /*
   * The init() method allocates memory and initializes all of the arrays that the
   * network will use.
   */
   public void init() {
      if (trainingOrRunning.equals("training")) 
      {
         bigOmega = new double[numHiddens];
         bigPsi = new double[numHiddens];
         thetaJ = new double[numHiddens];
         thetaI = new double[numOutputs];
         smallPsi = new double[numOutputs];
      } // if(trainingOrRunning.equals("training"))

      a = new double[numInputs];
      h = new double[numHiddens];
      F = new double[numOutputs];
      weightskj = new double[numInputs][numHiddens];
      weightsji = new double[numHiddens][numOutputs];

      truthTable = new double[numTestCases][numInputs];
      expectedTruthTable = new double[numTestCases][numOutputs];
      actualTruthTable = new double[numTestCases][numOutputs];

      truthTable[0][0] = 0.0;
      truthTable[0][1] = 0.0;
      expectedTruthTable[0][0] = 0.0;
      expectedTruthTable[0][1] = 0.0;
      expectedTruthTable[0][2] = 0.0;

      truthTable[1][0] = 0.0;
      truthTable[1][1] = 1.0;
      expectedTruthTable[1][0] = 0.0;
      expectedTruthTable[1][1] = 1.0;
      expectedTruthTable[1][2] = 1.0;

      truthTable[2][0] = 1.0;
      truthTable[2][1] = 0.0;
      expectedTruthTable[2][0] = 0.0;
      expectedTruthTable[2][1] = 1.0;
      expectedTruthTable[2][2] = 1.0;

      truthTable[3][0] = 1.0;
      truthTable[3][1] = 1.0;
      expectedTruthTable[3][0] = 1.0;
      expectedTruthTable[3][1] = 1.0;
      expectedTruthTable[3][2] = 0.0;

   } // public void init()

   /*
   * The echoConfigParameters() method prints out all of the parameters and the
   * network configuration.
   */
   public void echoConfigParameters() 
   {
      System.out.println("Network Configuration: " + numInputs + "-" + numHiddens + "-" + numOutputs);
      System.out.println("Training or running? " + trainingOrRunning);
      System.out.println("Randomize or set weights? " + loadOrRandomOrSetWeights);

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
         System.out.println("No valid integer found!");
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
   public void readFile(File iFile) 
   {
      inputFile = iFile;
      Scanner sc = null;

      try 
      {
         sc = new Scanner(inputFile);
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
      loadOrRandomOrSetWeights = scanString(sc.nextLine());
      randomMin = scanDouble(sc.nextLine());
      randomMax = scanDouble(sc.nextLine());
      errorThreshold = scanDouble(sc.nextLine());
      numTestCases = scanInteger(sc.nextLine());

      init();
   
      System.out.println(sc.nextLine());

      if (loadOrRandomOrSetWeights.equals("set")) 
      {
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
      } // if (loadOrRandomOrSetWeights.equals("set"))
      sc.close();
   } // public void readFile()

   /*
   * The populate() method fills all of the weight arrays with either random or
   * given values by the user. The truth table is also created here.
   */
   public void populate() 
   {
      if (loadOrRandomOrSetWeights.equals("random")) // randomize weights
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
      } // if(randomOrSetWeights.equals("random"))
   } // public void populate()

   /*
   * Initialize input values for a given input index.
   */
   public void setInputs(int inputIndex) 
   {
      for (int k = 0; k < numInputs; k++) 
      {
         a[k] = truthTable[inputIndex][k];
      }
   }

   /*
   * The trainNetwork() method runs through the entire training of the network. It
   * runs the network, calculates error, calculates weights, applies the weights,
   * runs the network, and calculates the average error over all of the
   * iterations. The network stops training if either the max number
   * of iterations has been reached, or the average error is less than the error
   * threshold. trainNetwork() is called by the method trainOrRun().
   */
   public void trainNetwork() 
   {
      long startTime = System.currentTimeMillis();
      totalRunTime = 0;
      totalError = 0.0;
      boolean finished = false;

      while (!finished)
      {
         avgError = 0.0;
         totalError = 0.0;

         for (int index = 0; index < numTestCases; index++) // iterates over every test case
         {
            setInputs(index);
            forwardEvaluationTraining(index);
            backpropagation(index);
            runNetworkCalculations();

            for (int i = 0; i < numOutputs; i++) 
            {
               totalError += 0.5 * (expectedTruthTable[index][i] - F[i]) * (expectedTruthTable[index][i] - F[i]);
               actualTruthTable[index][i] = F[i];
            }
         } // for (int index = 0; index < numTestCases; index++)
         avgError = totalError / ((double) numTestCases);

         if (avgError <= errorThreshold) 
         {
            finished = true;
            reasonForStopping = "the average error is below the error threshold.";
         }
         iterations++;
         if (iterations >= maxNumIterations) 
         {
            finished = true;
            reasonForStopping = "the maximum amount of iterations has been reached.";
         }
      } // while (!finished)
      totalRunTime = System.currentTimeMillis() - startTime;
   } // public void trainNetwork()

   /*
    * The runNetworkCalculations() method calculates the hidden neurons and the output neurons.
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
    * The forwardEvaluationTraining() method calculates the hidden neurons.
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
   * The backpropagation() method calculates the adjustment of weights using the backpropagation algorithm. 
   */
   public void backpropagation(int index) 
   {
      double omegaJ, upperPsiJ;
      for (int j = 0; j < numHiddens; j++) 
      {
         omegaJ = 0.0;
         for (int i = 0; i < numOutputs; i++) 
         {
            omegaJ += smallPsi[i] * weightsji[j][i];
            weightsji[j][i] += lambda * h[j] * smallPsi[i];
         } // for (int i = 0; i < numOutputs; i++)

         upperPsiJ = omegaJ * sigmoidFunction(thetaJ[j]);

         for (int k = 0; k < numInputs; k++) 
         {
            weightskj[k][j] += lambda * a[k] * upperPsiJ;
         } // for (int k = 0; k < numInputs; k++)
      } // for (int j = 0; j < numHiddens; j++)
   } // public void backpropagation(int index)

   /*
    * The runNetworkCalculations() method calculates the hidden neurons and the output neurons.
    */
   public void runNetwork() 
   {
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
   } // public void runNetwork();

   /*
    * The reportResults() method prints out the truth table and the output of the
    * network. If the network was training,
    * it also reports training information, like reason for stopping, number of
    * iterations, and error reached.
    */
   public void reportResults() 
   {
      System.out.println("***********************************");
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
         System.out.println("***********************************");
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
   } // public void reportResults()

   /*
    * The sigmoidFunction() method returns the value of a given x plugged into the
    * sigmoid activation function.
    */
   public double sigmoidFunction(double x) 
   {
      return 1.0 / (1 + Math.exp(-x));
   } // public double sigmoidFunction(double x)

   /*
    * The sigmoidDerivative() method returns the value of a given x plugged into
    * the derivative of the sigmoid activation function.
    */
   public double sigmoidDerivative(double x) 
   {
      double s = sigmoidFunction(x);
      return s * (1 - s);
   } // public double sigmoidFunction(double x)

   /*
    * The randomize() method returns a random value between the given min and max
    * values.
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
      // input file
      File iFile = new File("/Users/24IshaK/Desktop/NeuralNetworksProjects/" + args[0]);

      ABCBackprop perceptron = new ABCBackprop();
      perceptron.readFile(iFile);
      perceptron.echoConfigParameters();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults();
   } // public static void main(String[] args)
} // public class ABCBackProp
