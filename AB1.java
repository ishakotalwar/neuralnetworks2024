import java.io.*;

/*
 * @author Isha Kotalwar
 * @version 04/12/2024
 * 
 * The A-B-1 perceptron is based on the previous spreadsheet project and the document "1-Minimization the Single Output Error Function."
 * It can train a network based on set weights provided by the user or randomized weights. It can also just run if a user provides weights.
 * Training happens through steepest descent.
 */


public class AB1 
{
   public int numInputs;
   public int numHiddenLayers;
   public int numOutputs;
   public int maxNumIterations;
   public int numTestCases;
   public int numIterations;

   public double lambda;
   public double randomMin;
   public double randomMax;
   public double totalError;
   public double avgError;
   public double errorThreshold;

   public String trainingOrRunning;
   public String randomOrSetWeights;
   public String reasonForStopping;

   public double[] inputNeurons;
   public double[] hiddenNeurons;
   public double[] outputNeurons;
   public double[] bigOmega;
   public double[] bigPsi;
   public double[] bigHiddenTheta;

   public double[][] weightskj;
   public double[][] weightsj0;
   public double[][] deltaWeightskj;
   public double[][] deltaWeightsj0;
   public double[][] truthTable;
   public double[][] expectedTruthTable;
   public double[][] actualTruthTable;

   /*
   * The setConfigParameters() method sets all of the parameters of the network. The user edits them here.
   */
   public void setConfigParameters()
   {
      numInputs = 2;
      numHiddenLayers = 1;
      numOutputs = 1; // always 1 for AB1 network
      lambda = 0.3;
      maxNumIterations = 100000;
      errorThreshold = 2E-4;
      randomMin = -1.5;
      randomMax = 1.5;
      numTestCases = 4;
      trainingOrRunning = "running";
      randomOrSetWeights = "random";
   } // public void setConfigParameters()

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
   * The init() method allocates memory and initializes all of the arrays that the network will use. The truth table is also created here.
   */
   public void init()
   {
      if (trainingOrRunning.equals("training"))
      {
         deltaWeightskj = new double[numInputs][numHiddenLayers];
         deltaWeightsj0 = new double[numHiddenLayers][numOutputs];
         bigOmega = new double[numHiddenLayers];
         bigPsi = new double[numHiddenLayers];
         bigHiddenTheta = new double[numHiddenLayers];
       } // if (trainingOrRunning.equals("training"))

      inputNeurons = new double[numInputs];
      hiddenNeurons = new double[numHiddenLayers];
      outputNeurons = new double[numOutputs];
      weightskj = new double[numInputs][numHiddenLayers];
      weightsj0 = new double[numHiddenLayers][numOutputs];

      truthTable = new double[numTestCases][numInputs];
      expectedTruthTable = new double[numTestCases][numOutputs];
      actualTruthTable = new double[numTestCases][numOutputs];

      truthTable[0][0] = 0.0;
      truthTable[0][1] = 0.0;
      expectedTruthTable[0][0] = 0.0;

      truthTable[1][0] = 0.0;
      truthTable[1][1] = 1.0;
      expectedTruthTable[1][0] = 1.0;

      truthTable[2][0] = 1.0;
      truthTable[2][1] = 0.0;
      expectedTruthTable[2][0] = 1.0;

      truthTable[3][0] = 1.0;
      truthTable[3][1] = 1.0;
      expectedTruthTable[3][0] = 0.0;
   } // public void init()
   
   /*
   * The echoConfigParameters() method prints out all of the parameters and the network configuration. 
   */
   public void echoConfigParameters()
   {
      System.out.println("Network Configuration: " + numInputs + "-" + numHiddenLayers + "-" + numOutputs);
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
   * The populate() method fills all of the weight arrays with either random or given values by the user.
   */
   public void populate()
   {
      if (randomOrSetWeights.equals("random"))
      {
         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHiddenLayers; j++)
            {
               weightskj[k][j] = randomize(randomMin, randomMax);
            }
          }

         for (int k = 0; k < numHiddenLayers; k++)
         {
            for (int j = 0; j < numOutputs; j++) // numOutputs is always 1 for AB1
            {
               weightsj0[k][j] = randomize(randomMin, randomMax); 
            }
         }
      } // if (randomOrSetWeights.equals("random"))

      else // manually set weights 
      {
         weightskj[0][0] = 0.75;
         weightskj[1][0] = 0.6;
         weightskj[0][1] = 0.25;
         weightskj[1][1] = 0.9;

         weightsj0[0][0] = 0.5;
         weightsj0[1][0] = 0.05;  
      } // else
   } // public void populate()


   /*
   * The trainNetwork() method runs through the entire training of the network. It runs the network, calculates error, calculates weights, applies the weights,
   * runs the network, and calculates the average error over all of the iterations. The network stops training if either the max number
   * of iterations has been reached, or the average error is less than the error threshold. trainNetwork() is called by the method trainOrRun().
   */
   public void trainNetwork()
   {
      totalError = 0.0;
      numIterations = 0;
      boolean finished = false;

      while (!finished)
      {
         avgError = 0.0;
         totalError = 0.0;
            
         for (int index = 0; index < numTestCases; index++) // iterates over every test case
         {
            for (int k = 0; k < numInputs; k++)
            {
               inputNeurons[k] = truthTable[index][k];
            }
            runNetwork();
            doDeltaWeights(index);
            runNetwork();

            double omega = expectedTruthTable[index][numOutputs - 1] - outputNeurons[numOutputs - 1]; // numOutputs - 1 is always 0 for AB1
            totalError += 0.5 * omega * omega;
            actualTruthTable[index][0] = outputNeurons[0];
         } // for (int index = 0; index < numTestCases; index++)
         avgError = totalError / numTestCases;
        
         if (avgError <= errorThreshold)
         {
            finished = true;
            reasonForStopping = "the average error is below the error threshold.";
         }

         numIterations++;

         if (numIterations >= maxNumIterations)
         {
            finished = true;
            reasonForStopping = "the maximum amount of iterations has been reached.";
         }
      }  // while (!finished)
   } // public void trainNetwork()


   /*
   * The runNetworkCalculations() method calculates the hidden neurons and the output neurons.
   */
   public void runNetworkCalculations()
   {
      double hiddenTheta; 
      double outputTheta;
      for (int j = 0; j < numHiddenLayers; j++)
      {
         hiddenTheta = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            hiddenTheta += weightskj[k][j] * inputNeurons[k];
         }
         hiddenNeurons[j] = sigmoidFunction(hiddenTheta);
      } // for (int j = 0; j < numHiddenLayers; j++)

      for (int j = 0; j < numOutputs; j++)
      {
         outputTheta = 0.0;
         for (int k = 0; k < numHiddenLayers; k++)
         {
            outputTheta += weightsj0[k][j] * hiddenNeurons[k];
         }
         outputNeurons[j] = sigmoidFunction(outputTheta);
      } // for (int j = 0; j < numHiddenLayers; j++)
   } // public void runNetworkCalculations()

   /*
   * The runNetwork() method runs on all of the test cases if the status of trainingOrRunning is "running." Otherwise,
   * it calls runNetworkCalculations(), which is used in training.
   */
   public void runNetwork()
   {
      if (trainingOrRunning.equals("running"))
      {
         for (int index = 0; index < numTestCases; index++)
         {
            for (int i = 0; i < numInputs; i++)
            {
               inputNeurons[i] = truthTable[index][i];
            }
               runNetworkCalculations();
               actualTruthTable[index][0] = outputNeurons[0];
         } // for (int index = 0; index < numTestCases; index++)
      } // if (trainingOrRunning.equals("running"))
      else
      {
         runNetworkCalculations();
      } // else
   } // public void runNetwork();

    
   /*
   * The calculateWeightskj() method calculates the change of weights for the input to hidden weights (kj). They are calculated
   * using steepest descent.
   */
   public void calculateWeightskj(int k, int j, int testCase)
   {
      double bigTheta0, smallPsi, partialDerivEOverWeightskj;
      double smallOmega = expectedTruthTable[testCase][0] - outputNeurons[0];
      bigHiddenTheta[j] = 0.0; 

      for (int K = 0; K < numInputs; K++)
      {
         bigHiddenTheta[j] += inputNeurons[K] * weightskj[K][j];
      }

      hiddenNeurons[j] = sigmoidFunction(bigHiddenTheta[j]);
      bigTheta0 = 0.0;

      for (int J = 0; J < numHiddenLayers; J++)
      {
         bigTheta0 += hiddenNeurons[J] * weightsj0[J][0];
      }

      smallPsi = smallOmega * sigmoidDerivative(bigTheta0);
      bigOmega[j] = smallPsi * weightsj0[j][0];
      bigPsi[j] = bigOmega[j] * sigmoidDerivative(bigHiddenTheta[j]);
      partialDerivEOverWeightskj = -inputNeurons[k] * bigPsi[j];
      deltaWeightskj[k][j] = -lambda * partialDerivEOverWeightskj;
   } // public void calculateWeightskj(int k, int j, int testCase)


   /*
   * The calculateWeightsj0() method calculates the change of weights for the hidden to output weights (j0). They are calculated
   * using steepest descent.
   */
   public void calculateWeightsj0(int j, int testCase)
   {
      double bigTheta0, smallPsi, smallOmega, partialDerivEOverWeightsj0;
      bigTheta0 = 0.0;

      for (int J = 0; J < numHiddenLayers; J++)
      {
         bigTheta0 += hiddenNeurons[J] * weightsj0[J][0];
      }

      outputNeurons[0] = sigmoidFunction(bigTheta0);
      smallOmega = expectedTruthTable[testCase][0] - outputNeurons[0];
      smallPsi = smallOmega * sigmoidDerivative(bigTheta0);
      partialDerivEOverWeightsj0 = -hiddenNeurons[j] * smallPsi;
      deltaWeightsj0[j][0] = -lambda * partialDerivEOverWeightsj0;
   } // public void calculateWeightsj0(int j, int testCase)


   /*
   * The doDeltaWeights() method calculates the best weights for the network, for both the input to hidden layers and hidden to output layers
   * by calling calculateWeightskj() and calculateWeightsj0(). It also applies the calculated weights to the weight arrays.
   */
   public void doDeltaWeights(int testCase)
   {
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHiddenLayers; j++)
         {
            calculateWeightskj(k, j, testCase);
            calculateWeightsj0(j, testCase);
         }
      }

      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHiddenLayers; j++)
         {
            weightskj[k][j] += deltaWeightskj[k][j];
         }
      }

      for (int j = 0; j < numHiddenLayers; j++)
      {
         for (int i = 0; i < numOutputs; i++) // only 1 output for AB1
         {
            weightsj0[j][i] += deltaWeightsj0[j][i];
         }
      }
   } // public doDeltaWeights(int testCase)

   /*
   * The reportResults() method prints out the truth table and the output of the network. If the network was training, 
   * it also reports training information, like reason for stopping, number of iterations, and error reached.
   */
   public void reportResults()
   {
      System.out.println("***********************************");
      System.out.println("Truth Table");
      System.out.println("neuron1\tneuron2\t truth\t  output neuron");

      for (int row = 0; row < numTestCases; row++)
      {
         String output = "";
         for (int i = 0; i < numInputs; i++)
         {
            output += truthTable[row][i] + "\t  ";
         }

         output += expectedTruthTable[row][0] + "\t  ";
         output += actualTruthTable[row][0];
         System.out.println(output);
      } // for (row = 0; row < numTestCases; row++)

      if (trainingOrRunning.equals("training"))
      {
         System.out.println("***********************************");
         System.out.println("Training Info:");
         System.out.println("Training ended because " + reasonForStopping);
         System.out.println("Number of iterations reached: " + numIterations);
         System.out.println("Error reached: " + avgError);
         System.out.println();
      } // if (trainingOrRunning.equals("training"))
   } // public void reportResults()

   /*
   * The sigmoidFunction() method returns the value of a given x plugged into the sigmoid activation function.
   */
   public double sigmoidFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // public double sigmoidFunction(double x)

   /*
   * The sigmoidDerivative() method returns the value of a given x plugged into the derivative of the sigmoid activation function.
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
   * The main() method runs the network's 6 main methods.
   */
   public static void main(String[] args) 
   {
      AB1 perceptron = new AB1();
      perceptron.setConfigParameters();
      perceptron.echoConfigParameters();
      perceptron.init();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults(); 
   } // public static void main(String[] args)
} // public class AB1