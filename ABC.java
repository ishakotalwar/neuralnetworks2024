import java.io.*;

/* 
* @author Isha Kotalwar
* @version 04/15/2024
* 
* The A-B-C perceptron is based on the A-B-1 project and the document "2-Minimizing the Error Function." It can train 
* a network based on set weights provided by the user or randomized weights. It can also just run if a user provides weights.
* Training happens through steepest descent.
* 
* The methods saveWeightFiles() and loadWeightFiles() are adapted from the website: https://geeksprogramming.com/reading-and-writing-files-in-java/
*/

public class ABC
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
   public String weightsFile;

   public double[] a;
   public double[] h;
   public double[] F;
   public double[] bigOmega;
   public double[] bigPsi;
   public double[] thetaJ;
   public double[] smallOmega;
   public double[] smallPsi;
   public double[] thetaI;

   public double[][] weightskj;
   public double[][] weightsji;
   public double[][] deltaWeightskj;
   public double[][] deltaWeightsji;
   public double[][] truthTable;
   public double[][] expectedTruthTable;
   public double[][] actualTruthTable;

   /*
   * The setConfigParameters() method sets all of the parameters of the network. The user edits them here.
   */
  	 public void setConfigParameters()
   {
      numInputs = 3;
      numHiddens = 1;
      numOutputs = 3; 
      lambda = 0.3;
      maxNumIterations = 100000;
      errorThreshold = 2E-4;
      randomMin = 0.1;
      randomMax = 1.5;
      numTestCases = 4;
      trainingOrRunning = "training";
      loadOrRandomOrSetWeights = "random";
      weightsFile = "weightsABC.txt";
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
   * The init() method allocates memory and initializes all of the arrays that the network will use.
   */
   public void init()
   {
      if (trainingOrRunning.equals("training"))
      {
         deltaWeightskj = new double[numInputs][numHiddens];
         deltaWeightsji = new double[numHiddens][numOutputs];
         bigOmega = new double[numHiddens];
         bigPsi = new double[numHiddens];
         thetaJ = new double[numHiddens];
         thetaI = new double [numOutputs];
         smallPsi = new double [numOutputs];
         smallOmega = new double[numOutputs];
      } // if (trainingOrRunning.equals("training"))

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
   * The echoConfigParameters() method prints out all of the parameters and the network configuration. 
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
      } // if (trainingOrRunning.equals("training"))
   } // public void echoConfigParameters()

   /*
   * The saveWeightsFile() method saves weights to a file.
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
   * The loadWeightsFile() loads weights from a file.
   */
   public void loadWeightsFile()
   {
      try
      {
         BufferedReader reader = new BufferedReader(new FileReader(weightsFile));
         String line = reader.readLine();
         for (int k = 0; k < numInputs; k++)
         {
            for (int j = 0; j < numHiddens; j++)
            {
               double weight = Double.parseDouble(line);
               weightskj[k][j] = weight; 
               line = reader.readLine();
            } // for (j = 0; j < numHiddens; j++)
         } // for (k = 0; k < numInputs; k++)

         for (int j = 0; j < numHiddens; j++)
         {
            for (int i = 0; i < numOutputs; i++)
            {
               double weight = Double.parseDouble(line);
               weightsji[j][i] = weight; 
               line = reader.readLine();
            } // for (i = 0; i < numOutputs; i++)
         } // for (j = 0; j < numHiddens; j++)
      } // try
      catch (IOException e)
      {
         System.out.println("Exception " + e.toString());
      }
   } // public void loadWeightsFile()

   /*
   * The populate() method fills all of the weight arrays with either random or given values by the user.
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
      } // if (randomOrSetWeights.equals("random"))
      else if (loadOrRandomOrSetWeights.equals("set")) // manually set weights 
      {
         weightskj[0][0] = 0.75;
         weightskj[1][0] = 0.6;
            
         weightsji[0][0] = 0.5;
         weightsji[0][1] = 0.75;
         weightsji[0][2] = 0.35;  
      } // if (loadOrRandomOrSetWeights.equals("set"))
      else
      {
         loadWeightsFile();
      } // else
   } // public void populate()

   /*
   * The method setInputs() sets input values in the array a[] for a given input index.
   */
   public void setInputs(int inputIndex)
   {
      for (int k = 0; k < numInputs; k++)
      {
         a[k] = truthTable[inputIndex][k];
      }
   } // public void setInputs()

	/*
	 * The method setOutputs() sets the actual truth table actualTruthTable[][].
	 */
	public void setOutputs(int outputIndex)
	{
		for (int i = 0; i < numOutputs; i++)
		{
			actualTruthTable[outputIndex][i] = F[i];
		}
	} // public void setOutputs()

   /*
   * The trainNetwork() method runs through the entire training of the network. It runs the network, calculates error, calculates weights, applies the weights,
   * runs the network, and calculates the average error over all of the iterations. The network stops training if either the max number
   * of iterations has been reached, or the average error is less than the error threshold. trainNetwork() is called by the method trainOrRun().
   */
   public void trainNetwork()
   {
      totalError = 0.0;
		boolean finished = false;
		while (!finished)
		{
			avgError = 0.0;
         totalError = 0.0;  
         for (int index = 0; index < numTestCases; index++) // iterates over every test case
         {
            setInputs(index);
            runNetworkCalculations(); // forward pass to get the activations at the hidden and output layer
				doDeltaWeights(index);
            for (int i = 0; i < numOutputs; i++) // calculate error
            {
               double omega = expectedTruthTable[index][i] - F[i]; 
               totalError += 0.5 * omega * omega;
               actualTruthTable[index][i] = F[i];
            }
         } // for (int index = 0; index < numTestCases; index++)
         avgError = totalError / numTestCases;
			iterations++;

         if (avgError <= errorThreshold)
         {
            finished = true;
         }

      	if (iterations >= maxNumIterations)
      	{
				finished = true;
      	}
		} // while (!finished)
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
   * The runNetwork() method runs on all of the test cases. It is called when the status of trainingOrRunning is "running." 
   */
   public void runNetwork()
   {
      for (int index = 0; index < numTestCases; index++)
      {
         setInputs(index);
         runNetworkCalculations();
         setOutputs(index);
      } // for (int index = 0; index < numTestCases; index++)   
   } // public void runNetwork();

    
   /*
   * The calculateWeightskjAndji() method calculates the change of weights for the input to hidden weights (kj) and the hidden to output weights (ji). 
	* They are calculated using steepest descent.
   */
   public void calculateWeightskjAndji(int k, int j, int i, int testCase)
   {
      double partialDerivEOverWeightskj;
      thetaJ[j] = 0.0; 
      for (int K = 0; K < numInputs; K++)
      {
      	thetaJ[j] += a[K] * weightskj[K][j];
      }

      bigOmega[j] = 0.0;

      for (int I = 0; I < numOutputs; I++)
      {
         bigOmega[j] += smallPsi[I] * weightsji[j][I];
   	}

      bigPsi[j] = bigOmega[j] * sigmoidDerivative(thetaJ[j]);
      partialDerivEOverWeightskj = -a[k] * bigPsi[j];
      deltaWeightskj[k][j] = -lambda * partialDerivEOverWeightskj;

		double partialDerivEOverWeightsji;
      thetaI[i] = 0;

      for (int J = 0; J < numHiddens; J++)
      {
         thetaI[i] += h[J] * weightsji[J][i];
      }

      F[i] = sigmoidFunction(thetaI[i]);
      smallOmega[i] = expectedTruthTable[testCase][i] - F[i];
      smallPsi[i] = smallOmega[i] * sigmoidDerivative(thetaI[i]);
      partialDerivEOverWeightsji = -h[j] * smallPsi[i];
      deltaWeightsji[j][i] = -lambda * partialDerivEOverWeightsji;
   } // public void calculateWeightskjAndji(int k, int j, int testCase)

	/*
	* The doDeltaWeights() method calculates the best weights for the network, for both the input to hidden layers and hidden to output layers
   * by calling calculateWeightskjAndji(). It also applies the calculated weights to the weight arrays.
	*/
	public void doDeltaWeights(int index)
	{
		for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHiddens; j++)
         {
         	for (int i = 0; i < numOutputs; i++)
            {
               calculateWeightskjAndji(k, j, i, index);
            }
         }
      } // for (int k = 0; k < numInputs; k++)

      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHiddens; j++)
         {
         	weightskj[k][j] += deltaWeightskj[k][j];
         }
      } // for (int k = 0; k < numInputs; k++)

      for (int j = 0; j < numHiddens; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            weightsji[j][i] += deltaWeightsji[j][i];
         }
		} // for (int j = 0; j < numHiddens; j++)
	}
	
   /*
   * The reportResults() method prints out the truth table and the output of the network. If the network was training, 
   * it also reports training information, like reason for stopping, number of iterations, and error reached.
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
      return s  * (1.0 - s);
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
      ABC perceptron = new ABC();
      perceptron.setConfigParameters();
      perceptron.echoConfigParameters();
      perceptron.init();
      perceptron.populate();
      perceptron.trainOrRun();
      perceptron.reportResults(); 
   } // public static void main(String[] args)
} // public class ABC

