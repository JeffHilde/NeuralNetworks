using System;
using System.Collections.Generic;

namespace Inteligenics
{
	public interface ITrain
	{
		void ForwardSignal();

		void FeedBackError();

		void UpdateGain(float gain);

		void UpdateWeight(float gain, float momentum);
	}

	public class NeuralNetwork : SynapticLayer
	{
		public NeuralNetwork(int I, int J, params int[] HiddenLayerSizes) : base(I, J)
		{
			//I am here, really!
		}
	}

	public class SigmoidNeuralLayer : NeuralLayer
	{
		public SigmoidNeuralLayer(int I) : base(I)
		{

		}

		public override void ForwardSignal()
		{
			for (int i = 0; i < I; i++)
			{
				SignalRespond[i] = 2 / (1 + (float)Math.Exp( - 2 * (SignalReceive[i] + Bias[i]))) - 1;
			}
		}



		public override void FeedBackError()
		{
			for (int i = 0; i < I; i++)
			{
				ErrorRespond[i] = ErrorReceive[i] * (1 - SignalRespond[i] * SignalRespond[i]);

				DeltaBias[i] += ErrorRespond[i];
			}
		}

		public override void UpdateGain(float gain)
		{
			for (int i = 0; i < I; i++)
			{
				SignalRespond[i] = 2 / (1 + (float)Math.Exp( - 2 * (SignalReceive[i] + (Bias[i] + gain * DeltaBias[i])))) - 1;
			}
		}
	}

	public class NeuralLayer : ITrain
	{
		public int I;

		// [I]

		public float[] SignalReceive, SignalRespond;

		public float[] ErrorReceive, ErrorRespond;

		public float[] Bias, DeltaBias;

		public NeuralLayer(int I)
		{
			this.I = I;

			Bias = new float[I];

			DeltaBias = new float[I];

			SignalReceive = new float[I];

			SignalRespond = new float[I];

			ErrorReceive = new float[I];

			ErrorRespond = new float[I];
		}



		public virtual void ForwardSignal()
		{
			for (int i = 0; i < I; i++)
			{
				SignalRespond[i] = SignalReceive[i] + Bias[i];
			}
		}

		public virtual void FeedBackError()
		{
			for (int i = 0; i < I; i++)
			{
				DeltaBias[i] += ErrorReceive[i];
				ErrorRespond[i] = ErrorReceive[i];
			}
		}

		public virtual void UpdateGain(float gain)
		{
			for (int i = 0; i < I; i++)
			{
				SignalRespond[i] = SignalReceive[i] + (Bias[i] + gain * DeltaBias[i]);
			}
		}

		public virtual void UpdateWeight(float gain, float momentum)
		{
			for (int i = 0; i < I; i++)
			{
				Bias[i] += gain * DeltaBias[i];

				DeltaBias[i] = momentum * DeltaBias[i];
			}
		}
	}

	public class SynapticLayer : ITrain
	{
		public int I, J;

		// [I]
		public float[] SignalReceive, ErrorRespond;

		// [J]
		public float[] ErrorReceive, SignalRespond;

		// [J, I]
		public float[,] Weights, DeltaWeights;

		public SynapticLayer(int I, int J)
		{
			this.I = I;

			this.J = J;

			Weights = new float[J, I];

			DeltaWeights = new float[J, I];
		}



		public virtual void ForwardSignal()
		{
			for (int j = 0; j < J; j++)
			{
				float sum = 0;

				for (int i = 0; i < I; i++)
				{
					sum += SignalReceive[i] * Weights[j, i];
				}

				SignalRespond[j] = sum;
			}
		}



		public virtual void FeedBackError()
		{
			for (int i = 0; i < I; i++)
			{
				ErrorRespond[i] = 0;
			}

			for (int j = 0; j < J; j++)
			{
				for (int i = 0; i < I; i++)
				{
					DeltaWeights[j, i] += SignalReceive[i] * ErrorReceive[j];

					ErrorRespond[i] += ErrorReceive[j] * Weights[j, i];
				}
			}
		}



		public void UpdateGain(float gain)
		{
			for (int j = 0; j < J; j++)
			{
				float sum = 0;

				for (int i = 0; i < I; i++)
				{
					sum += SignalReceive[i] * (Weights[i, i] + gain * DeltaWeights[i, i]);
				}

				SignalRespond[j] = sum;
			}
		}

		public void UpdateWeight(float gain, float momentum)
		{
			for (int j = 0; j < J; j++)
			{
				for (int i = 0; i < I; i++)
				{
					Weights[j, i] += gain * DeltaWeights[j, i];

					DeltaWeights[j, i] = momentum * DeltaWeights[j, i];
				}
			}
		}
	}
}
