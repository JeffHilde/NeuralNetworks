using System;
using RoboticGenetics;
using Xunit;

namespace NeuralNetTest
{
    public class UnitTestSynapticLayer
    {
        [Fact]
        public void TestConstructor()
        {
            var syn = new SynapticLayer(11, 13);

            Assert.True(syn.I == 11);
            Assert.True(syn.J == 13);

            Assert.Null(syn.SignalReceive);
            Assert.Null(syn.ErrorRespond);
            Assert.Null(syn.ErrorReceive);
            Assert.Null(syn.SignalRespond);

            Assert.NotNull(syn.Weights);
            Assert.NotNull(syn.DeltaWeights);

            Assert.True(syn.Weights.GetLength(0) == 13);
            Assert.True(syn.Weights.GetLength(1) == 11);

            Assert.True(syn.DeltaWeights.GetLength(0) == 13);
            Assert.True(syn.DeltaWeights.GetLength(1) == 11);
        }

        [Fact]
        public void TestForwardSignal()
        {
            var syn = new SynapticLayer(3, 5);

            syn.SignalReceive = new float[] { 1, 2, 3 };
            syn.SignalRespond = new float[] { 1, 1, 1, 1, 1 };

            syn.Weights = new float[,] { { 1, 2, 3 },
                                         { 0, 0, 0 },
                                         { 0, 0, 0 },
                                         { 0, 0, 0 },
                                         { 6, 5, 4 }};

            syn.ForwardSignal();

            Assert.True(syn.SignalRespond[0] == 14.0f);
            Assert.True(syn.SignalRespond[1] == 0.0f);
            Assert.True(syn.SignalRespond[2] == 0.0f);
            Assert.True(syn.SignalRespond[3] == 0.0f);
            Assert.True(syn.SignalRespond[4] == 28.0f);
        }

        [Fact]
        public void TestFeedBack()
        {
            int InputCount = 2, OutputCount = 3;

            SynapticLayer synLayerA = new SynapticLayer(InputCount, OutputCount);
            SynapticLayer synLayerB = new SynapticLayer(InputCount, OutputCount);

            var Input = new float[InputCount];
            var Delta = new float[OutputCount];

            synLayerA.SignalReceive = Input;
            synLayerA.SignalRespond = new float[OutputCount];

            synLayerA.ErrorReceive = new float[OutputCount];
            synLayerA.ErrorRespond = new float[InputCount];

            synLayerB.SignalReceive = Input;
            synLayerB.SignalRespond = new float[OutputCount];

            synLayerB.ErrorReceive = Delta;
            synLayerB.ErrorRespond = new float[InputCount];

            var rand = new Random(123456);

            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    synLayerA.Weights[j, i] = (float)rand.NextDouble() - 0.5F;
                    synLayerB.Weights[j, i] = (float)rand.NextDouble() - 0.5F;
                }
            }

            float sum = 0, err;

            for (int iteration = 0; iteration < 200; iteration++)
            {
                for (int i = 0; i < InputCount; i++)
                {
                    Input[i] = (float)rand.NextDouble() - 0.5F;
                }

                synLayerA.ForwardSignal();
                synLayerB.ForwardSignal();

                sum = 0;

                for (int j = 0; j < OutputCount; j++)
                {
                    err = synLayerB.ErrorReceive[j] = synLayerA.SignalRespond[j] - synLayerB.SignalRespond[j];

                    sum += Math.Abs(err);
                }

                synLayerB.FeedBackError();

                synLayerB.UpdateWeight(0.5f, 0.25f);
            }

            sum = 0;

            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    sum += Math.Abs(synLayerA.Weights[j, i] - synLayerB.Weights[j, i]);
                }
            }

            Assert.Equal(0.0, sum, 3);
        }

        [Fact]
        public void TestAll()
        {
            int InputCount = 2;
            int HiddenCount = 10;
            int OutputCount = 1;

            NeuralLayer InputLayer = new NeuralLayer(InputCount);
            NeuralLayer HiddenLayer = new NeuralLayer(HiddenCount);
            NeuralLayer OutputLayer = new NeuralLayer(OutputCount);

            SynapticLayer SynLayerIH = new SynapticLayer(InputCount, HiddenCount);
            SynapticLayer SynLayerHO = new SynapticLayer(HiddenCount, OutputCount);

            SynLayerIH.SignalReceive = InputLayer.SignalRespond;
            SynLayerIH.SignalRespond = HiddenLayer.SignalReceive;

            SynLayerHO.SignalReceive = HiddenLayer.SignalRespond;
            SynLayerHO.SignalRespond = OutputLayer.SignalReceive;

            SynLayerHO.ErrorReceive = OutputLayer.ErrorRespond;
            SynLayerHO.ErrorRespond = HiddenLayer.ErrorReceive;

            SynLayerIH.ErrorReceive = HiddenLayer.ErrorRespond;
            SynLayerIH.ErrorRespond = InputLayer.ErrorReceive;

            var rand = new Random(123456);

            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    SynLayerIH.Weights[j, i] = (float)rand.NextDouble();
                }
            }

            for (int i = 0; i < HiddenCount; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    SynLayerHO.Weights[j, i] = (float)rand.NextDouble();
                }
            }

            float O0 = 0;

            for (int i = 0; i < 10; i++)
            {
                for (float I0 = -1; I0 < 1.1f; I0 += 2.0f)
                {
                    for (float I1 = -1; I1 < 1.1f; I1 += 2.0f)
                    {
                        O0 = I1 < I0 ? 1 : (I1 > I0 ? 1 : -1);

                        //InputLayer.ForwardSignal();
                        InputLayer.SignalRespond[0] = I0;
                        InputLayer.SignalRespond[1] = I1;

                        SynLayerIH.ForwardSignal();
                        HiddenLayer.ForwardSignal();
                        SynLayerHO.ForwardSignal();
                        OutputLayer.ForwardSignal();

                        OutputLayer.ErrorRespond[0] = OutputLayer.SignalRespond[0] - O0;

                        OutputLayer.FeedBackError();
                        SynLayerHO.FeedBackError();
                        HiddenLayer.FeedBackError();
                        SynLayerIH.FeedBackError();
                        InputLayer.FeedBackError();

                        InputLayer.UpdateWeight(0.1f, 0.1f);
                        SynLayerIH.UpdateWeight(0.1f, 0.1f);
                        HiddenLayer.UpdateWeight(0.1f, 0.1f);
                        SynLayerHO.UpdateWeight(0.1f, 0.1f);
                        OutputLayer.UpdateWeight(0.1f, 0.1f);
                    }
                }


            }
        }
    }

    public class UnitTestNeuralLayer
    {
        [Fact]
        public void TestConstructor()
        {
            for (int i = 0; i < 3; i++)
            {
                var NL = new NeuralLayer(i);

                Assert.Equal(i, NL.I);

                Assert.NotNull(NL.SignalReceive);
                Assert.NotNull(NL.SignalRespond);
                Assert.NotNull(NL.DeltaBias);
                Assert.NotNull(NL.Bias);
                Assert.NotNull(NL.ErrorReceive);
                Assert.NotNull(NL.ErrorRespond);

                Assert.Equal(i, NL.SignalReceive.Length);
                Assert.Equal(i, NL.SignalRespond.Length);
                Assert.Equal(i, NL.DeltaBias.Length);
                Assert.Equal(i, NL.Bias.Length);
                Assert.Equal(i, NL.ErrorReceive.Length);
                Assert.Equal(i, NL.ErrorRespond.Length);
            }
        }

        [Fact]
        public void TestForwardSignal()
        {
            var rand = new Random(123456);

            var NLA = new NeuralLayer(1);
            var NLB = new NeuralLayer(1);

            NLA.SignalReceive = NLB.SignalReceive;
            NLA.Bias[0] = (float)rand.NextDouble();

            for (int iteration = 0; iteration < 10; iteration++)
            {
                NLA.SignalReceive[0] = (float)rand.NextDouble();

                NLA.ForwardSignal();
                NLB.ForwardSignal();

                NLB.ErrorReceive[0] = NLA.SignalRespond[0] - NLB.SignalRespond[0];

                NLB.FeedBackError();

                NLB.UpdateWeight(0.5f, 0.25f);
            }

            Assert.Equal(NLB.Bias[0], NLA.Bias[0], 3);
        }
    }

    public class UnitTestSigmoidNeuralLayer
    {
        [Fact]
        public void TestFeedBack()
        {
            var rand = new Random(123456);

            var NLA = new SigmoidNeuralLayer(1);
            var NLB = new SigmoidNeuralLayer(1);

            NLA.SignalReceive = NLB.SignalReceive;
            NLA.Bias[0] = (float)rand.NextDouble();

            for (int iteration = 0; iteration < 20; iteration++)
            {
                NLA.SignalReceive[0] = (float)rand.NextDouble();

                NLA.ForwardSignal();
                NLB.ForwardSignal();

                NLB.ErrorReceive[0] = NLA.SignalRespond[0] - NLB.SignalRespond[0];

                NLB.FeedBackError();

                NLB.UpdateWeight(0.5f, 0.25f);
            }

            Assert.Equal(NLB.Bias[0], NLA.Bias[0], 3);
        }
    }
}

