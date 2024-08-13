# Coursework for COMP3710 Pattern Recognition and Analysis

## Set up and Dependencies

Ideally this should run in a virtual environment such as miniconda

will need to install numpy, matplotlib and pytorch
e.g.

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install matplotlib
```

The parallelised version of the Newton fractal was adapted from the following
source https://scipython.com/book2/chapter-8-scipy/examples/the-newton-fractal/

Help was sought from chatGPT for explanations, debugging, syntax clarification and feedback on code, and the conversation is [available here](https://chatgpt.com/share/8fa06898-1d53-4390-a9a8-8928e0405145)

I encountered issues with the lack of support for operations on complex numbers in PyTorch. I ended up having to use a workaround that decomposed the complex values into their real and imaginary parts, and then recombining and finding unique values - code was assisted using chatGPT
