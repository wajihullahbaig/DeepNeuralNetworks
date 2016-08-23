# DeepNeuralNetworks
Python based Deep Neural Network code set - A "copy-paste" effort from Matlab  (rasmusbergpalm matlab code) to Python.
I have been learning about Python and Deep Learning. To fully understand the inners workings of Deep Neural Networks, 
I decided to translate the Matlab code from https://github.com/rasmusbergpalm/DeepLearnToolbox to Python.

Note you are free to update this code and submit fixes. It is a code purely written for educational purpose. 
If you are stuck at hefty packages like caffe, cuda, torch , tensorflow and feel lost with deep neural networks, 
then the matlab version of the code or this python version is where you should start.

This is currently a work in progress. I will be comitting codes as soon as they are stable. 
Please note that I have tried as much as possible to produce a "Copy-Paste" version of the code, despite that, you will find many code differences. Like handling of overlfows, logs, exponentials and classes. 

I hope you will enjoy using the code as much as I enjoyed translating it :)

# Note on code Translations
I have tried my best to keep the code in a more understanable way. I know there are various functions that could have been done far better and clean in a more pythonic way, but that does not serve the purpose of a clear code understanding. I have also tried my best to develop the code that is close to matlab in terms of accuracy.

 ## Translation notes
Markup : 1. Random states for both Python and Matlab have been set using the 'twister' algorithm
 	 2. For debugging the code, I have added a single line of code that runs Matlab/Pyhon in a sequential way.
 	 3. Some functions from Matlab like nnsetup() have been replaced by a constructor in Python like FFBONN.NN
 	 
Markup : 1. A numbered list
           1. A nested numbered list
           2. Which is numbered
          2. Which is numbered
          3. 
# Dependencies/Dev Tools/Platform
	1- Anaconda 4.1.1
	2- PyDev
	3- Eclipse Luna
	4- Windows 10
	5- Matlab 2015a
	6- Python 3.5

# Accuracy comparison (Python vs Matlab)
The following table shows the full batch errors produced by Python and Matlab. It is evident that that both are close 
in terms of accuracy.  
With epochs = 1, we have the following outputs

| Neural Networks        | Python          | Matlab   |
| ---------------------- |:---------------:| --------:|
| Vanilla		         | 0.071304        | 0.070366 | 
| L2 Decay		         | 0.066278        | 0.063632 | 
| Dropout		         | 0.071164        | 0.070079 | 
| Sigmoid		         | 0.067494        | 0.067301 | 
| Softmax		         | 0.246335        | 0.385170 |
