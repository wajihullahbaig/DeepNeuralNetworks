# DeepNeuralNetworks
Pure NUMPY based deep neural network implementation.
Python based Deep Neural Network code set - A "copy-paste" effort from Matlab  (rasmusbergpalm matlab code) to Python.
I have been learning about Python and Deep Learning. To fully understand the inners workings of Deep Neural Networks, 
I decided to translate the Matlab code from https://github.com/rasmusbergpalm/DeepLearnToolbox to Python. Please note any
improvements/suggestions are highly welcomed.

I have completed the code for the following

		1. NN
		2. CNN
		3. DBN
		4. SAE
		5. CAE - This seems to be incomplete from the original source. But will try to implement it once I understand more about it.
		
Note you are free to update this code and submit fixes. It is a code purely written for educational purpose. 
If you are stuck at hefty packages like caffe, cuda, torch , tensorflow and feel lost with deep neural networks, 
then the matlab version of the code or this python version is where you should start.

Please note that I have tried as much as possible to produce a "Copy-Paste" version of the code, despite that, 
you will find many code differences. Like handling of overlfows, logs, exponentials and classes. 

I hope you will enjoy using the code as much as I enjoyed translating it :)

# Note on code Translations
I have tried my best to keep the code in a more understanable way. I know there are various functions that could have been done far better and clean in a more pythonic way, but that does not serve the purpose of a clear code understanding. I have also tried my best to develop the code that is close to matlab in terms of accuracy.

        1. Random states for both Python and Matlab have been set using the 'twister' algorithm. This is important to attain correct accuracies.	
        2. For debugging the code, I have added a single line of code that runs Matlab/Pyhon in a sequential way.
        3. Collections in NN module are data structures used by the different classes in NN. These are helpful while writing clean code
 	 
# Dependencies/Dev Tools/Platform
	1- Anaconda 4.1.1
	2- PyDev
	3- Eclipse Luna
	4- Windows 10
	5- Matlab 2015a
	6- Python 3.5

# Accuracy comparison (Python vs Matlab)
It is evident that that both are close from following tables that accuracies are very close


With epochs = 1,  batches = 100, total batches = 600

(Full Batch Errors) 

| Neural Networks        | Python          | Matlab   |
| ---------------------- |:---------------:| --------:|
| Tanh                   | 0.071304        | 0.070366 | 
| L2 Decay		         | 0.066278        | 0.063632 | 
| Dropout		         | 0.071164        | 0.070079 | 
| Sigmoid		         | 0.067494        | 0.067301 | 
| Softmax		         | 0.246335        | 0.385170 |


With epochs = 1,  and batches = 50, total batches = 1200  

| CNN                            | Python          | Matlab   |
| ------------------------------ |:---------------:| --------:|
| 				                 | 0.1256          | 0.1229   |
|                                |                            |
| Network Structure              | 
|--------------------------------|
| C: OuputMaps=6, kernel size=5  |
| S: Scale =2        			 |
| C: OuputMaps=12, kernel size=5 | 
| S: Scale =2                    |  

With epochs = 1,  and batches = 100, total batches = 600  

| DBN                            | Python        | Matlab     |
| ------------------------------ |:-------------:| ----------:|
| size = 100	                 |    Reconstruction Error    |	
|                                |  66.15        | 66.2       |
| size = (100,100)               |    Reconstruction Error    |
|                                |  10.93        | 10.97      | 
| Unfolded to NN                 |    Full Batch Error        |	 					             
|								 |	0.0667		 | 0.067      |	


With epochs = 1,  and batches = 100, total batches = 600  

(Full Batch Errors)

| SAE (DE-NOISING)               | Python        | Matlab     |
| ------------------------------ |:-------------:| ----------:|
| size = (784,100,10)            |  13.46        | 12.70	  |                                 
|							     |				 |			  |	    	 					             
| Using FFNN to initialize NN    |  0.13         | 0.12       |
	
