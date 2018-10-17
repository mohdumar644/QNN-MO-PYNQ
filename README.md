# QNN-MO-PYNQ  Example

This repo provides a complete workflow for implementing a quantized ConvNet on MNIST using QNN-MO-PYNQ.

It only includes additional files for the MNIST demo. Please refer to the original repository for details. 


## Quick Start

Install QNN-MO-PYNQ as described in their repo.

Clone this repo into a suitable folder on the PYNQ board (preferably ``jupyter_notebooks/QNN``).

Then run the MNIST example from  ``notebooks/mnist.ipynb``.


## Repo organization

The repo is organized as follows:

-	qnn:  
	-	src: 
		- network: overlay topology (W1A2 MNIST) HLS top functions, host code and make script for HW and SW built
		- training: python scripts for training the network in Tensorpack, and the Finnthesizer script for QNN
	-	bitstreams: .bit, .tcl and network .json for the above network
	-	params: set of trained parameters and the layers .json for the MNIST overlay
	
-	notebooks: MNIST W1A2 jupyter notebook

 
 
