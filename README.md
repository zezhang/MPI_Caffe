#Major Changes in the following files:


/tools/caffe.cpp:
		Initialize the MPI environment. 
		Update the neural network after each iteration. 
		Forward & Backward computation for other nodes. 

/src/caffe/net.cpp:
		Change some parameters. 

/src/caffe/solver.cpp:
		Add functions for updating the neural network. 

/src/caffe/layers/conv_layer.cpp:
		Modify the forward & backward functions by applying the MPI library. 

/include/caffe/vision_layers.hpp:
		Change the data structure. 

There are also lots of minor changes in other files.

#Add tons of "std::cout" or "LOG(INFO)" in order to understand how does the caffe work
