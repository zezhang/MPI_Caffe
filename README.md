#Major Changes in the following files:


/tools/caffe.cpp:
		Initialize the MPI environment
		Update the nerual network after each iteration
		Forward & Backward computation for other nodes

/src/caffe/net.cpp:
		Change some parameters

/src/caffe/solver.cpp:
		Add functions for updating the nerual network

/src/caffe/layers/conv_layer.cpp:
		Change the forward & backward computation by applying MPI functions

/include/caffe/vision_layers.hpp:
		Change the data structure to suit MPI funcion

#There are also minor changes in others files

#Add tons of "std::cout" or "LOGO(INFO)" in order to understand how does the caffe work
