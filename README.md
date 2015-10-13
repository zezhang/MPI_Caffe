#Major Changes in the following files:


/tools/caffe.cpp:\n
		Initialize the MPI environment\n
		Update the nerual network after each iteration\n
		Forward & Backward computation for other nodes\n

/src/caffe/net.cpp:
		Change some parameters

/src/caffe/solver.cpp:
		Add functions for updating the nerual network

/src/caffe/layers/conv_layer.cpp:
		Change the forward & backward computation by applying MPI functions

/include/caffe/vision_layers.hpp:
		Change the data structure to suit MPI funcion

#There are also minor changes in others files

#Add tons of "std::cout" or "LOG(INFO)" in order to understand how does the caffe work
