#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>
#include <string>

#include "boost/algorithm/string.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  //std::cout << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));
  
  //record the convolution layer
  vector<int> conv_layer_record;
  for(int i = 0; i < solver->net()->layers().size(); ++i) {
    if (solver->net()->layers()[i]->layer_param().type() == "Convolution") {
      conv_layer_record.push_back(i);
    }
  }
  for(int i = 0; i < conv_layer_record.size(); ++i) {
    LOG(INFO) << conv_layer_record[i];
  }





//////////////////////////////initialize MPI stuff here//////////////////////////////
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  if (world_rank != 0) {
    //things that others nodes need to do
    int num_node = 4;
    int image_per_node = solver->net()->bottom_vecs()[1][0]->num() / num_node;
    //std::cout << image_per_node << std::endl;
    //int reminder_image = this->num_ % num_node;
//////synchronize the parameters inside of each layer after the network initialization////// 
    for (int m = 0; m < conv_layer_record.size(); ++m) {
      int i = conv_layer_record[m];
      for (int j = 0; j < solver->net()->layers()[i]->blobs().size(); ++j) {
        Blob<float>& blob = *solver->net()->layers()[i]->blobs()[j];
        MPI_Barrier(MPI_COMM_WORLD);
        //receive the dada here
        MPI_Bcast(blob.data().get()->mutable_cpu_data(), blob.count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //check the data
        //float data_abs_val_mean = blob.asum_data() / blob.count();
        //std::cout << "[Initialization] -" << world_rank 
          //<< "- Layer " << solver->net()->layer_names()[i] << " data: " << data_abs_val_mean
          //<< std::endl;
      }
    }

    for (int count = 0; count <= solver_param.max_iter(); ++count) {
      //std::cout << "Current count from node-" << world_rank << " is " << count << std::endl;
////////////////////////the training procedure/////////////////////////////////////
      
/////////////////FORWARD IN GENERAL//////////////////////////////
      for (int m = 0; m < conv_layer_record.size(); ++m) {
        int i = conv_layer_record[m];
        for(int j = 0; j < solver->net()->bottom_vecs()[i].size(); ++j)
        {
          float* bottom_data = const_cast<float*> (solver->net()->bottom_vecs()[i][j]->cpu_data());
          float* top_data = solver->net()->top_vecs()[i][j]->mutable_cpu_data();
          const float* weight = solver->net()->layers()[i]->blobs()[0]->cpu_data();
          int num_received = solver->net()->bottom_vecs()[i][j]->offset(1);
          int num_sended = solver->net()->top_vecs()[i][j]->offset(1);
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Scatter(bottom_data, image_per_node * num_received, MPI_FLOAT, bottom_data,
                      image_per_node * num_received, MPI_FLOAT, 0, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
    ////////////////////do the forward processes///////////////////////
          class caffe::ConvolutionLayer<float>* conv_ptr = dynamic_cast<caffe::ConvolutionLayer<float> *>
                                                                       (solver->net()->layers()[i].get());
          for (int n = 0; n < image_per_node; ++n) {
            conv_ptr->forward_cpu_gemm(bottom_data + solver->net()->bottom_vecs()[i][j]->offset(n), weight,
                                       top_data + solver->net()->top_vecs()[i][j]->offset(n));
            if (solver->net()->layers()[i]->layer_param().convolution_param().bias_term()) {
              const float* bias = solver->net()->layers()[i]->blobs()[1]->cpu_data();
              conv_ptr->forward_cpu_bias(top_data + solver->net()->top_vecs()[i][j]->offset(n), bias);
            }
          }
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Gather(top_data, image_per_node * num_sended, MPI_FLOAT, top_data,
                     image_per_node * num_sended, MPI_FLOAT, 0, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
        }
      }


      








/*      class caffe::ConvolutionLayer<float>* conv_ptr = dynamic_cast<caffe::ConvolutionLayer<float> *>
                                                                       (solver->net()->layers()[1].get());
      class caffe::ConvolutionLayer<float>* conv_ptr_2 = dynamic_cast<caffe::ConvolutionLayer<float> *>
                                                                       (solver->net()->layers()[3].get());
      
      const float* weight = solver->net()->layers()[1]->blobs()[0]->cpu_data();
      const float* weight_2 = solver->net()->layers()[3]->blobs()[0]->cpu_data();
      int num_received = solver->net()->bottom_vecs()[1][0]->offset(1);
      int num_sended = solver->net()->top_vecs()[1][0]->offset(1);
      int num_received_2 = solver->net()->bottom_vecs()[3][0]->offset(1);
      int num_sended_2 = solver->net()->top_vecs()[3][0]->offset(1);
      float* bottom_data = const_cast<float*> (solver->net()->bottom_vecs()[1][0]->cpu_data());
      float* top_data = solver->net()->top_vecs()[1][0]->mutable_cpu_data();
      float* bottom_data_2 = const_cast<float*> (solver->net()->bottom_vecs()[3][0]->cpu_data());
      float* top_data_2 = solver->net()->top_vecs()[3][0]->mutable_cpu_data();*/







      if(count < solver_param.max_iter()) {    
///////////////////BACKWARD PROPOGATION IN GENERAL/////////////////
        for(int m = conv_layer_record.size() - 1; m >= 0; --m) {
          int i = conv_layer_record[m];
          class caffe::ConvolutionLayer<float>* conv_ptr = dynamic_cast<caffe::ConvolutionLayer<float> *>
                                                                       (solver->net()->layers()[i].get());
          const float* weight = solver->net()->layers()[i]->blobs()[0]->cpu_data();
          float* weight_diff = solver->net()->layers()[i]->blobs()[0]->mutable_cpu_diff(); 
          if (solver->net()->layers()[i]->param_propagate_down(0)) {
            caffe::caffe_set(solver->net()->layers()[i]->blobs()[0]->count(), float(0), weight_diff);
          }
          if (conv_ptr->bias_term_ && solver->net()->layers()[i]->param_propagate_down(1)) {
            caffe::caffe_set(solver->net()->layers()[i]->blobs()[1]->count(), float(0),
                            solver->net()->layers()[i]->blobs()[1]->mutable_cpu_diff());
          }
          for(int j = 0; j < solver->net()->top_vecs()[i].size(); ++j) {
            float* top_diff = const_cast<float*>(solver->net()->top_vecs()[i][j]->cpu_diff());
            float* bottom_data = const_cast<float*> (solver->net()->bottom_vecs()[i][j]->cpu_data());
            float* bottom_diff = solver->net()->bottom_vecs()[i][j]->mutable_cpu_diff();
            if (conv_ptr->bias_term_ && solver->net()->layers()[i]->param_propagate_down(1)) {
              float* bias_diff = solver->net()->layers()[i]->blobs()[1]->mutable_cpu_diff();
              int num_sended = solver->net()->top_vecs()[i][j]->offset(1);
              MPI_Barrier(MPI_COMM_WORLD);
              MPI_Scatter(top_diff, image_per_node * num_sended, MPI_FLOAT, top_diff,
                          image_per_node * num_sended, MPI_FLOAT, 0, MPI_COMM_WORLD);
              MPI_Barrier(MPI_COMM_WORLD);
              for (int n = 0; n < image_per_node; ++n) {                                                                       
                conv_ptr->backward_cpu_bias(bias_diff, top_diff + solver->net()->top_vecs()[i][j]->offset(n));            
              }
              MPI_Barrier(MPI_COMM_WORLD);
              MPI_Gather(bias_diff, conv_ptr->num_output_, MPI_FLOAT, bias_diff, conv_ptr->num_output_, MPI_FLOAT, 0, MPI_COMM_WORLD);
              MPI_Barrier(MPI_COMM_WORLD);
            }
            if (solver->net()->layers()[i]->param_propagate_down(0) ||
              solver->net()->bottom_need_backward()[i][j]) {
              if (solver->net()->layers()[i]->param_propagate_down(0)) {
                for (int n = 0; n < image_per_node; ++n) {                                                                    
                  conv_ptr->weight_cpu_gemm(bottom_data + solver->net()->bottom_vecs()[i][j]->offset(n),
                                            top_diff + solver->net()->top_vecs()[i][j]->offset(n), weight_diff);        
                }
                int backward_send_size = (conv_ptr->conv_out_channels_ / conv_ptr->group_) *
                                         (conv_ptr->kernel_dim_ / conv_ptr->group_);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Gather(weight_diff, backward_send_size, MPI_FLOAT,
                           weight_diff, backward_send_size, MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);

              }
              if (solver->net()->bottom_need_backward()[i][j]) {
                for (int n = 0; n < image_per_node; ++n) {                                                                    
                  conv_ptr->backward_cpu_gemm(top_diff + solver->net()->top_vecs()[i][j]->offset(n), weight,
                                              bottom_diff + solver->net()->bottom_vecs()[i][j]->offset(n));      
                }
                int num_send_second = image_per_node * solver->net()->bottom_vecs()[i][j]->offset(1);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Gather(bottom_diff, num_send_second, MPI_FLOAT,
                           bottom_diff, num_send_second, MPI_FLOAT,
                           0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
              }
            }
          }
        }
























































        
      /*/////////////////////Layer 3//////////////////
        //std::cout << "Check Point 1 from node-" << world_rank << std::endl;
        float* top_diff_2 = const_cast<float*>(solver->net()->top_vecs()[3][0]->cpu_diff());
        float* bias_diff_2 = solver->net()->layers()[3]->blobs()[1]->mutable_cpu_diff();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter(top_diff_2, image_per_node * num_sended_2, MPI_FLOAT, top_diff_2,
                    image_per_node * num_sended_2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(top_diff_2, image_per_node * num_sended_2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int n = 0; n < image_per_node; ++n) {                                         
          //LOG(INFO) << "[1]:---------------n is" << n;                               
          conv_ptr_2->backward_cpu_bias(bias_diff_2, top_diff_2 + solver->net()->top_vecs()[3][0]->offset(n));            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(bias_diff_2, conv_ptr_2->num_output_, MPI_FLOAT, bias_diff_2, conv_ptr_2->num_output_, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i < conv_ptr_2->num_output_; ++i) {
          *(bias_diff_2 + i) = 0;
        }
        float* weight_diff_2 = solver->net()->layers()[3]->blobs()[0]->mutable_cpu_diff();
        for (int n = 0; n < image_per_node; ++n) {                                                                    
          conv_ptr_2->weight_cpu_gemm(bottom_data_2 + solver->net()->bottom_vecs()[3][0]->offset(n),
                                top_diff_2 + solver->net()->top_vecs()[3][0]->offset(n), weight_diff_2);        
        }
        int backward_send_size_2 = (conv_ptr_2->conv_out_channels_ / conv_ptr_2->group_) *
                                   (conv_ptr_2->kernel_dim_ / conv_ptr_2->group_);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(weight_diff_2, backward_send_size_2, MPI_FLOAT,
                   weight_diff_2, backward_send_size_2, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i < backward_send_size_2; ++i) {
          *(weight_diff_2 + i) = 0;
        }
        //std::cout << "NODES! - 1" << std::endl;
        float* bottom_diff_2 = solver->net()->bottom_vecs()[3][0]->mutable_cpu_diff();
        for (int n = 0; n < image_per_node; ++n) {                                                                    
          conv_ptr_2->backward_cpu_gemm(top_diff_2 + solver->net()->top_vecs()[3][0]->offset(n), weight_2,
                                     bottom_diff_2 + solver->net()->bottom_vecs()[3][0]->offset(n));      
        }
        //std::cout << "NODES! - 2" << std::endl;
        //int layer_3_only = (conv_ptr_2->kernel_dim_ / conv_ptr_2->group_) * conv_ptr_2->conv_out_spatial_dim_ * image_per_node;
        //std::cout << "nodes has value " << layer_3_only << ", ground truth is "<< solver->net()->bottom_vecs()[3][0]->offset(1) << std::endl;
        int layer_3_only = image_per_node * solver->net()->bottom_vecs()[3][0]->offset(1);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(bottom_diff_2, layer_3_only, MPI_FLOAT,
                   bottom_diff_2, layer_3_only, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "NODES is successful!"<< std::endl;

        //std::cout << "NODES! - 3" << std::endl;
        /////////////////////Layer 1//////////////////
        //std::cout << "Check Point 2 from node-" << world_rank << std::endl;
        float* top_diff = const_cast<float*>(solver->net()->top_vecs()[1][0]->cpu_diff());
        float* bias_diff = solver->net()->layers()[1]->blobs()[1]->mutable_cpu_diff();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter(top_diff, image_per_node * num_sended, MPI_FLOAT, top_diff,
                    image_per_node * num_sended, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(top_diff, image_per_node * num_sended, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int n = 0; n < image_per_node; ++n) {                                         
          //LOG(INFO) << "[1]:---------------n is" << n;                               
          conv_ptr->backward_cpu_bias(bias_diff, top_diff + solver->net()->top_vecs()[1][0]->offset(n));            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(bias_diff, conv_ptr->num_output_, MPI_FLOAT, bias_diff, conv_ptr->num_output_, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i < conv_ptr->num_output_; ++i) {
          *(bias_diff + i) = 0;
        }
        float* weight_diff = solver->net()->layers()[1]->blobs()[0]->mutable_cpu_diff();
        for (int n = 0; n < image_per_node; ++n) {                                                                    
          conv_ptr->weight_cpu_gemm(bottom_data + solver->net()->bottom_vecs()[1][0]->offset(n),
                                top_diff + solver->net()->top_vecs()[1][0]->offset(n), weight_diff);        
        }
        int backward_send_size = (conv_ptr->conv_out_channels_ / conv_ptr->group_) *
                                   (conv_ptr->kernel_dim_ / conv_ptr->group_);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(weight_diff, backward_send_size, MPI_FLOAT,
                   weight_diff, backward_send_size, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i = 0; i < backward_send_size; ++i) {
          *(weight_diff + i) = 0;
        }

        //std::cout << "Check Point 3 from node-" << world_rank << std::endl;*/

//////synchronize the parameters inside of each layer after the network update//////
        for (int m = 0; m < conv_layer_record.size(); ++m) {
          int i = conv_layer_record[m];
          for (int j = 0; j < solver->net()->layers()[i]->blobs().size(); ++j) {
            Blob<float>& blob = *solver->net()->layers()[i]->blobs()[j];
            MPI_Barrier(MPI_COMM_WORLD);
            //receive the data here
            MPI_Bcast(blob.data().get()->mutable_cpu_data(), blob.count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            //check the data
            //float data_abs_val_mean = blob.asum_data() / blob.count();
            //std::cout << "[Update] -" << world_rank 
              //<< "- Layer " << solver->net()->layer_names()[i] << " data: " << data_abs_val_mean
              //<< std::endl;
          }
        }
      }
    }
  }

  else {
    //things that the root needs to do

//////Broadcast the parameters inside of each layer after the network initialization//////
    for (int m = 0; m < conv_layer_record.size(); ++m) {
      int i = conv_layer_record[m];
      for (int j = 0; j < solver->net()->layers()[i]->blobs().size(); ++j) {
        Blob<float>& blob = *solver->net()->layers()[i]->blobs()[j];
        MPI_Barrier(MPI_COMM_WORLD);
        //send the data here
        MPI_Bcast(blob.data().get()->mutable_cpu_data(), blob.count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //check the data
        //float data_abs_val_mean = blob.asum_data() / blob.count();
        //std::cout << "[Initialization] -" << world_rank 
          //<< "- Layer " << solver->net()->layer_names()[i] << " data: " << data_abs_val_mean
          //<< std::endl;
      }
    }

//////Do the training//////

    if (FLAGS_snapshot.size()) {
      LOG(INFO) << "Resuming from " << FLAGS_snapshot;
      solver->Solve(FLAGS_snapshot);
    } else if (FLAGS_weights.size()) {
      CopyLayers(&*solver, FLAGS_weights);
      solver->Solve();
    } else {
      solver->Solve();
    }
    LOG(INFO) << "Optimization Done.";
  }

/////////////////////MPI Done///////////////////////

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();

  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight =
        caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      // Although Reshape should be essentially free, we include it here
      // so that we will notice Reshape performance bugs.
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
