#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "mpi.h"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //std::cout << "This layer is Convolution Layer!!" << std::endl;
  int num_node = 4;
  int image_per_node = this->num_ / num_node;
  int reminder_image = this->num_ % num_node;
  //LOG(INFO) << "image_per_node is " << image_per_node;
  //LOG(INFO) << "reminder_image is" << reminder_image;

  const Dtype* weight = this->blobs_[0]->cpu_data();
  //std::cout << "cpu data is at " << weight 
            //<< "         transfer data is at " << this->blobs_[0]->data().get()->mutable_cpu_data()
            //<< std::endl;
  LOG(INFO) << "  FORWARD bottom.size is " << bottom.size() << std::endl;
  LOG(INFO) << "  FORWARD this->num_ is " << this->num_ << std::endl;
  for (int i = 0; i < bottom.size(); ++i) {
    //std::cout << i << std::endl;
    Dtype* bottom_data = const_cast<Dtype*>(bottom[i]->cpu_data());
    //LOG(INFO) << "      Ze---FORWARD---Bottom_data's sum is " << bottom[i]->asum_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    //std::cout << "this->num_ is " << this->num_ << std::endl;

    if (this->num_ == 100) {
      //send all the data out

      /*std::cout << "Send Data: " << std::endl;
      for (int m = 0 ; m < bottom[i]->offset(1); ++m) {
        std::cout << bottom_data[m + 784] << std::endl;
      }*/

      //Dtype* bottom_data_ptr = (Dtype *)malloc(sizeof(Dtype) * bottom[i]->offset(1));
      //LOG(INFO) << "  Send out sieze is  " << bottom[i]->offset(1);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Scatter(bottom_data, image_per_node * bottom[i]->offset(1), MPI_FLOAT, bottom_data,
                  image_per_node * bottom[i]->offset(1), MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "Forward, calculate the Convolution"  << std::endl;
        //std::cout << "bottom[i]->offset(n) is " << bottom[i]->offset(n) << std::endl;
      for (int n = 0; n < image_per_node; ++n) {
        //std::cout << "Forward, calculate the Convolution"  << std::endl;
        //std::cout << "bottom[i]->offset(n) is " << bottom[i]->offset(n) << std::endl;
        this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
            top_data + top[i]->offset(n));
      ///////original code///////////
        //this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
            //top_data + top[i]->offset(n));
      //for (int n = 0; n < this->num_; ++n) {
        if (this->bias_term_) {
          //std::cout << "Forward, calculate the bias"  << std::endl;
          //std::cout << "top[i]->offset(n) is " << top[i]->offset(n) << std::endl;
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Gather(top_data, image_per_node * top[i]->offset(1), MPI_FLOAT, top_data,
                 image_per_node * top[i]->offset(1), MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      if(reminder_image) {
        LOG(INFO) << "Has remainder! Needs to do more!";
        for (int n = (this->num_ - reminder_image); n < this->num_; ++n) {
          //std::cout << "Forward, calculate the Convolution"  << std::endl;
          //std::cout << "bottom[i]->offset(n) is " << bottom[i]->offset(n) << std::endl;
          this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
              top_data + top[i]->offset(n));
          if (this->bias_term_) {
            //std::cout << "Forward, calculate the bias"  << std::endl;
            //std::cout << "top[i]->offset(n) is " << top[i]->offset(n) << std::endl;
            const Dtype* bias = this->blobs_[1]->cpu_data();
            this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
          }
        }
      }
      /*std::cout << "node 1 reveived!!!" << std::endl;
      for(int m = 0; m < 11520; ++m) {
        std::cout << top_data[m + top[i]->offset(1)] << std::endl;
      }*/
      //MPI_Barrier(MPI_COMM_WORLD);
      //}
    }
    else {
      //do the normal thing
      /*std::cout << "bottom[i]->offset(0) is " << bottom[i]->offset(0) << std::endl;
      std::cout << "bottom[i]->offset(1) is " << bottom[i]->offset(1) << std::endl;
      std::cout << "bottom[i]->offset(2) is " << bottom[i]->offset(2) << std::endl;
      std::cout << "bottom[i]->offset(3) is " << bottom[i]->offset(3) << std::endl;
      std::cout << "top[i]->offset(0) is " << top[i]->offset(0) << std::endl;
      std::cout << "top[i]->offset(1) is " << top[i]->offset(1) << std::endl;
      std::cout << "top[i]->offset(2) is " << top[i]->offset(2) << std::endl;
      std::cout << "top[i]->offset(3) is " << top[i]->offset(3) << std::endl;*/
      for (int n = 0; n < this->num_; ++n) {
        //std::cout << "Forward, calculate the Convolution"  << std::endl;
        //std::cout << "bottom[i]->offset(n) is " << bottom[i]->offset(n) << std::endl;
        this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
            top_data + top[i]->offset(n));
        if (this->bias_term_) {
          //std::cout << "Forward, calculate the bias"  << std::endl;
          //std::cout << "top[i]->offset(n) is " << top[i]->offset(n) << std::endl;
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "Ze:---------------Backward in this function!";
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  //LOG(INFO) << "      Ze---BACKWARD---Weight_diff's sum is " << this->blobs_[0]->asum_diff();
  if (this->param_propagate_down_[0]) {
    //goes in here!
    //LOG(INFO) << "    Ze:---------------Blobs-0!";
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    //goes in here!
    //LOG(INFO) << "    Ze:---------------Blobs-1!";
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }

  int num_node = 4;
  int image_per_node = this->num_ / num_node;
  int reminder_image = this->num_ % num_node;

  //LOG(INFO) << "Ze:---------------top.size() is " << top.size();
  LOG(INFO) << "  BACKWARD: bottom.size() is " << top.size();
  for (int i = 0; i < top.size(); ++i) {
    //LOG(INFO) << "Ze:---------------i is " << i;
    Dtype* top_diff = const_cast<Dtype*>(top[i]->cpu_diff());
    const Dtype* bottom_data = bottom[i]->cpu_data();
    //LOG(INFO) << "      Ze---BACKWARD---Bottom_data's sum is " << bottom[i]->asum_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      LOG(INFO) << "  BACKWARD: first num is " << this->num_;
      //LOG(INFO) << "[1]:---------------has bais term ";
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      //LOG(INFO) << "[1]:---------Before------bias_diff data is " << this->blobs_[1].get()->asum_data() / this->blobs_[1].get()->count();
      //LOG(INFO) << "[1]:---------Before------bias_diff diff is " << this->blobs_[1].get()->asum_diff() / this->blobs_[1].get()->count();
      if(this->num_ == 100) {
        //do the MPI stuff
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Scatter(top_diff, image_per_node * top[i]->offset(1), MPI_FLOAT, top_diff,
                    image_per_node * top[i]->offset(1), MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(top_diff, image_per_node * top[i]->offset(1), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int n = 0; n < image_per_node; ++n) {                                         
          //LOG(INFO) << "[1]:---------------n is" << n;                               
          this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));            
        }
        if(reminder_image) {
          LOG(INFO) << "Has remainder! Needs to do more!";
          for (int n = (this->num_ - reminder_image); n < this->num_; ++n) {
            this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n)); 
          }
        }
        Dtype* temp_result = new Dtype [num_node * this->num_output_];
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(temp_result, this->num_output_, MPI_FLOAT, temp_result, this->num_output_, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "GROUND_TRUTH: ";
        for(int n = 1; n < num_node; ++n){
          //std::cout << *(bias_diff + n) << " ";
          caffe_axpy<Dtype>(this->num_output_, 1, temp_result + (n * this->num_output_), bias_diff);
          //*(bias_diff+n) = *(bias_diff+n) + *(bias_diff+ 1 * this->num_output_ +n) + *(bias_diff+ 2 * this->num_output_ +n)
                                          //+ *(bias_diff+ 3 * this->num_output_ +n);
        }
        delete[] temp_result;
      }
      else {
        for (int n = 0; n < this->num_; ++n) {                                         
          //LOG(INFO) << "[1]:---------------n is" << n;                               
          this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));            
        }
      }
///////////////////////////ORIGINAL CODE/////////////////////////////////////////////////
//      for (int n = 0; n < this->num_; ++n) {                                         //
//        //LOG(INFO) << "[1]:---------------n is" << n;                               //
//        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));            //
//      }                                                                              //
///////////////////////////ORIGINAL CODE/////////////////////////////////////////////////

      //LOG(INFO) << "[1]:---------After------bias_diff data is " << this->blobs_[1].get()->asum_data() / this->blobs_[1].get()->count();
      //LOG(INFO) << "[1]:---------After------bias_diff diff is " << this->blobs_[1].get()->asum_diff() / this->blobs_[1].get()->count();
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if(this->num_ ==100) {
        ////////////////////////MPI/////////////////
        // gradient w.r.t. weight. Note that we will accumulate diffs. 
        LOG(INFO) << "  BACKWARD: second num is " << this->num_;
        if (this->param_propagate_down_[0]) {

          for (int n = 0; n < image_per_node; ++n) {                                                                    
            this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
                top_diff + top[i]->offset(n), weight_diff);        
          }
          if(reminder_image) {
            LOG(INFO) << "Has remainder! Needs to do more!";
            for (int n = (this->num_ - reminder_image); n < this->num_; ++n) {
              this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
                top_diff + top[i]->offset(n), weight_diff);
            }
          }
          int backward_recv_size = (this->conv_out_channels_ / this->group_) *
                                   (this->kernel_dim_ / this->group_);
          Dtype* temp_result_2 = new Dtype [num_node * backward_recv_size];
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Gather(temp_result_2, backward_recv_size, MPI_FLOAT,
                     temp_result_2, backward_recv_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
          for(int n = 1; n < num_node; ++n){
            caffe_axpy<Dtype>(backward_recv_size, 1, temp_result_2 + (n * backward_recv_size), weight_diff);
          }
          delete[] temp_result_2;
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          //std::cout << "Final Check Point 1" << std::endl;
          for (int n = 0; n < image_per_node; ++n) {
            this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
          }
          //std::cout << "Final Check Point 2" << std::endl;
          if(reminder_image) {
            LOG(INFO) << "Has remainder! Needs to do more!";
            for (int n = (this->num_ - reminder_image); n < this->num_; ++n) {
              this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
                bottom_diff + bottom[i]->offset(n));
            }
          }
          //std::cout << "Final Check Point 3" << std::endl;
          //int layer_3_only = (this->kernel_dim_ / this->group_) * this->conv_out_spatial_dim_ * image_per_node;
          //std::cout << "root has value " << layer_3_only << std::endl;
          //Dtype* temp_result_3 = new Dtype [layer_3_only];
          int layer_3_only = image_per_node * bottom[i]->offset(1);
          MPI_Barrier(MPI_COMM_WORLD);
          MPI_Gather(bottom_diff, layer_3_only, MPI_FLOAT, bottom_diff, layer_3_only, MPI_FLOAT, 0, MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
          //std::cout << "ROOT is successful!"<< std::endl;
        }
      }
      else {
        //LOG(INFO) << "[2]:---------------has weight term ";
        //LOG(INFO) << "[2]:---------------num_ is" << this->num_;
        for (int n = 0; n < this->num_; ++n) {
          //LOG(INFO) << "[2]:---------------n is" << n;
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            //LOG(INFO) << "[2]:---------------first";
            this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
                top_diff + top[i]->offset(n), weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            //LOG(INFO) << "[2]:---------------second";
            this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
                bottom_diff + bottom[i]->offset(n));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
