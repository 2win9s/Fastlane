/*
# Using Re:Zero(Bachlechner et Al), in order to ensure gradients propagate
# https://arxiv.org/abs/2003.04887
# As Re:Zero stars off with the identity function only we use Xavier initilisation(Glorot et Al)
# https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# ReLU(Agarap) actiation function is also employed 
# https://arxiv.org/abs/1803.08375
# include <variant>
*/
#pragma once

#include<vector>
#include<set>
#include<cmath>
#include<iostream>
#include<cstdlib>
#include<string>
#include<random>
#include<fstream>
#include<limits>
#include<iomanip>
#include<algorithm>
#include<filesystem>
#include<variant>
#include<omp.h>
#include<array>

const float max_val = 10000000000;
const float n_zero = 0.0001; //try and stop things from freaking out 

std::uniform_real_distribution<float> zero_to_one(0.0,1.0);

// keyboard mashed the name for these because standard names like r_dev will be defined elsewhere and clash
std::random_device rsksksksksks;                          
std::mt19937 ttttt(rsksksksksks());

inline float sign_of(float x){
    return((std::signbit(x) * -2) + 1);
}

//based on "A handy approximation for the error function and its inverse" by Sergei Winitzki.,
//https://www.academia.edu/9730974/A_handy_approximation_for_the_error_function_and_its_inverse
float approx_erfinv(float x){
    static const double a = 0.6802721088435374;         //this is 1/1.47
    static const float pita_inv = 0.4330746750799873;  //this is 1/1.47 times 1/pi times 2
    float w = 1-x*x;
    w = (w == 1) ? 0.999999:w;
    w = std::log(w);
    float u = pita_inv+(w*0.5);
    float val = std::sqrt(-u + std::sqrt((u * u) - (a * w)));
    return(val * sign_of(x));
}




/*
# activation functions and their derivatives
*/

inline float relu(float x){
    return ((x>0.0f) ? x:0.0f);
}
inline float drelu(float fx){
    return ((fx>0.0f) ? 1.0f:0.0f);
}

// it may be useful to utilise log space 
// so this function is simply ln(relu(x)+1)
inline float log_relu(float x){
    return std::logf((x>0.0f) ? (x+1.0f):1.0f);
}
inline float dlog_relu(float fx){
    return ((fx>0.0f) ? std::expf(-fx):0.0f);
}

// function to compute cos(x) from sin(x)
// i.e. the derivative of sin(x) activation function
inline float cos_sinx(float sinx){
    return std::sqrtf(1 - (sinx*sinx));
}

bool broken_float(float x){
    if (std::isnan(x)){
        return true;
    }
    else if (std::isinf(x)){
        return true;
    }
    else{
        return false;
    }
}

void soft_max(std::vector<float> &output){
    double denominator = 0;
    std::vector<float> expout(output.size());
    for (int i = 0; i < output.size(); i++)
    {
        expout[i] = std::exp(output[i]);
        if (broken_float(expout[i]))
        {
            expout[i] = max_val;
        }
        denominator += expout[i]; 
    }
    denominator = 1 / denominator; 
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = expout[i] * denominator;
    }
}

int prediction(std::vector<float> &output){
    int guess;
    guess = std::distance(output.begin(),std::max_element(output.begin(),output.end()));
    return guess;
}

void dsoft_max(std::vector<float> &output, std::vector<float> &target, std::vector<float> &loss){
    for (int i = 0; i < output.size(); i++)
    {
        loss[i] = output[i] - target[i];
    }
}

float cross_entrophy(std::vector<float> &target, std::vector<float> & output){
    float loss = 0;
    for (int i = 0; i < output.size(); i++)
    {
        loss -= (target[i] * std::log(output[i]));
    }
    return loss;
}

// struct for holding gradients for individual neurons before update
// neurons will have 1 input 1 output and 2 hidden layers of 8 units
struct neuron_gradients
{
    float bias[16] = {0};
    float alpha[16] = {0};
    float weights[9][7] = {0};
    float padding;
    inline void valclear(){    //wrapper function for memset
        memset(alpha,0,sizeof(alpha));
        memset(bias,0,sizeof(bias));
        memset(weights,0,sizeof(weights));
    }   
    template <typename neuron>
    inline void sgd_with_momentum(neuron &n, float learning_rate, float momentum, neuron_gradients current_grad){    
        #pragma omp for simd
        for (uint8_t i = 0; i < 16; i++)
        {
            bias[i] *= momentum;
            current_grad.bias[i] *= learning_rate;
            bias[i] += current_grad.bias[i];
            n.bias[i] -= bias[i];
        }
        
        #pragma omp for simd
        for (uint8_t i = 0; i < 16; i++)
        {
            alpha[i] *= momentum;
            current_grad.alpha[i] *= learning_rate;
            alpha[i] += current_grad.alpha[i];
            n.alpha[i] -= alpha[i];
        }

        for (uint8_t i = 0; i < 9; i++)
        {
            #pragma omp simd
            for (uint8_t j = 0; j < 7; j++)
            {
                weights[i][j] *= momentum;
                current_grad.weights[i][j] *= learning_rate;
                weights[i][j] += current_grad.weights[i][j];
                n.weights[i][j] -= weights[i][j];
            }
            
        }
        
    }
};


/*
# each neuron is basically a simple 2 hidden layer nn 
# Re:Zero is used to allow the gradients to pass through 
# 1 input 7 hidden units in the 2 layers then 1 output
# each struct implements a neuron with a different activation function
*/

struct neuron_unit
{
    float units[16] = {0};     // 0 will be input and 18 will be output
    float bias[16]  = {0};   
    float alpha[16] = {0};     //from Re:Zero is all you need 
    float weights[9][7];
    int isinput = 0;         
    neuron_unit(){
        //  Xavier initialisation
        std::normal_distribution<float> a (0,0.5);            // input to 1st hidden layer
        std::normal_distribution<float> b (0,0.377964473);    // input to 2nd hidden layer
        for (uint8_t i = 0; i < 7; i++)
        {
            weights[0][i] = a(ttttt);
            weights[8][i] = a(ttttt);
        }
        for (uint8_t i = 1; i < 8; i++)
        {
            for (uint8_t j = 0; j < 8; j++)
            {
                weights[i][j] = b(ttttt);
            }
        }
    }
};

struct relu_neuron : neuron_unit
{
    using neuron_unit::neuron_unit;
};


struct sine_neuron : neuron_unit
{
    using neuron_unit::neuron_unit;
};


struct log_relu_neuron : neuron_unit
{
    using neuron_unit::neuron_unit;
};


/*
# template functions for neurons
*/

// wrapper function for relu and drelu
inline float act_func(float x, relu_neuron &){
    return relu(x);
}
inline float dact_func(float fx, relu_neuron &){
    return drelu(fx);
}

// wrapper function for sine and cos(arcsin(x))
inline float act_func(float x, sine_neuron &){
    return std::sinf(x);
}
inline float dact_func(float fx, sine_neuron &){
    return cos_sinx(fx);
}

// wrapper function for log_relu and dlog_relu
inline float act_func(float x, log_relu_neuron &){
    return log_relu(x);
}

inline float dact_func(float fx, log_relu_neuron &){
    return dlog_relu(fx);
}

template <typename neuron>
inline void valclear(neuron &n){    //wrapper function for memset on units
    memset(n.units,0,sizeof(n.units));
};

template <typename neuron>
inline void forwardpass(neuron &n,float input, std::array<float,16> &pacts){ //pacts here refers to values obtained after applying activation function
    n.units[0] = input;
    n.units[0] += n.bias[0];
    pacts[0] = act_func(n.units[0],n);
    n.units[0] += (pacts[0] * n.alpha[0]);
    #pragma omp simd
    for (uint8_t i = 1; i < 8; i++)
    {
        n.units[i] = n.units[0] * n.weights[0][i-1];
        n.units[i] += n.bias[i];
        pacts[i] = act_func(n.units[i],n);
        n.units[i] += pacts[i] * n.alpha[i];
    }
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        n.units[i] = n.bias[i];        
    }
    for (uint8_t i = 8; i < 15; i++)
    {
        #pragma omp simd
        for (uint8_t j = 0; j < 7; j++)
        {
            n.units[i] += n.units[j+1] * n.weights[i-7][j];
        }    
    }
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        pacts[i] = act_func(n.units[i],n);
        n.units[i] += pacts[i] * n.alpha[i];
    }
    n.units[15] = n.bias[15];
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        n.units[15] += n.units[i] * n.weights[8][i-8];
    }
    pacts[15] = act_func(n.units[15],n);
    n.units[15] += pacts[15] * n.alpha[15];
}

// forwardpass without recording post activations for each unit
template <typename neuron>
inline void forwardpass(neuron &n, float input){
    n.units[0] = input;
    n.units[0] += n.bias[0];
    n.units[0] += (act_func(n.units[0],n) * n.alpha[0]);
    #pragma omp simd
    for (uint8_t i = 1; i < 8; i++)
    {
        n.units[i] = n.units[0] * n.weights[0][i-1];
        n.units[i] += n.bias[i];
        n.units[i] += act_func(n.units[i],n) * n.alpha[i];
    }
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        n.units[i] = n.bias[i];        
    }
    for (uint8_t i = 8; i < 15; i++)
    {
        #pragma omp simd
        for (uint8_t j = 0; j < 7; j++)
        {
            n.units[i] += n.units[j+1] * n.weights[i-7][j];
        }    
    }
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        n.units[i] += act_func(n.units[i],n) * n.alpha[i];
    }
    n.units[15] = n.bias[15];
    #pragma omp simd
    for (uint8_t i = 8; i < 15; i++)
    {
        n.units[15] += n.units[i] * n.weights[8][i-8];
    }
    n.units[15] += act_func(n.units[15],n) * n.alpha[15];
}

// note the units array will be used to store gradients
template <typename neuron>
inline float backprop(neuron &n,float dldz, std::array<float,16> &past_unit, std::array<float,16> &pacts, neuron_gradients &gradients)
{   
    gradients.alpha[15] += dldz * pacts[15];
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    gradients.bias[15] += dldz;
    memset(n.units,0,8*sizeof(float));
    #pragma omp simd collapse(2)
    for(int i = 8 ; i < 15; i++){
        n.units[i] = dldz*n.weights[8][i-8];
        gradients.weights[8][i-8] += dldz*past_unit[i];
        gradients.alpha[i] += n.units[i] * pacts[i];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        gradients.bias[i] += n.units[i];
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += n.units[i] * n.weights[i-7][j];
            gradients.weights[i-7][j] += n.units[i]*past_unit[j+1];
        }
    }
    for (int i = 1; i < 8; i++)
    {
        gradients.alpha[i] += n.units[i] * pacts[i];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        gradients.bias[i] += n.units[i];
        n.units[0] += n.units[i] * n.weights[0][i-1];
    }
    gradients.alpha[0] += n.units[0] * pacts[0];
    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    gradients.bias[0] += n.units[0];
    return n.units[0];
}    



// backprop but return gradient absolute value <=1, unused bool arguement for template overloading
template <typename neuron>
inline float backprop(neuron &n,float dldz, std::array<float,16> &past_unit, std::array<float,16> &pacts, neuron_gradients &gradients, bool &)
{   
    gradients.alpha[15] += dldz * pacts[15];
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    gradients.bias[15] += dldz;
    memset(n.units,0,8*sizeof(float));
    #pragma omp simd collapse(2)
    for(int i = 8 ; i < 15; i++){
        n.units[i] = dldz*n.weights[8][i-8];
        gradients.weights[8][i-8] += dldz*past_unit[i];
        gradients.alpha[i] += n.units[i] * pacts[i];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        gradients.bias[i] += n.units[i];
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += n.units[i] * n.weights[i-7][j];
            gradients.weights[i-7][j] += n.units[i]*past_unit[j+1];
        }
    }
    for (int i = 1; i < 8; i++)
    {
        gradients.alpha[i] += n.units[i] * pacts[i];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        gradients.bias[i] += n.units[i];
        n.units[0] += n.units[i] * n.weights[0][i-1];
    }
    gradients.alpha[0] += n.units[0] * pacts[0];
    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    gradients.bias[0] += n.units[0];
    return ((n_units[0] > 1) ? sign_of(n.units[0]):n_units[0]);
} 


// backprop but we aren't interested in gradients for the neuron and only the gradient passed out
template <typename neuron>
inline float backprop(neuron &n,float dldz, std::array<float,16> &past_unit, std::array<float,16> &pacts){   
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    memset(n.units,0,8*sizeof(float));
    #pragma omp simd collapse(2)
    for(int i = 8 ; i < 15; i++){
        n.units[i] = dldz*n.weights[8][i-8];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += n.units[i] * n.weights[i-7][j];
            gradients.weights[i-7][j] += n.units[i]*past_unit[j+1];
        }
    }
    for (int i = 1; i < 8; i++)
    {
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        n.units[0] += n.units[i] * n.weights[0][i-1];
    }
    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    return n.units[0];
}

// backprop but we aren't interested in gradients for the neuron and only the gradient passed out
template <typename neuron>
inline float backprop(neuron &n,float dldz, std::array<float,16> &past_unit, std::array<float,16> &pacts,bool &){   
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    memset(n.units,0,8*sizeof(float));
    #pragma omp simd collapse(2)
    for(int i = 8 ; i < 15; i++){
        n.units[i] = dldz*n.weights[8][i-8];
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += n.units[i] * n.weights[i-7][j];
            gradients.weights[i-7][j] += n.units[i]*past_unit[j+1];
        }
    }
    for (int i = 1; i < 8; i++)
    {
        n.units[i] = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        n.units[0] += n.units[i] * n.weights[0][i-1];
    }
    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    return ((n_units[0] > 1) ? sign_of(n.units[0]):n_units[0]);
}



// pdf is f(x) = m(a^2*e^(a^2)) where a(x) = 10(x-0.5), m is a constant approx =11.28379...., in the interval 0 < x < 1
// has a shape with 2 humps around 0.5
float custom_dist(){
    float x = zero_to_one(ttttt);
    x = 1 - 2*x;
    x = 0.1 * (5 - approx_erfinv(x));
    return x;
}

struct neural_net_record{
    std::vector<std::array<float,16>> values;
    void valclear(){
        for (int i = 0; i < values.size(); i++)
        {
            #pragma omp simd
            for (int j = 0; j < 16; j++)
            {
                values[i][j] = 0;
            }
            
        }
    }
    neural_net_record(int size)
    : values(size)
    {
        valclear();
    }
};


/*starting off simple with homogenous fastlane networks*/
struct relu_neural_network{
    struct index_value_pair
    {
        int index;
        float value;
        index_value_pair(int x, float y){
            index = x;
            value = y;
        }
    };
    std::vector<relu_neuron> relu_net;
    //std::vector<log_relu_neuron> log_relu_net;
    //std::vector<sine_neuron> sine_neuron_net;
    std::vector<std::vector<index_value_pair>> weights;// to improve performance consider implementing a interface to a 1d vector instead
    struct network_gradient
    {
        std::vector<neuron_gradients> net_grads;
        std::vector<std::vector<float>> weight_gradients;
        network_gradient(relu_neural_network & NN)
        :net_grads(NN.relu_net.size(),neuron_gradients())
        ,weight_gradients(NN.relu_net.size())
        {
            for (int i = 0; i < NN.relu_net.size(); i++)
            {
                weight_gradients[i].resize(NN.weights[i].size());
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = 0;
                }
            }
            
        }
        void sync(relu_neural_network & NN){
            for (int i = 0; i < NN.relu_net.size(); i++)
            {
                weight_gradients[i].resize(NN.weights[i].size());
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = 0;
                }
            }
        }

        void valclear(){
            for (int i = 0; i < net_grads.size(); i++)
            {
                net_grads[i].valclear();
            }
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = 0;
                }
                
            }
            
        }

        //neuron &n, float learning_rate, float momentum, neuron_gradients current_grad
        inline void sgd_with_momentum(relu_neural_network &n, float learning_rate, float momentum, network_gradient &current_gradient){
            for (int i = 0; i < n.relu_net.size(); i++)
            {
                net_grads[i].sgd_with_momentum(n.relu_net[i],learning_rate,momentum,current_gradient.net_grads[i]);
            }

            for (int i = 0; i < weight_gradients.size(); i++)
            {
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] *= momentum;
                    current_gradient.weight_gradients[i][j] *= learning_rate;
                    weight_gradients[i][j] += current_gradient.weight_gradients[i][j];
                    n.weights[i][j].value -= weight_gradients[i][j];
                }
                
            }
            
        }
    };

    
    std::vector<int> input_index;           //indexing recording input neurons
    std::vector<int> output_index;          //indexing recording output neurons
    std::vector<std::vector<int>> layermap;
    std::vector<std::vector<bool>> dependency; //reduces time needed to update layermap, warning for very VERY large neural nets will eat up memory
    
    // for creating/updating the layermap
    void layermap_sync()
    {
        layermap.clear();
        dependency.reserve(relu_net.size());
        dependency.resize(relu_net.size());
        //std::vector<bool> index_label(neural_net.size(),true);          
        //this is to help label the neurons (not included yet in layermap = true)
        bool* index_label = new bool[relu_net.size()]; 
        std::vector<std::vector<int>> input_tree(relu_net.size());
        for (int i = 0; i < relu_net.size(); i++)
        {
            index_label[i] = true;
        }
        for (int i = 0; i < dependency.size(); i++)
        {
            dependency[i].resize(relu_net.size(),false);
            std::fill(dependency[i].begin(),dependency[i].end(),false);
            for (int j = 0; j < weights[i].size(); j++)
            {
                if (weights[i][j].index > i)
                {
                    continue;
                }
                dependency[i][weights[i][j].index] = true;  //set union may be faster test for use case
            }
        }
        std::vector<int> layermap_layer_candidate;
        int initial_neuron = 0;                                         //the neuron to be included into layermap with highest priority at beginning is at index 0, order of neuron index is order of firing
        int mapped_n = 0;
        while(true)                                 
        {
            if (relu_net.size() == 0)
            {
                std::cout<<"ERROR, neural net with 0 neurons"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            layermap_layer_candidate.clear();
            layermap_layer_candidate.reserve(relu_net.size() - mapped_n);
            for (int i = initial_neuron; i < relu_net.size(); i++)
            {   
                if(index_label[i])
                {
                    bool independent = true;
                    for (int j = initial_neuron; j < relu_net.size(); j++)
                    {
                        bool tf = dependency[i][j];                    
                        if (tf && index_label[j])
                        {
                            independent = false;
                            break;
                        }
                    }
                    if (independent)
                    {
                        layermap_layer_candidate.push_back(i);
                    }   
                }
            }
            if (layermap_layer_candidate.size() == 0)
            {
                for (int i = 0; i < relu_net.size(); i++)
                {
                    if (index_label[i])
                    {
                        /*
                        for (int j = 0; j < neural_net[i].weights.size(); j++)
                        {
                            std::cout<<neural_net[i].weights[j].index<<"\n";
                        }*/
                        std::cout<<std::endl;
                        std::cout<<"something went wrong in layermap function, check weights of neuron "<<i<<std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
                
            }
            mapped_n += layermap_layer_candidate.size();
            for (int i = 0; i < layermap_layer_candidate.size(); i++)
            {
                index_label[layermap_layer_candidate[i]] = false;
            }
            layermap.emplace_back(layermap_layer_candidate);            //adding a new layer
            for (int i = initial_neuron; i < relu_net.size(); i++)
            {
                if(index_label[i])
                {
                    initial_neuron = i;
                    break;
                }
                if (i == (relu_net.size() - 1))                       //if all neurons are now in layermap the function has completed
                {
                    delete[] index_label;
                    return;
                }       
            }
        }
        delete[] index_label;
    }

    relu_neural_network(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, float connection_density, float connection_sd)
    :relu_net(size,relu_neuron())
    ,input_index(input_neurons)
    ,output_index(output_neurons)
    ,weights(size)
    {
        std::normal_distribution<float> connection_dist(connection_density, connection_sd);
        for (int i = 0; i < input_index.size(); i++)
        {
            relu_net[input_index[i]].isinput = 1;
        }
        for (int i = 0; i < size; i++)
        {
            if (relu_net[i].isinput)
            {
                continue;
            }
            float density = connection_dist(ttttt);         //number of connections for each neuron is distributed normally around connection_density with standard deviation connection_sd
            if(density < 0){
                density = 0;
            }
            else if (density > 1)
            {
                density = 1;
            }
            int connect_n =  std::floor(density * (size - 1));          //this is the number of input connections
            std::set<int> input_i = {};
            for (int j = 0; j < connect_n; j++)
            {
                int remaining = size - 2 - weights[i].size();
                int input_neuron_index = std::floor((remaining) * (custom_dist() - 0.5));
                int wrap_around  = (input_neuron_index < 0) * (remaining + input_neuron_index);
                wrap_around = (wrap_around > input_neuron_index) ? wrap_around:input_neuron_index;
                input_neuron_index = (wrap_around % (remaining));
                int pos = 0;
                bool gapped = true;
                for(int k = 0; k < weights[i].size(); k++)
                {
                    if ((input_neuron_index >= i) && gapped)
                    {
                        gapped = false;
                        input_neuron_index++;
                    }
                    if (input_neuron_index >= weights[i][k].index)
                    {
                        input_neuron_index++;
                        pos++;
                        continue;
                    }
                    if ((input_neuron_index >= i) && gapped)
                    {
                        gapped = false;
                        input_neuron_index++;
                        continue;
                    }
                    break;
                }
                if ((input_neuron_index >= i) && gapped)
                {
                    input_neuron_index++;
                }
                weights[i].insert(weights[i].begin() + pos,index_value_pair(input_neuron_index,0.0f));    
            }
        }
        layermap_sync();
        for (int i = 0; i < size; i++)
        {   //not really the correct way to use He initialisation but eh
            float input_layer_size = weights[i].size();
            //Xavier initialisation
            std::uniform_real_distribution<float> weight_init_dist(-std::sqrt(1/input_layer_size),std::sqrt(1/input_layer_size));
            for(int j = 0; j < input_layer_size; j++)
            {
                if (weights[i][j].index > i)
                {
                    weights[i][j].value = 0;
                }
                else{
                    weights[i][j].value = weight_init_dist(ttttt) /*multiplier*/;
                }
            } 
        }
    }

    inline void valclear(){
        for (int i = 0; i < relu_net.size(); i++)
        {
            for (int j = 0; j < 16; j++)
            {
                relu_net[i].units[j] = 0;
            }
            
        }
    }

    inline void record(neural_net_record &post){
        for (int i = 0; i < relu_net.size(); i++)
        {
            for (int j = 0; j < 16; j++)
            {
                post.values[i][j] = relu_net[i].units[j];
            }   
        }
    }

    inline void sforwardpass(std::vector<float> &inputs, neural_net_record &pre, neural_net_record &post){
        valclear();
        for (int i = 0; i < input_index.size(); i++)
        {
            relu_net[input_index[i]].units[0] = inputs[i];
        }
        for (int i = 0; i < layermap.size(); i++)
        {
            for (int j = 0; j < layermap[i].size(); j++)
            {   
                for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                {
                    const int in_indx = weights[layermap[i][j]][l].index;
                    const float in = (in_indx > layermap[i][j]) ? 0.0f : relu_net[in_indx].units[15];
                    //apologies for the naming scheme
                    relu_net[layermap[i][j]].units[0] += weights[layermap[i][j]][l].value * in;
                }
                forwardpass(relu_net[layermap[i][j]],relu_net[layermap[i][j]].units[0],pre.values[layermap[i][j]]);
            }
        }
        record(post);
    }

    inline void sforwardpass(std::vector<float> &inputs){
        valclear();
        for (int i = 0; i < input_index.size(); i++)
        {
            relu_net[input_index[i]].units[0] = inputs[i];
        }
        for (int i = 0; i < layermap.size(); i++)
        {
            for (int j = 0; j < layermap[i].size(); j++)
            {   
                for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                {
                    const int in_indx = weights[layermap[i][j]][l].index;
                    const float in = (in_indx > layermap[i][j]) ? 0.0f : relu_net[in_indx].units[15];
                    //apologies for the naming scheme
                    relu_net[layermap[i][j]].units[0] += weights[layermap[i][j]][l].value * in;
                }
                forwardpass(relu_net[layermap[i][j]],relu_net[layermap[i][j]].units[0]);
            }
        }
    }

    inline void sbackpropagation(std::vector<float> &dloss, neural_net_record &pre, neural_net_record &post, network_gradient &net_grad){
        std::vector<float> gradients(relu_net.size(),0);
        for (int i = 0; i < output_index.size(); i++)
        {
            gradients[output_index[i]] = dloss[i];
        }
        for (int i = layermap.size() - 1; i >= 0; i--)
        {
            for(int j = 0; j < layermap[i].size(); j++){
                float dldz = backprop(relu_net[layermap[i][j]],gradients[layermap[i][j]],post.values[layermap[i][j]],pre.values[layermap[i][j]],net_grad.net_grads[layermap[i][j]]);
                for (int k = 0; k < weights[layermap[i][j]].size(); k++)
                {
                    gradients[weights[layermap[i][j]][k].index] += dldz * weights[layermap[i][j]][k].value;
                    net_grad.weight_gradients[layermap[i][j]][k] += dldz * post.values[weights[layermap[i][j]][k].index][15];
                }
            }
        }
    }

    // saving to and loading from a text file
    inline void save_to_txt(std::string file_name){
        std::ofstream file(file_name,std::fstream::trunc);
        file << "number_of_neurons:"<<"\n";
        file << relu_net.size() << "\n";
        file << "input_index" <<"\n";
        
        for (int i = 0; i < input_index.size(); i++)
        {
            file << input_index[i] << " ";
        }
        file << "\n";
        file << "output_index" << "\n";
        for (int i = 0; i < output_index.size(); i++)
        {
            file << output_index[i] << " ";
        }
        file << "\n";
        file << "number_of_layers" << "\n";
        file << layermap.size() << "\n";
        file << "<layermap>" << "\n";
        for (int i = 0; i < layermap.size(); i++)
        {
            file << "no_of_neurons" << "\n";
            file << layermap[i].size() << "\n";
            for (int j = 0; j < layermap[i].size(); j++)
            {
                file << layermap[i][j] << " ";
            }
            file << "\n";
            if(i != (layermap.size() -1)){
                file << "next_layer" << "\n";
            }
        }
        file << "</layermap>" <<"\n";
        file << "<weights>" << "\n";
        for (int i = 0; i < relu_net.size(); i++)
        {
            file << "no_of_weights" << "\n";
            file << weights[i].size() << "\n";
            for (int j = 0; j < weights[i].size(); j++)
            {
                file << weights[i][j].index << " ";
                file << std::fixed<<std::setprecision(std::numeric_limits<float>::max_digits10)
                << weights[i][j].value << "\n";
            }
            file << "---------" << "\n";
        }
        file << "</weights>" << std::endl;
        for (int i = 0; i < relu_net.size(); i++)
        {
            file << "<neuron>" << "\n";
            file << "<bias>" << "\n";
            for (int j = 0; j < 16; j++)
            {
                file << std::fixed<<std::setprecision(std::numeric_limits<float>::max_digits10)
                << relu_net[i].bias[j] << "\n";
            }
            file << "</bias>" << "\n";

            file << "<alpha>" << "\n";
            for (int j = 0; j < 16; j++)
            {
                file << std::fixed<<std::setprecision(std::numeric_limits<float>::max_digits10)
                << relu_net[i].alpha[j] << "\n";
            }
            file << "</alpha>" << "\n";
            file << "<nweights>" << "\n";
            for (int j = 0; j < 9; j++)
            {
                for (int k = 0; k < 7; k++)
                {
                    file << std::fixed<<std::setprecision(std::numeric_limits<float>::max_digits10)
                    << relu_net[i].weights[j][k] << "\n";
                }
                
            }
            file << "</nweights>" << "\n";
            file << "</neuron>" << "\n";
        }
        file << "<end>" << "\n";
        file.close();
    }
    relu_neural_network(std::string file_name){
        std::string str_data;
        std::vector<int> output_in;
        std::vector<int> input_in;
        std::ifstream file(file_name);
        if (file.good()){;}else{std::cout<<"ERROR "<<file_name<<" does not exist"<<std::endl; std::exit(EXIT_FAILURE);}
        file >> str_data;
        file >> str_data;
        relu_net.resize(std::stoi(str_data),relu_neuron());
        weights.resize(std::stoi(str_data));
        file >> str_data;
        while (true)
        {
            std::string data;
            file >> data;
            if(data == "output_index"){
                break;
            }
            input_index.emplace_back(std::stoi(data));
        }
        while (true)
        {
            std::string data;
            file >> data;
            if(data == "number_of_layers"){
                break;
            }
            output_index.emplace_back(std::stoi(data));
        }
        file >> str_data;
        layermap.resize(std::stoi(str_data),{});
        file >> str_data;
        int itr = 0;
        while(true){
            std::string data;
            file >> data;
            if (data == "</layermap>")
            {
                break;
            }
            else if (data == "no_of_neurons")
            {
                file >> data;
                layermap[itr].reserve(std::stoi(data));
            }
            else if(data == "next_layer")
            {
                itr++;
            }
            else{
                layermap[itr].emplace_back(std::stoi(data));
            }
        }
        file >> str_data;
        itr = 0;
        std::cout<<str_data<<std::endl;
        while (true)
        {
            std::string data;
            file >> data;
            if (data == "</weights>")
            {
                break;
            }
            else if (data == "no_of_weights")
            {
                file >> data;
                weights[itr].reserve(std::stoi(data));
            }
            else if(data == "---------")
            {
                itr++;
            }
            else{
                int index = std::stoi(data);
                file >> data;
                float value= std::stof(data);
                weights[itr].emplace_back(index_value_pair(index,value));
            }
        }
        itr = 0;
        while (true)
        {
            std::string data;
            file >> data;
            if (data == "<neuron>")
            {
                continue;
            }else if (data == "<bias>")
            {
                for (int i = 0; i < 16; i++)
                {
                    file >> data;
                    float b = std::stof(data);
                    relu_net[itr].bias[i] = b;
                }
                file >> data;
            }else if (data == "<alpha>")
            {
                for (int i = 0; i < 16; i++)
                {
                    file >> data;
                    float al = std::stof(data);
                    relu_net[itr].alpha[i] = al;
                }
                file >> data;
            }else if (data == "<nweights>")
            {
                for(int i = 0; i < 9; i++){
                    for (int j = 0; j < 7; j++)
                    {
                        file >> data;
                        float we = std::stof(data);
                        relu_net[itr].weights[i][j] = we;
                    }
                }
            file >> data;
            }
            else if (data == "<end>")
            {
                break;
            }
            else{
                itr++;
            }
        }
        file.close();
        for (int i = 0; i < input_index.size(); i++)
        {
            relu_net[input_index[i]].isinput = true;
        }
        
    }
};

