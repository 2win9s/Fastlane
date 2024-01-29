/*
# Using Re:Zero(Bachlechner et Al), in order to ensure gradients propagate
# https://arxiv.org/abs/2003.04887
# As Re:Zero stars off with the identity function only we use Xavier initilisation(Glorot et Al)
# https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# ReLU(Agarap) actiation function is also employed 
# https://arxiv.org/abs/1803.08375
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
#include <omp.h>

std::random_device rsksksksksks;                          
std::mt19937 ttttt(rsksksksksks());

/*
# activation functions and their derivatives
*/

inline float relu(float x){
    return ((x>0) ? x:0);
}
inline float drelu(float fx){
    return ((fx>0) ? 1:0);
}

// it may be useful to utilise log space 
// so this function is simply ln(relu(x)+1)
inline float log_relu(float x){
    return std::logf((x>0) ? (x+1):1);
}
inline float dlog_relu(float fx){
    return ((fx>0) ? std::expf(-fx):0);
}

// function to compute cos(x) from sin(x)
// i.e. the derivative of sin(x) activation function
inline float cos_sinx(float sinx){
    return std::sqrtf(1 - (sinx*sinx));
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
    float padding = 0;         //pads it to 112 fp32s (commonly), a multiple of 64 bytes (common cache line size)
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
inline void forwardpass(neuron &n,float input, float (&pacts)[16]){ //pacts here refers to values obtained after applying activation function
    n.units[0] = input;
    n.units[0] += n.bias[0];
    pacts[0] = act_func(n.units[0],n);
    n.units[0] += (pacts[0] * n.alpha[0]);
    #pragma omp simd
    for (uint8_t i = 1; i < 8; i++)
    {
        n.units[i] = n.units[0] * n.weights[0][i];
        n.units[i] += n.bias[i];
        pacts[i] = act_func(n.units[i],n);
        n.units[i] += pacts[i] * n.alpha[i];
    }
    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        n.units[i] = n.bias[i];        
    }
    for (uint8_t i = 7; i < 14; i++)
    {
        #pragma omp simd
        for (uint8_t j = 0; j < 7; j++)
        {
            n.units[i] += n.units[j+1] * n.weights[i-6][j];
        }    
    }
    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        pacts[i] = act_func(n.units[i],n);
        n.units[i] += pacts[i] * n.alpha[i];
    }
    n.units[15] = n.bias[15];
    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        n.units[15] += n.units[i] * n.weights[8][i-7];
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
        n.units[i] = n.units[0] * n.weights[0][i];
        n.units[i] += n.bias[i];
        n.units[i] += act_func(n.units[i],n) * n.alpha[i];
    }


    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        n.units[i] = n.bias[i];        
    }
    for (uint8_t i = 7; i < 14; i++)
    {
        #pragma omp simd
        for (uint8_t j = 0; j < 7; i++)
        {
            n.units[i] += n.units[j+1] * n.weights[i-6][j];
        }    
    }
    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        n.units[i] += act_func(n.units[i],n) * n.alpha[i];
    }
   
    n.units[15] = n.bias[15];
    #pragma omp simd
    for (uint8_t i = 7; i < 14; i++)
    {
        n.units[15] += n.units[i] * n.weights[8][i-7];
    }
    n.units[15] += act_func(n.units[15],n) * n.alpha[15];
}

// note the units array will be used to store gradients
template <typename neuron>
inline float backprop(neuron &n,float dldz, float (&past_unit)[16], float (&pacts)[16], neuron_gradients &gradients)
{   
    gradients.alpha[15] += dldz * pacts[15];
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    gradients.bias[15] += dldz;
    memset(n.units,0,8*sizeof(float));
    #pragma omp simd collapse(2)
    for(int i = 7 ; i < 16; i++){
        n.units[i] = dldz*n.weights[8][i-7];
        gradients.weights[8][i-7] += dldz*past_unit[i];
        float dz = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        gradients.bias[i] += dz;
        gradients.alpha[i] += n.units[i] * pacts[i];
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += dz * n.weights[i-6][j];
            gradients.weights[i-6][j] += dz*past_unit[j+1];
        }
    }
    for (int i = 0; i < 7; i++)
    {
        float dz = n.units[i+1]*(1 + (n.alpha[i+1] * dact_func(pacts[i+1],n))); 
        gradients.bias[i+1] += dz;
        gradients.alpha[i+1] += n.units[i+1] * pacts[i+1];
        n.units[0] += dz * n.weights[0][i];
        gradients.weights[0][i] -= dz*past_unit[0];
    }
    gradients.alpha[0] += n.units[0] * pacts[0];
    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    gradients.bias[0] += n.units[0];
    return n.units[0];
}    
// backprop but we aren't interested in gradients for the neuron and only the gradient passed out
template <typename neuron>
inline float backprop(neuron &n,float dloss, float (&past_unit)[16], float (&pacts)[16]){   
    dldz = dldz * (1 + (n.alpha[15] * dact_func(pacts[15],n)));
    memset(neruon.units,0,8*sizeof(float));
    
    #pragma omp simd collapse(2)
    for(int i = 7 ; i < 16; i++){
        n.units[i] = dldz*n.weights[8][i-7];
        float dz = n.units[i]*(1+(n.alpha[i]*dact_func(pacts[i],n)));
        for (int j = 0; j < 7; j++)
        {
            n.units[j+1] += dz * n.weights[i-6][j];
        }
    }

    #pragma omp simd
    for (int i = 0; i < 7; i++)
    {
        float dz = n.units[i+1]*(1 + (n.alpha[i+1] * dact_func(pacts[i+1],n))); 
        n.units[0] += dz * weights[0][i];
    }

    n.units[0] *= (1 + (n.alpha[0] * dact_func(pacts[0],n)));
    return n.units[0];
}



struct neural_network{
    std::vector<neuron_unit *> network;
    std::vector<std::vector<float>> weights;// to improve performance consider implementing a interface to a 1d vector instead
    std::vector<int> input_index;           //indexing recording input neurons
    std::vector<int> output_index;          //indexing recording output neurons
    std::vector<std::vector<int>> layermap;
    std::vector<std::vector<bool>> dependency;
    
    neural_network(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, std::vector<int> memory_neurons, float connection_density, float connection_sd){

    }
    
    inline void forwardpass();
    inline void backpropagation();



    // saving to and loading from a text file
    inline void save_to_txt();
    neural_network(std::string textfile);
};

