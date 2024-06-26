/*
# Using Skip connection inspired by Resnet(He et al)https://arxiv.org/abs/1512.03385 as well as 0 initialisation from  
# fixup(Zhange et al)https://arxiv.org/abs/1901.09321.

# Xavier initilisation(Glorot et al)
# https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

# ReLU(Agarap) actiation function is also employed 
# https://arxiv.org/abs/1803.08375

# Nesterov momentum
# Nesterov, Y (1983). "A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)". Doklady AN USSR. 269: 543–547.

# Layernorm(Ba et Al) : https://arxiv.org/abs/1607.06450

# Random Orthogonal Matrices using algorithm by Stewart(1980) 
# The Efficient Generation of Random Orthogonal Matrices with an Application to Condition Estimators
# https://www.jstor.org/stable/2156882

# eventually will implement jagged array to replace all std::vector<std::vector<T>>
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
const float n_zero = 0.00001; //try and stop things from freaking out 
const float leak = 0;

std::uniform_real_distribution<float> zero_to_one(0.0,1.0);

// keyboard mashed the name for these because standard names like r_dev to avoid clashing
std::random_device rsksksksksks;                          
std::mt19937 ttttt(rsksksksksks());



inline float sign_of(float x){
    return((std::signbit(x) * -2.0f) + 1.0f);
}

// turn N by N array into identity matrix 
template <size_t N>
void inline identity_matrix(std::array<std::array<float, N>, N> &array)
{
    // zero out array then write the diagonals
    #pragma omp simd collapse(2)
    for(int i = 0 ; i < N; i++){
       for(int j = 0 ; j < N ; j++){
            array[i][j] = 0.0f;        
        }
    }
    #pragma omp simd
    for(int i = 0 ; i < N; i++){
        array[i][i] = 1.0f;
    }
}



// generate a random orthogonal array using method 
// of Householder Matrix
// Stewart(1980) The Efficient Generation of Random Orthogonal Matrices with an Application to Condition Estimators
// https://www-jstor-org.ucd.idm.oclc.org/stable/2156882?seq=4
template <size_t N>
void rorthog_arr(std::array<std::array<float, N>, N> &array)
{
    std::normal_distribution<float> stdnorm(0.0f,1.0f);
    std::array<std::array<float, N>, N> I_n;
    std::array<std::array<float, N>, N> Ortho;
    identity_matrix(I_n);
    Ortho = I_n;
    std::array<std::array<float, N>, N> D_e=I_n;
    for(int i=1;i<N;i++){
        std::array<std::array<float, N>, N> H_k;
        H_k = I_n;
        std::array<std::array<float, N>, N> product;
        std::vector<float> rand_norm_vec(N-i+1,0);
        float norm = 0;
        for(int j = 0; j < N-i+1; j++){
            rand_norm_vec[j] = stdnorm(ttttt);
            norm += rand_norm_vec[j] * rand_norm_vec[j];
        }
        float x_norm = sqrt(norm);
        norm -= rand_norm_vec[0] * rand_norm_vec[0]; 
        D_e[i-1][i-1]=sign_of(rand_norm_vec[0]);
        rand_norm_vec[0] -= D_e[i-1][i-1]*x_norm;
        norm += rand_norm_vec[0] * rand_norm_vec[0]; 
        norm = (1.0f/norm);
        for(int j = 0; j < N-i+1; j++){
            for(int k = 0; k < N-i+1; k++){
                H_k[j+i-1][k+i-1]-=2 * rand_norm_vec[j]*rand_norm_vec[k]*norm;
            }
        }
        // matrix multiplication naive implementation hope the cache is big
        for(int j = 0 ; j < N; j++){
            for(int k = 0; k < N; k++){
                product[j][k] = 0;
            }
            for(int k = 0; k < N; k++){
                for(int l = 0; l < N; l++){
                    product[j][k] +=  H_k[j][l]*Ortho[l][k];
                }
            }
        }
        Ortho=product;
    }
    // final result optionally try and mess with determinant
    for(int j = 0 ; j < N; j++){
        for(int k = 0; k < N; k++){
            array[j][k] = 0;
        }
        for(int k = 0; k < N; k++){
            for(int l = 0; l < N; l++){
                array[j][k] += D_e[j][l]*Ortho[l][k];
            }
        }
    }
}

template <size_t N>
void print_matrix(std::array<std::array<float, N>, N> &array)
{
    for(int i = 0 ; i < N; i++){
        std::cout<<"{";
        for(int j = 0 ; j < N-1; j++){
            std::cout<<array[i][j]<<", ";
        }
        std::cout<<array[i][N-1]<<"}\n";
    }
    std::cout<<std::endl;
}

float quadractic_increase(float x){
    if (std::abs(x) > 1)
    {
        int sign = sign_of(x);
        x = std::sqrt(std::abs(x)) * sign;
        return x;
    }
    else{
        return x;
        }
}

float log_increase(float x){
    if (std::abs(x) > 1)
    {
        int sign = sign_of(x);
        x = std::log(std::abs(x) + 1) * sign;
        return x;
    }
    else{
        return x;
        }
}
//based on "A handy approximation for the error function and its inverse" by Sergei Winitzki.,
//https://www.academia.edu/9730974/A_handy_approximation_for_the_error_function_and_its_inverse
float approx_erfinv(float x){
    static const double a = 0.6802721088435374;         //this is 1/1.47
    static const float pita_inv = 0.4330746750799873;  //this is 1/1.47 times 1/pi times 2
    float w = 1-x*x;
    w = (w == 1) ? 0.999999f:w;
    w = std::log(w);
    float u = pita_inv+(w*0.5);
    float val = std::sqrt(-u + std::sqrt((u * u) - (a * w)));
    return(val * sign_of(x));
}

/*
# activation functions and their derivatives
*/

#pragma omp declare simd
inline float relu(float x){
    return ((x>0.0f) ? x:0.0f);
}
#pragma omp declare simd
inline float drelu(float fx){
    return (fx>0.0f);
}

// it may be useful to utilise log space 
// so this function is simply ln(relu(x)+1)
#pragma omp declare simd
inline float log_relu(float x){
    return std::log((x>0.0f) ? (x+1.0f):1.0f);
}
#pragma omp declare simd
inline float dlog_relu(float fx){
    return ((fx>0.0f) ? std::exp(-fx):0.0f);
}

// function to compute cos(x) from sin(x)
// i.e. the derivative of sin(x) activation function
#pragma omp declare simd
inline float cos_sinx(float sinx){
    return std::sqrt(1 - (sinx*sinx));
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
#pragma omp declare simd
inline float lrelu(float x){
    return((x>0) ? x:leak*x);
}
#pragma omp declare simd
inline float sigmoid(float x){
    x = (std::abs(x)<50) ?x:50*sign_of(x);     //to prevent underflow and overflow
    return 1.0f/(std::exp(-x)+1);
}

void soft_max(std::vector<float> &output){
    double denominator = 0;
    #pragma omp parallel for simd reduction(+:denominator)
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = (std::abs(output[i])<50) ?output[i]:50*sign_of(output[i]);
        output[i] = std::exp(output[i]);
        denominator += output[i]; 
    }
    denominator = (std::abs(denominator) < n_zero) ? n_zero:denominator;
    denominator = 1 / denominator; 
    #pragma omp parallel for simd
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = output[i] * denominator;
    }
}

int prediction(std::vector<float> &output){
    int guess = std::distance(output.begin(),std::max_element(output.begin(),output.end()));
    return guess;
}

void dsoft_max(std::vector<float> &output, std::vector<float> &target, std::vector<float> &loss){
    #pragma omp parallel for simd
    for (int i = 0; i < output.size(); i++)
    {
        loss[i] = output[i] - target[i];
    }
}

float pmf_entrophy(std::vector<float> &pmf){
    float entrophy = 0;
    #pragma omp parallel for simd reduction(-:entrophy)
    for(int i = 0 ; i < pmf.size(); i++){
        entrophy -= pmf[i] * std::log(pmf[i]);
    }
    return entrophy;
}

float cross_entrophy_loss(std::vector<float> &target, std::vector<float> & output){
    float loss = 0;
    #pragma omp parallel for simd reduction(-:loss)
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = (output[i] > n_zero) ? output[i]:n_zero;
        loss -= (target[i] * std::log(output[i]));
    }
    return loss;
}





// struct for holding gradients for individual neurons before update
// neurons will have 1 input 1 output and 2 hidden layers of 8 units
struct neuron_gradients
{
    float bias[16] = {0};
    float weights[9][7] = {0};
    float padding_or_param1 = 0;
    inline void valclear(){    //wrapper function for memset
        #pragma omp simd
        for(uint_fast8_t i = 0; i < 16; i++){
            bias[i] = 0;
        }
        #pragma omp simd collapse(2)
        for(uint_fast8_t i = 0 ; i < 9; i ++){
            for(uint_fast8_t j = 0 ; j < 7; j++){
                weights[i][j] = 0;
            }
        }
        padding_or_param1=0;
    }   

    // zeros out the current gradient
    // so exponential averageing but who to cite?
    template <typename neuron>
    inline void sgd_with_momentum(neuron &n, float learning_rate, float beta, neuron_gradients &current_grad, float b_correction=1){    
        float b = 1 - beta;
        learning_rate *= (-b_correction);
        padding_or_param1 *= beta;
        padding_or_param1 += current_grad.padding_or_param1 * b;
        n.padding_or_param1 += padding_or_param1 * learning_rate;
        current_grad.padding_or_param1 = 0;
        #pragma omp simd
        for (uint_fast8_t i = 0; i < 16; i++)
        {
            bias[i] *= beta;
            bias[i] += current_grad.bias[i] * b;
            n.bias[i] += bias[i] * learning_rate;
            current_grad.bias[i] = 0;
        }
        #pragma omp simd collapse(2)
        for (uint_fast8_t i = 0; i < 9; i++)
        {   
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                weights[i][j] *= beta;
                weights[i][j] += current_grad.weights[i][j] * b;
                n.weights[i][j] += weights[i][j] * learning_rate;
                current_grad.weights[i][j] = 0;
            }
        }
        
    }

    template <typename neuron>
    inline void sgd_with_nesterov(neuron &n, float learning_rate, float beta, neuron_gradients &current_grad){    
        padding_or_param1 *= beta;
        padding_or_param1 += current_grad.padding_or_param1;
        n.padding_or_param1 -= (padding_or_param1 + current_grad.padding_or_param1*beta) * learning_rate;
        current_grad.padding_or_param1 = 0;
        #pragma omp simd
        for (uint_fast8_t i = 0; i < 16; i++)
        {
            bias[i] *= beta;
            bias[i] += current_grad.bias[i];
            n.bias[i] -= (bias[i]+current_grad.bias[i] * beta) * learning_rate;
            current_grad.bias[i] = 0;
        }
        #pragma omp simd collapse(2)
        for (uint_fast8_t i = 0; i < 9; i++)
        {   
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                weights[i][j] *= beta;
                weights[i][j] += current_grad.weights[i][j];
                n.weights[i][j] -= (current_grad.weights[i][j] + weights[i][j]*beta) * learning_rate;
                current_grad.weights[i][j] = 0;
            }
        }
        
    }

    inline void add(neuron_gradients & grad){
        padding_or_param1 += grad.padding_or_param1;
        #pragma omp simd
        for (int i = 0; i < 16; i++)
        {
            bias[i] += grad.bias[i];
        }
        #pragma omp simd collapse(2)
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 7; j++)
            {
                weights[i][j] += grad.weights[i][j];
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
    float units[16] = {0};     // 0 will be input and 15 will be output
    float bias[16]  = {0};   
    float weights[9][7];
    int32_t isinput = 0;         
    int32_t a_f = 0;          //activation function defaults to ReLU with layernorm
    float padding_or_param1=0;
    float padding_or_param2=0;
    float padding_or_param3=0;
    //tags for tag dispatching
    struct relu_neuron{relu_neuron(){}};            // a_f = 0
    struct sine_neuron{sine_neuron(){}};            // a_f = 1
    struct log_relu_neuron{log_relu_neuron(){}};    // a_f = 2
    struct memory_neuron{memory_neuron(){}};        // a_f = 3
    struct sin_memory_neuron{sin_memory_neuron(){}};        // a_f = 4
    struct layernorm_tag{layernorm_tag(){}};
    
    neuron_unit(){
        //  Xavier initialisation sorta?
        float Xavier_init_a = 0.5f;
        std::normal_distribution<float> a (0,Xavier_init_a);            // input to 1st hidden layer
        std::array<std::array<float,7>,7> orthogonal_weight_matrix;
        rorthog_arr(orthogonal_weight_matrix);
        float norm = 0;
        for (uint_fast8_t i = 0; i < 7; i++)
        {
            weights[0][i] = a(ttttt);
            norm += weights[0][i] * weights[0][i];
        }
        norm = std::sqrt(1.0f/norm);
        for (uint_fast8_t i = 0; i < 7; i++)
        {
            weights[0][i] *= norm;
        }
        #pragma omp simd
        for (uint_fast8_t i = 0; i < 7; i++)
        {
            weights[8][i] = 0;
        }
        #pragma omp simd collapse(2)
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                weights[i][j] = orthogonal_weight_matrix[i-1][j];
            }
        }
    }
    
    inline void valclear(){    //wrapper function for memset on units
        #pragma omp simd
        for(uint_fast8_t i = 0; i < 16; i++){
            units[i] = 0;
        }
    };

    // wrapper function for relu and drelu
    #pragma omp declare simd
    inline float act_func(float x, relu_neuron){
        return relu(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, relu_neuron){
        return drelu(fx);
    }

    // wrapper function for sine and cos(arcsin(x))
    #pragma omp declare simd
    inline float act_func(float x, sine_neuron){
        return std::sin(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, sine_neuron){
        return cos_sinx(fx);
    }

    // wrapper function for log_relu and dlog_relu
    #pragma omp declare simd
    inline float act_func(float x, log_relu_neuron){
        return log_relu(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, log_relu_neuron){
        return dlog_relu(fx);
    }
    
    inline void reinit_mem(){
        a_f=3;
        float norm=0;
        float Xavier_init_a = 0.5f;
        std::normal_distribution<float> a (0,Xavier_init_a);            // input to 1st hidden layer
        for (uint_fast8_t i = 0; i < 7; i++)
        {
            weights[8][i] = a(ttttt);
        }
        for(uint_fast8_t i = 0; i < 3; i++){
            norm += weights[8][i]*weights[8][i];
        }
        norm = std::sqrt(1.0f/norm);
        for(uint_fast8_t i = 0; i < 3; i++){
            weights[8][i]*=norm;
        }
        norm = 0;
        for(uint_fast8_t i = 3; i < 7; i++){
            norm += weights[8][i]*weights[8][i];
        }
        norm = std::sqrt(1.0f/norm);
        for(uint_fast8_t i = 3; i < 7; i++){
            weights[8][i]*=norm;
        }
    }

    template <typename T>
    inline void f_p(T af){ //pacts here refers to values obtained after applying activation function
        units[0] += bias[0];
        units[0] = act_func(units[0],af);
        #pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = units[0] * weights[0][i-1];
            units[i] += bias[i];
            units[i] = act_func(units[i],af);
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 16; i++)
        {
            units[i] = bias[i];        
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[i] += units[j+1] * weights[i-7][j];
            }    
        } 
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = act_func(units[i],af);
        }
        #pragma omp simd reduction(+:units[15])
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[15] += units[i] * weights[8][i-8];
        }
        //units[15] = act_func(units[15],af);
    }

    template <typename T>
    inline void f_p(T af, layernorm_tag &){ //pacts here refers to values obtained after applying activation function
        units[0] += bias[0];
        units[0] = act_func(units[0],af);
        float mean = 0;
        float sd = 0;
        //#pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = units[0] * weights[0][i-1];
            units[i] += bias[i];
            units[i] = act_func(units[i],af);

            // for layernorm
            mean += units[i];
        }

        mean *= 0.14285714285714285f;           // divide by 7

        //#pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            sd += (units[i] - mean)*(units[i] - mean);
        }

        sd *= 0.14285714285714285f;           // divide by 7
        sd = std::sqrt(sd);
        sd = 1.0f/(sd+n_zero);
        // layernorm part
        //#pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = (units[i] - mean)*sd;
        }

        //#pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = bias[i];        
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[i] += units[j+1] * weights[i-7][j];
            }    
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = act_func(units[i],af);
        }
        units[15] = bias[15];
        //#pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[15] += units[i] * weights[8][i-8];
        }
        //units[15] = act_func(units[15],af);
    }

    inline void f_p(memory_neuron &){ //pacts here refers to values obtained after applying activation function
        units[0] += bias[0];
        //units[0] = relu(units[0]);
        #pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = units[0] * weights[0][i-1];
            units[i] += bias[i];
            units[i] = relu(units[i]);
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = bias[i];        
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[i] += units[j+1] * weights[i-7][j];
            }    
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = relu(units[i]);
        }
        // the output will be determined elsewhere
    }

    inline void mem_update(float ht_m1){
        float a = padding_or_param1;
        float b = bias[15];
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++)
        {
            a += units[i] * weights[8][i-8];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++)
        {
            b += units[i] * weights[8][i-8];
        }
        units[15] = ht_m1 + std::tanh(b)*sigmoid(a);
    }

    inline void forward_pass(float ht_m1=0.0f){
        sine_neuron s;
        log_relu_neuron l;
        relu_neuron r;
        memory_neuron m;
        sin_memory_neuron si;
        layernorm_tag lt;
        float skip_c = units[0];
        switch (a_f)
        {
        case 1:
            f_p(s);
            units[15] += skip_c;
            return;
        case 2:
            f_p(l);
            units[15] += skip_c;
            return;
        case 3:
            f_p(m);
            mem_update(ht_m1);
            //units[15] += skip_c;
            return;
        case 5:
            f_p(r,lt);
            units[15] += skip_c;
            return;
        default:
            f_p(r);
            units[15] += skip_c;
            return;
        }
    }

    inline float backprop(float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, memory_neuron&, float &tm1grad)
    {   
        float a = padding_or_param1;
        float b = bias[15];
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++)
        {
            a += past_unit[i] * weights[8][i-8];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++)
        {
            b += past_unit[i] * weights[8][i-8];
        }
        #pragma omp simd
        for(uint_fast8_t i = 0 ; i < 8; i++){
            units[i] = 0;
        }
        tm1grad += dldz;
        float tanhb = std::tanh(b);
        float siga = sigmoid(a);
        float da = dldz * siga*(1-siga)*tanhb;
        float db = dldz * siga*(1-(tanhb*tanhb));
        gradients.padding_or_param1 += da;
        gradients.bias[15] += db;
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++){
            units[i] = da*weights[8][i-8];
            gradients.weights[8][i-8] += da*past_unit[i];
            units[i]*=drelu(past_unit[i]);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++){
            units[i] = db*weights[8][i-8];
            gradients.weights[8][i-8] += db*past_unit[i];
            units[i] *= drelu(past_unit[i]);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd collapse(2) safelen(7) 
        for (uint_fast8_t i = 8; i < 15; i++){
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[j+1] += units[i] * weights[i-7][j];
                gradients.weights[i-7][j] += units[i]*past_unit[j+1];
            }
        }
        #pragma omp simd reduction(+:units[0])
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i]*=drelu(past_unit[i]);
            gradients.bias[i] += units[i];
            units[0] += units[i] * weights[0][i-1];
        }
        //units[0] *= drelu(past_unit[0]);
        gradients.bias[0] += units[0];
        return units[0];
    }    


    // note the units array will be used to store gradients
    template <typename T>
    inline float backprop(float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, T af)
    {   
        #pragma omp simd
        for(uint_fast8_t i = 0 ; i < 8; i++){
            units[i] = 0;
        }
        gradients.bias[15] += dldz;
        #pragma omp simd
        for(uint_fast8_t i = 8 ; i < 15; i++){
            units[i] = dldz * weights[8][i-8];
            gradients.weights[8][i-8]+=dldz*past_unit[i];
            units[i]*=dact_func(past_unit[i],af);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for(uint_fast8_t i = 8 ; i < 15; i++){
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[j+1] += units[i] * weights[i-7][j];
                gradients.weights[i-7][j] += units[i]*past_unit[j+1];
            }
        }
        #pragma omp simd reduction(+:units[0])
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] *= dact_func(past_unit[i],af);
            gradients.bias[i] += units[i];
            units[0] += units[i] * weights[0][i-1];
        }
        units[0] *= dact_func(past_unit[0],af);
        gradients.bias[0] += units[0];
        return units[0];
    }    
    

    // note the units array will be used to store gradients
    template <typename T>
    inline float backprop(float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, T af, layernorm_tag &)
    {   
        /*
        units[15] = bias[15];
        //#pragma omp simd
        for(uint_fast8_t i = 8 ; i < 15; i++){
            units[15] += weights[8][i-8] * past_unit[i];
        }
        units[15] = act_func(units[15],af);
        dldz = dldz * dact_func(units[15],af);*/
        gradients.bias[15] += dldz;
        //#pragma omp simd
        for(uint_fast8_t i = 0 ; i < 8; i++){
            units[i] = 0;
        }
        float mean = 0;
        float sd = 0;
        //#pragma omp simd
        for (int i = 1; i < 8; i++)
        {
            mean += past_unit[i];
        }
        
        mean *= 0.14285714285714285f;           // divide by 7

        //#pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            sd += (past_unit[i] - mean)*(past_unit[i] - mean);
        }

        sd *= 0.14285714285714285f;           // divide by 7
        sd = std::sqrt(sd);
        float den = 0.14285714285714285f/(sd+n_zero);
        sd = 1.0f/(sd+n_zero);
        for(int i = 8 ; i < 15; i++){
            units[i] = dldz*weights[8][i-8];
            gradients.weights[8][i-8] += dldz*past_unit[i];
            units[i] = units[i]*dact_func(past_unit[i],af);
            gradients.bias[i] += units[i];
            //#pragma omp simd
            for (int j = 0; j < 7; j++)
            {
                units[j+1] += units[i] * weights[i-7][j];
                gradients.weights[i-7][j] += units[i]*past_unit[j+1];
            }
        }
        units[8] = 0;
        units[9] = 0;
        units[10] = 0;
        units[11] = 0;
        units[12] = 0;
        units[13] = 0;
        units[14] = 0;
        //#pragma omp simd collapse(2)
        for (int i = 1; i < 8; i++)
        {
            float component = (past_unit[i]-mean) * sd * sd * den;            
            for (int j = 8; j < 15; j++)
            {
                units[j] += units[i] * (sd * ((i==(j-7))-0.14285714285714285f) - component * (past_unit[j-7]-mean));
            }
        }
        //#pragma omp simd
        for (int i = 1; i < 8; i++)
        {
            units[i+7] = units[i+7]*dact_func(past_unit[i],af);
            gradients.bias[i] += units[i+7];
            units[0] += units[i+7] * weights[0][i-1];
        }
        units[0] *= dact_func(past_unit[0],af);
        gradients.bias[0] += units[0];
        return units[0];
    }    

    inline float backpropagation(float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients,float &tm1grad){
        sine_neuron s;
        log_relu_neuron l;
        relu_neuron r;
        memory_neuron m;
        sin_memory_neuron si;
        layernorm_tag lt;
        switch (a_f)
        {
        case 1:
            return (backprop(dldz, past_unit, gradients, s) + dldz);
        case 2:
            return (backprop(dldz, past_unit, gradients, l) + dldz);
        case 3:
            return (backprop(dldz,past_unit,gradients, m,tm1grad));   
        case 5:
            return (backprop(dldz, past_unit, gradients, r, lt) + dldz);
        default:
            return (backprop(dldz,past_unit,gradients, r) + dldz);   
        }
    }
};

struct neuron_unit_copy
{
    float units[16] = {0};     // 0 will be input and 15 will be output
    //tags for tag dispatching
    struct relu_neuron{relu_neuron(){}};            // a_f = 0
    struct sine_neuron{sine_neuron(){}};            // a_f = 1
    struct log_relu_neuron{log_relu_neuron(){}};    // a_f = 2
    struct memory_neuron{memory_neuron(){}};        // a_f = 3
    struct sin_memory_neuron{sin_memory_neuron(){}};        // a_f = 4
    struct layernorm_tag{layernorm_tag(){}};
    
    neuron_unit_copy(){}
    
    inline void valclear(){    //wrapper function for memset on units
        #pragma omp simd
        for(uint_fast8_t i = 0; i < 16; i++){
            units[i] = 0;
        }
    };

    // wrapper function for relu and drelu
    #pragma omp declare simd
    inline float act_func(float x, relu_neuron){
        return relu(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, relu_neuron){
        return drelu(fx);
    }

    // wrapper function for sine and cos(arcsin(x))
    #pragma omp declare simd
    inline float act_func(float x, sine_neuron){
        return std::sin(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, sine_neuron){
        return cos_sinx(fx);
    }

    // wrapper function for log_relu and dlog_relu
    #pragma omp declare simd
    inline float act_func(float x, log_relu_neuron){
        return log_relu(x);
    }
    #pragma omp declare simd
    inline float dact_func(float fx, log_relu_neuron){
        return dlog_relu(fx);
    }

    template <typename T>
    inline void f_p(neuron_unit & n, T af){ //pacts here refers to values obtained after applying activation function
        units[0] += n.bias[0];
        units[0] = act_func(units[0],af);
        #pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = units[0] * n.weights[0][i-1];
            units[i] += n.bias[i];
            units[i] = act_func(units[i],af);
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 16; i++)
        {
            units[i] = n.bias[i];        
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[i] += units[j+1] * n.weights[i-7][j];
            }    
        } 
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = act_func(units[i],af);
        }
        #pragma omp simd reduction(+:units[15])
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[15] += units[i] * n.weights[8][i-8];
        }
        //units[15] = act_func(units[15],af);
    }

    template <typename T>
    inline void f_p(T af, layernorm_tag &){ //pacts here refers to values obtained after applying activation function
        //will eventually get to it?
    }

    inline void f_p(neuron_unit & n, memory_neuron &){ //pacts here refers to values obtained after applying activation function
        units[0] += n.bias[0];
        //units[0] = relu(units[0]);
        #pragma omp simd
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] = units[0] * n.weights[0][i-1];
            units[i] += n.bias[i];
            units[i] = relu(units[i]);
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = n.bias[i];        
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[i] += units[j+1] * n.weights[i-7][j];
            }    
        }
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 15; i++)
        {
            units[i] = relu(units[i]);
        }
        // the output will be determined elsewhere
    }

    inline void mem_update(neuron_unit & n, float ht_m1){
        float a = n.padding_or_param1;
        float b = n.bias[15];
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++)
        {
            a += units[i] * n.weights[8][i-8];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++)
        {
            b += units[i] * n.weights[8][i-8];
        }
        units[15] = ht_m1 + std::tanh(b)*sigmoid(a);
    }

    inline void forward_pass(neuron_unit & n, float ht_m1=0.0f){
        sine_neuron s;
        log_relu_neuron l;
        relu_neuron r;
        memory_neuron m;
        sin_memory_neuron si;
        layernorm_tag lt;
        float skip_c = units[0];
        switch (n.a_f)
        {
        case 1:
            f_p(n,s);
            units[15] += skip_c;
            return;
        case 2:
            f_p(n,l);
            units[15] += skip_c;
            return;
        case 3:
            f_p(n,m);
            mem_update(n,ht_m1);
            //units[15] += skip_c;
            return;
        case 5:
            //f_p(r,lt);
            units[15] += skip_c;
            return;
        default:
            f_p(n,r);
            units[15] += skip_c;
            return;
        }
    }

    inline float backprop(neuron_unit & n,float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, memory_neuron&, float &tm1grad)
    {   
        float a = n.padding_or_param1;
        float b = n.bias[15];
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++)
        {
            a += past_unit[i] * n.weights[8][i-8];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++)
        {
            b += past_unit[i] * n.weights[8][i-8];
        }
        #pragma omp simd
        for(uint_fast8_t i = 0 ; i < 8; i++){
            units[i] = 0;
        }
        tm1grad += dldz;
        float tanhb = std::tanh(b);
        float siga = sigmoid(a);
        float da = dldz * siga*(1-siga)*tanhb;
        float db = dldz * siga*(1-(tanhb*tanhb));
        gradients.padding_or_param1 += da;
        gradients.bias[15] += db;
        #pragma omp simd
        for (uint_fast8_t i = 8; i < 11; i++){
            units[i] = da*n.weights[8][i-8];
            gradients.weights[8][i-8] += da*past_unit[i];
            units[i]*=drelu(past_unit[i]);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd
        for (uint_fast8_t i = 11; i < 15; i++){
            units[i] = db*n.weights[8][i-8];
            gradients.weights[8][i-8] += db*past_unit[i];
            units[i] *= drelu(past_unit[i]);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd collapse(2) safelen(7) 
        for (uint_fast8_t i = 8; i < 15; i++){
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[j+1] += units[i] * n.weights[i-7][j];
                gradients.weights[i-7][j] += units[i]*past_unit[j+1];
            }
        }
        #pragma omp simd reduction(+:units[0])
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i]*=drelu(past_unit[i]);
            gradients.bias[i] += units[i];
            units[0] += units[i] * n.weights[0][i-1];
        }
        //units[0] *= drelu(past_unit[0]);
        gradients.bias[0] += units[0];
        return units[0];
    }    


    // note the units array will be used to store gradients
    template <typename T>
    inline float backprop(neuron_unit &n,float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, T af)
    {   
        #pragma omp simd
        for(uint_fast8_t i = 0 ; i < 8; i++){
            units[i] = 0;
        }
        gradients.bias[15] += dldz;
        #pragma omp simd
        for(uint_fast8_t i = 8 ; i < 15; i++){
            units[i] = dldz * n.weights[8][i-8];
            gradients.weights[8][i-8]+=dldz*past_unit[i];
            units[i]*=dact_func(past_unit[i],af);
            gradients.bias[i] += units[i];
        }
        #pragma omp simd collapse(2) safelen(7) //reduction(+:units)
        for(uint_fast8_t i = 8 ; i < 15; i++){
            for (uint_fast8_t j = 0; j < 7; j++)
            {
                units[j+1] += units[i] * n.weights[i-7][j];
                gradients.weights[i-7][j] += units[i]*past_unit[j+1];
            }
        }
        #pragma omp simd reduction(+:units[0])
        for (uint_fast8_t i = 1; i < 8; i++)
        {
            units[i] *= dact_func(past_unit[i],af);
            gradients.bias[i] += units[i];
            units[0] += units[i] * n.weights[0][i-1];
        }
        units[0] *= dact_func(past_unit[0],af);
        gradients.bias[0] += units[0];
        return units[0];
    }    
    

    // note the units array will be used to store gradients
    template <typename T>
    inline float backprop(float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients, T af, layernorm_tag &)
    {   
        // work in progress
    }    

    inline float backpropagation(neuron_unit & n, float dldz, std::array<float,16> &past_unit, neuron_gradients &gradients,float &tm1grad){
        sine_neuron s;
        log_relu_neuron l;
        relu_neuron r;
        memory_neuron m;
        sin_memory_neuron si;
        layernorm_tag lt;
        switch (n.a_f)
        {
        case 1:
            return (backprop(n,dldz, past_unit, gradients, s) + dldz);
        case 2:
            return (backprop(n,dldz, past_unit, gradients, l) + dldz);
        case 3:
            return (backprop(n,dldz,past_unit,gradients, m,tm1grad));   
        case 5:
            return (dldz);
        default:
            return (backprop(n,dldz,past_unit,gradients, r) + dldz);   
        }
    }
};

// pdf is f(x) = m(a^2*e^(a^2)) where a(x) = 10(x-0.5), m is a constant approx =11.28379...., in the interval 0 < x < 1
// has a shape with 2 humps around 0.5
float custom_dist(){
    float x = zero_to_one(ttttt);
    x = 1 - 2*x;
    x = 0.1 * (5 - approx_erfinv(x));
    return x;
}

// vector accessible as vector of arrays
// more work needed on this to eliminate the need for vector of vectors
template<typename T>
struct vec_of_arr
{
    std::vector<T> vec;
    int arr_size;
    int vec_size;
    T & operator ()(int i, int j) const {
        return vec[arr_size*i + j];
    }
    T & operator ()(int i, int j){
        return vec[arr_size*i + j];
    }
    vec_of_arr(){}
    vec_of_arr(int vec_len, int arr_len):vec(vec_len*arr_len,0){arr_size = arr_len;vec_size=vec_len;}
    //appending 'array' to this vector, assumes T is POD
    void app_arr(std::vector<T> &arr){
        vec.reserve(vec.size() + arr_size);
        vec.resize(vec.size() + arr_size);
        for (int i = 0; i < arr_size; i++)
        {
            vec[arr_size*vec_size + i] = arr[i];
        }
        vec_size++;
    }
    void inline resize(int a){
        int b = vec_size;
        vec_size = a;
        vec.reserve(a*arr_size);
        vec.resize(a*arr_size);
        for (int i = b; i < a-b; i++)
        {
            vec[i]=0;
        }
    }
};

struct int_float
{
    int x;
    float y;
    inline int_float(int a, float b){x=a;y=b;}
};

struct  int_int
{
    int x;
    int y;
    inline int_int(int a, int b){x=a;y=b;}
};

struct NN
{
    std::vector<neuron_unit> neural_net;
    std::vector<std::vector<int_float>> weights;
    std::vector<std::vector<int_int>> bweights;

    struct network_gradient
    {
        std::vector<neuron_gradients> net_grads;
        std::vector<std::vector<float>> weight_gradients;
        network_gradient(){}
        network_gradient(NN & nn)
        :net_grads(nn.neural_net.size(),neuron_gradients())
        ,weight_gradients(nn.neural_net.size())
        {
            for (int i = 0; i < nn.neural_net.size(); i++)
            {
                weight_gradients[i].resize(nn.weights[i].size());
                #pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = 0;
                }
            }
        }
        void sync(NN & nn){
            for (int i = 0; i < nn.neural_net.size(); i++)
            {
                weight_gradients[i].resize(nn.weights[i].size());
                #pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = 0;
                }
            }
        }
        void valclear(){
            #pragma omp parallel 
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < net_grads.size(); i++)
                {
                    net_grads[i].valclear();
                }
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < weight_gradients.size(); i++)
                {
                    #pragma omp simd
                    for (int j = 0; j < weight_gradients[i].size(); j++)
                    {
                        weight_gradients[i][j] = 0;
                    }
                }
            }
        }

        //neuron &n, float learning_rate, float momentum, neuron_gradients current_grad
        inline void sgd_with_momentum(NN &n, float learning_rate, float beta, network_gradient &current_gradient, float iter_bias_correction=-1){
            iter_bias_correction = (iter_bias_correction > 60) ? -1:iter_bias_correction;
            iter_bias_correction = (iter_bias_correction>=0) ? (1.0f/(1.0f-std::pow(beta,iter_bias_correction+1))):1.0f;
            float ombeta  = 1-beta;
            #pragma omp parallel shared(learning_rate)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < n.neural_net.size(); i++)
                {
                    net_grads[i].sgd_with_momentum(n.neural_net[i],learning_rate,beta,current_gradient.net_grads[i],iter_bias_correction);
                }
                #pragma omp single
                {
                    learning_rate *= (-iter_bias_correction);
                }
                #pragma omp barrier
                #pragma omp for schedule(dynamic,16)
                for (int i = 0; i < weight_gradients.size(); i++)
                {
                    #pragma omp simd
                    for (int j = 0; j < weight_gradients[i].size(); j++)
                    {
                        weight_gradients[i][j] *= beta;
                        weight_gradients[i][j] += current_gradient.weight_gradients[i][j]* ombeta;
                        n.weights[i][j].y += weight_gradients[i][j]*learning_rate;
                        current_gradient.weight_gradients[i][j] = 0;
                    }   
                }
            }
        }

        inline void sgd_with_nesterov(NN &n, float learning_rate, float beta, network_gradient &current_gradient, int unused_arg=0){
            #pragma omp parallel shared(learning_rate)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < n.neural_net.size(); i++)
                {
                    net_grads[i].sgd_with_nesterov(n.neural_net[i],learning_rate,beta,current_gradient.net_grads[i]);
                }
                #pragma omp barrier
                #pragma omp for schedule(dynamic,16)
                for (int i = 0; i < weight_gradients.size(); i++)
                {
                    #pragma omp simd
                    for (int j = 0; j < weight_gradients[i].size(); j++)
                    {
                        weight_gradients[i][j] *= beta;
                        weight_gradients[i][j] += current_gradient.weight_gradients[i][j];
                        n.weights[i][j].y -= (weight_gradients[i][j]+current_gradient.weight_gradients[i][j] * beta)*learning_rate;
                        current_gradient.weight_gradients[i][j] = 0;
                    }   
                }
            }
        }


        //restrict gradients from -1 to 1
        inline void restrict_tanh(){
            for (int i = 0; i < net_grads.size(); i++)
            {
                //#pragma omp simd
                for (int j = 0; j < 16; j++)
                {
                    net_grads[i].bias[j] = std::tanh(net_grads[i].bias[j]);
                }
                net_grads[i].padding_or_param1 = std::tanh(net_grads[i].padding_or_param1);
                #pragma omp simd collapse(2)
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        net_grads[i].weights[j][k] = std::tanh(net_grads[i].weights[j][k]);
                    }   
                }
            }
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                //#pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = std::tanh(weight_gradients[i][j]);
                }
            }
        }
        inline void restrict_log(){
            #pragma omp for
            for (int i = 0; i < net_grads.size(); i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    net_grads[i].bias[j] = log_increase(net_grads[i].bias[j]);
                }
                net_grads[i].padding_or_param1 = log_increase(net_grads[i].padding_or_param1);
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        net_grads[i].weights[j][k] = log_increase(net_grads[i].weights[j][k]);
                    }   
                }
            }
            #pragma omp for
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = log_increase(weight_gradients[i][j]);
                }
            }
        }

        inline void restrict_quadratic(){
            #pragma omp for
            for (int i = 0; i < net_grads.size(); i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    net_grads[i].bias[j] = quadractic_increase(net_grads[i].bias[j]);
                }
                net_grads[i].padding_or_param1 = quadractic_increase(net_grads[i].padding_or_param1);
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        net_grads[i].weights[j][k] = quadractic_increase(net_grads[i].weights[j][k]);
                    }   
                }
            }
            #pragma omp for
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                //#pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = quadractic_increase(weight_gradients[i][j]);
                }
            }
        }

        inline float clip(float x, float max){
            return ((std::abs(x)<max) ? x : (sign_of(x)*max));
        }
        inline void clip(float max){
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < net_grads.size(); i++)
            {
                //#pragma omp simd
                for (int j = 0; j < 16; j++)
                {
                    net_grads[i].bias[j] = clip(net_grads[i].bias[j],max);
                }
                net_grads[i].padding_or_param1 = clip(net_grads[i].padding_or_param1,max);
                #pragma omp simd collapse(2)
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        net_grads[i].weights[j][k] = clip(net_grads[i].weights[j][k],max);
                    }   
                }
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                //#pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j] = clip(weight_gradients[i][j],max);
                }
            }
        }

        inline float global_norm_clip(float max){
            double gradient_l2_norm = 0;
            double returnnorm = 0;
            bool retu = false;
            #pragma omp parallel shared(gradient_l2_norm)
            {
                #pragma omp for reduction(+:gradient_l2_norm)
                for (int i = 0; i < net_grads.size(); i++)
                {
                    #pragma omp simd reduction(+:gradient_l2_norm)
                    for (int j = 0; j < 16; j++)
                    {
                        gradient_l2_norm += net_grads[i].bias[j]*net_grads[i].bias[j];
                    }
                    gradient_l2_norm += net_grads[i].padding_or_param1 * net_grads[i].padding_or_param1;
                    #pragma omp simd collapse(2) reduction(+:gradient_l2_norm)
                    for (int j = 0; j < 9; j++)
                    {
                        for (int k = 0; k < 7; k++)
                        {
                            gradient_l2_norm +=net_grads[i].weights[j][k]*net_grads[i].weights[j][k];
                        }   
                    }
                }
                #pragma omp for reduction(+:gradient_l2_norm)
                for (int i = 0; i < weight_gradients.size(); i++)
                {   
                    #pragma omp simd reduction(+:gradient_l2_norm)
                    for (int j = 0; j < weight_gradients[i].size(); j++)
                    {
                        gradient_l2_norm+=weight_gradients[i][j]*weight_gradients[i][j];
                    }
                }
                #pragma omp single
                {
                    gradient_l2_norm = std::sqrt(gradient_l2_norm);
                    /*
                    std::cout<<std::endl;
                    std::cout<<"gradient norm"<<gradient_l2_norm<<std::endl;
                    std::cout<<std::endl;*/
                    if (std::isnan(gradient_l2_norm))
                    {
                        std::cout<<"error, nan gradient norm"<<std::endl;
                        std::exit(0);
                    }  
                    returnnorm=gradient_l2_norm;
                    retu = gradient_l2_norm<max;
                    gradient_l2_norm = (!retu)? (max/gradient_l2_norm):gradient_l2_norm;
                }
                if (retu){}
                else{
                    #pragma omp for
                    for (int i = 0; i < net_grads.size(); i++)
                    {
                        #pragma omp simd
                        for (int j = 0; j < 16; j++)
                        {
                            net_grads[i].bias[j] *= gradient_l2_norm;
                        }
                        net_grads[i].padding_or_param1 *= gradient_l2_norm;
                        #pragma omp simd collapse(2)
                        for (int j = 0; j < 9; j++)
                        {
                            for (int k = 0; k < 7; k++)
                            {
                            net_grads[i].weights[j][k] *= gradient_l2_norm;
                            }   
                        }
                    }
                    #pragma omp for
                    for (int i = 0; i < weight_gradients.size(); i++)
                    {
                        #pragma omp simd
                        for (int j = 0; j < weight_gradients[i].size(); j++)
                        {
                            weight_gradients[i][j] *= gradient_l2_norm;
                        }
                    }
                }
            }
            return returnnorm;
        }

        inline void condense(std::vector<network_gradient> & multi_grad){
            for (int i = 0; i < multi_grad.size(); i++)
            {
                #pragma omp for
                for (int j = 0; j < multi_grad[i].net_grads.size(); j++)
                {
                    net_grads[j].add(multi_grad[i].net_grads[j]);
                }
                #pragma omp for
                for (int j = 0; j < multi_grad[i].weight_gradients.size(); j++)
                {
                    #pragma omp simd
                    for (int k = 0; k < multi_grad[i].weight_gradients[j].size(); k++)
                    {
                        weight_gradients[j][k] += multi_grad[i].weight_gradients[j][k];
                    }
                    
                }
            }
        }
        
        inline void condense_clear(network_gradient & grad){
            #pragma omp parallel 
            {
                #pragma omp for
                for (int j = 0; j < grad.net_grads.size(); j++)
                {
                    net_grads[j].add(grad.net_grads[j]);
                    grad.net_grads[j].valclear();
                }
                #pragma omp for
                for (int j = 0; j < grad.weight_gradients.size(); j++)
                {
                    #pragma omp simd
                    for (int k = 0; k < grad.weight_gradients[j].size(); k++)
                    {
                        weight_gradients[j][k] += grad.weight_gradients[j][k];
                        grad.weight_gradients[j][k] = 0;
                    }
                }
            }
        }
        inline void component_norm_clip(float neuron_unit_max, float weight_max){
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < net_grads.size(); i++)
            {
                double gradient_l2_norm = 0;
                #pragma omp simd reduction(+:gradient_l2_norm)
                for (int j = 0; j < 16; j++)
                {
                    gradient_l2_norm += net_grads[i].bias[j]*net_grads[i].bias[j];
                }
                gradient_l2_norm += net_grads[i].padding_or_param1 * net_grads[i].padding_or_param1;
                #pragma omp simd collapse(2) reduction(+:gradient_l2_norm)
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        gradient_l2_norm +=net_grads[i].weights[j][k]*net_grads[i].weights[j][k];
                    }   
                }
                gradient_l2_norm = std::sqrt(gradient_l2_norm);
                if(gradient_l2_norm < neuron_unit_max){
                    continue;
                }
                gradient_l2_norm = (neuron_unit_max/gradient_l2_norm);
                #pragma omp simd
                for (int j = 0; j < 16; j++)
                {
                    net_grads[i].bias[j] *= gradient_l2_norm;
                }
                net_grads[i].padding_or_param1 *= gradient_l2_norm;
                #pragma omp simd collapse(2)
                for (int j = 0; j < 9; j++)
                {
                    for (int k = 0; k < 7; k++)
                    {
                        net_grads[i].weights[j][k] *= gradient_l2_norm;
                    }   
                }
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < weight_gradients.size(); i++)
            {
                double gradient_l2_norm = 0;
                #pragma omp simd reduction(+:gradient_l2_norm)
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    gradient_l2_norm+=weight_gradients[i][j]*weight_gradients[i][j];
                }
                gradient_l2_norm = std::sqrt(gradient_l2_norm);
                if(gradient_l2_norm<weight_max){
                    continue;
                }
                gradient_l2_norm = (weight_max/gradient_l2_norm);
                #pragma omp simd
                for (int j = 0; j < weight_gradients[i].size(); j++)
                {
                    weight_gradients[i][j]*=gradient_l2_norm;
                }
            }            
        }

    };
    std::vector<int> input_index;           //indexing recording input neurons
    std::vector<int> output_index;          //indexing recording output neurons
    std::vector<std::vector<int>> layermap;
    unsigned int max_layer_size=0;
    unsigned int max_input_dim=0;
    unsigned int max_output_dim=0;
    std::vector<int> depth;                    // for layermap purposes

    vec_of_arr<int> recurrent_connection;      // taking inspiration from attention the reccurent weights will be generated on the fly
    std::vector<float> rweights;
    std::vector<float> routput;
    void set_recurrent_c(std::vector<int> & weight_index, std::vector<int> & recurrent_in, std::vector<int> & recurrent_to){
        if((weight_index.size()==recurrent_in.size())&&(recurrent_in.size()==recurrent_to.size())){
            recurrent_connection.arr_size=recurrent_in.size();
            recurrent_connection.vec.resize(3*recurrent_connection.arr_size);
            rweights.resize(recurrent_connection.arr_size);
            routput.resize(recurrent_connection.arr_size);
            for(int i = 0; i < recurrent_connection.arr_size; i++){
                recurrent_connection(0,i) = weight_index[i];
                //std::cout<<recurrent_connection(0,i)<<"|"<<weight_index[i]<<std::endl;
            }
            for(int i = 0; i < recurrent_connection.arr_size; i++){
                recurrent_connection(1,i) = recurrent_in[i];
            }
            for(int i = 0; i < recurrent_connection.arr_size; i++){
                recurrent_connection(2,i) = recurrent_to[i];
                neural_net[recurrent_to[i]].isinput=2;
            }
        }
        else{
            std::cout<<"ERROR different length vectors passed to set_recurrent_c()"<<std::endl;
            std::exit(0);
        }
    }
    inline void connect_recurrent(vec_of_arr<float> & states, int tstep){
        if(recurrent_connection.vec.size()==0){}
        else{
            #pragma omp for schedule(static,16) nowait
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                rweights[i] = std::tanh(states(tstep-1,recurrent_connection(0,i)));
            } 
            #pragma omp for schedule(static,16)
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                routput[i] = sigmoid(states(tstep-1,recurrent_connection(1,i)));
            } 
            #pragma omp for schedule(static,16)
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                neural_net[recurrent_connection(2,i)].units[0]=routput[i]*rweights[i];
            }
        }
    }

    inline void connect_recurrent(std::vector<float> & tminus_1){
        if(recurrent_connection.vec.size()==0){}
        else{
            #pragma omp for schedule(static,16) nowait
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                rweights[i] = std::tanh(tminus_1[recurrent_connection(0,i)]);
            } 
            #pragma omp for schedule(static,16)
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                routput[i] = sigmoid(tminus_1[recurrent_connection(1,i)]);
            } 
            #pragma omp for schedule(static,16)
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                neural_net[recurrent_connection(2,i)].units[0]=routput[i]*rweights[i];
            }
        }
    }

    inline void drecurrent_connect(int tstep, vec_of_arr<float> &gradients, vec_of_arr<float> &states){
        if(recurrent_connection.vec.size()==0){}
        else{
            #pragma omp for
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                rweights[i] = std::tanh(states(tstep-1,recurrent_connection(0,i)));
            } 
            #pragma omp for
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                routput[i] = sigmoid(states(tstep-1,recurrent_connection(1,i)));
            } 
            #pragma omp for
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                gradients(tstep-1,recurrent_connection(1,i)) += gradients(tstep,recurrent_connection(2,i)) * rweights[i] * routput[i] * (1-routput[i]);
            }
            #pragma omp for simd
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                rweights[i] *= rweights[i];
                rweights[i] = 1-rweights[i];
                rweights[i] *= routput[i];
            }
            #pragma omp for
            for(int i = 0; i < recurrent_connection.arr_size;i++){
                gradients(tstep-1,recurrent_connection(0,i)) += gradients(tstep,recurrent_connection(2,i)) * rweights[i];
            }
        }
    }

    // sort asecending by index, should always be almost sorted at least thus use insertion sort
    void weight_index_sort(){
        //this layer of parallelization should be enough
        #pragma omp parallel for
        for (int i = 0; i < weights.size(); i++)
        {
            for(int j = 1; j < weights[i].size();j++){
                int_float p = weights[i][j];
                int k = j - 1;
                while( (k >= 0)&&(weights[i][k].x > p.x))
                {
                    weights[i][k+1] = weights[i][k];
                    k--;
                }
                weights[i][k+1] = p;
            }
        }
    }
    
    void update_max_sizes(){
        int maxi = 0;
        for(int i = 0; i < layermap.size(); i++){
            maxi = (maxi>=layermap[i].size()) ? maxi:layermap[i].size();
        }
        max_layer_size=maxi;
        maxi = 0;
        for(int i = 0; i < neural_net.size(); i++){
            maxi = (maxi>=weights[i].size()) ? maxi:weights[i].size();
        }
        max_input_dim = maxi;
        maxi = 0;
        for(int i = 0; i < neural_net.size(); i++){
            maxi = (maxi>=bweights[i].size()) ? maxi:bweights[i].size();
        }
        max_output_dim = maxi;
    }

    void layermap_sync(){
        layermap.clear();
        depth.reserve(neural_net.size());
        depth.resize(neural_net.size());
        std::fill(depth.begin(),depth.end(),0);
        layermap.reserve(neural_net.size());
        layermap.resize(1,{});
        layermap[0].reserve(input_index.size());
        weight_index_sort();
        for (int i = 0; i < input_index.size(); i++)
        {
            layermap[0].emplace_back(input_index[i]);
        }
        for (int i = 0; i < depth.size(); i++)
        {
            if (neural_net[i].isinput)
            {
                continue;
            }
            for (int j = 0; j < weights[i].size(); j++)
            {
                if (weights[i][j].x > i)
                {
                    break;
                }
                depth[i] = std::max(depth[i],depth[weights[i][j].x] + 1);
            }
            if (layermap.size() < depth[i]+1)
            {
                layermap.resize(depth[i]+1,{});
            }
            layermap[depth[i]].emplace_back(i);
        }
    } 
    
    void bweights_sync(){
        depth.reserve(neural_net.size());
        depth.resize(neural_net.size());
        std::fill(depth.begin(),depth.end(),0);
        bweights.resize(neural_net.size());
        for(int i = 0 ; i < neural_net.size(); i++){
            for(int j = 0; j < weights[i].size(); j++){
                depth[weights[i][j].x] += 1;
            }
        }
        for(int i = 0 ; i <neural_net.size(); i++){
            bweights[i].resize(depth[i],int_int(0,0));
        }
        std::fill(depth.begin(),depth.end(),0);
        for(int i = 0 ; i < neural_net.size(); i++){
            for(int j = 0; j < weights[i].size(); j++){
                bweights[weights[i][j].x][depth[weights[i][j].x]].x = i;
                bweights[weights[i][j].x][depth[weights[i][j].x]].y = j;
                depth[weights[i][j].x] += 1;
            }
        }
    }

    NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, float connection_density, float connection_sd, int min_layer_size = 1, int connection_radius = -1)
    :neural_net(size)
    ,input_index(input_neurons)
    ,output_index(output_neurons)
    ,weights(size)
    ,recurrent_connection(3,0)
    {
        for(int i = 0 ; i < size; i++){
            neural_net[i] = neuron_unit();
        }
        std::normal_distribution<float> connection_dist(connection_density, connection_sd);
        for (int i = 0; i < input_index.size(); i++)
        {
            neural_net[input_index[i]].isinput = 1;
        }
        const int crad = connection_radius;
        for (int i = 0; i < size; i++)
        {
            if (neural_net[i].isinput)
            {
                continue;
            }
            float density = connection_dist(ttttt);         //number of connections for each neuron is distributed normally around connection_density with standard deviation connection_sd
            if(density < 0){
                density = 0;
            }
            else if (density >= 1.0f)
            {
                density = 1.0f;
            }
            int connect_n =  std::floor(density * (i));          //this is the number of input connections
            std::set<int> input_i = {};
            for (int j = 0; j < connect_n; j++)
            {
                connection_radius = ((crad>0)&&(crad<i)) ? crad:i;
                int remaining = connection_radius - min_layer_size - weights[i].size();
                if (remaining < 1)
                {
                    break;
                }
                int input_neuron_index = std::floor((remaining) * (custom_dist() - 0.5));
                int wrap_around  = (input_neuron_index < 0) * (remaining + input_neuron_index);
                wrap_around = (wrap_around > input_neuron_index) ? wrap_around:input_neuron_index;
                input_neuron_index = (wrap_around % (remaining)) + (i-connection_radius);
                int pos = 0;
                for(int k = 0; k < weights[i].size(); k++)
                {
                    if (input_neuron_index >= weights[i][k].x)
                    {
                        input_neuron_index++;
                        pos++;
                        continue;
                    }
                    break;
                }
                if (((input_neuron_index-(i - connection_radius))  >= connection_radius))
                {
                    input_neuron_index = ((input_neuron_index-(i - connection_radius)) % connection_radius) + (i - connection_radius);
                }
                weights[i].insert(weights[i].begin() + pos,int_float(input_neuron_index,0.0f));    
            }
        }
        layermap_sync();
        // using this now to store other information will clear at the end    
        bweights_sync();
        for (int i = 0; i < size; i++)
        {   //not really the correct way to use He initialisation but eh
            float input_layer_size = weights[i].size();
            //int recurrent_connections = 0;
            //Xavier initialisation
            for(int j = 0; j < weights[i].size(); j++)
            {
                //if (weights[i][j].x > i)
                //{
                    //weights[i][j].y = 0; /*this seems to stop gradient explosion*/
                //}
                //else{
                float Xavier_init = std::sqrt(2.0f/(input_layer_size + bweights[weights[i][j].x].size()));
                std::normal_distribution<float> weight_init_dist(0,Xavier_init);
                weights[i][j].y = weight_init_dist(ttttt);
                //}
            } 
        }
        update_max_sizes();
        /*
        for(int i = 0 ; i < size; i++){
            float norm = 0;
            for(int j = 0 ; j < weights[i].size();j++){
                norm += weights[i][j].y*weights[i][j].y;
            }
            norm = std::sqrt(1.0f/norm);
            for(int j = 0 ; j < weights[i].size(); j++){
                weights[i][j].y *= norm;
            }
        }*/
    }

    inline void valclear(){
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < neural_net.size(); i++)
        {
            neural_net[i].valclear();
        }
    }
    struct state{
        std::vector<std::array<float,16>> values;
        void valclear(){
            #pragma omp for simd collapse(2)
            for (int i = 0; i < values.size(); i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    values[i][j] = 0;
                }
            }
        }
        state(){}
        state(int size)
        : values(size)
        {
            valclear();
        }
    };
    // post for post activation values
    inline void record_state(state &post){
        #pragma omp for
        for (int i = 0; i < neural_net.size(); i++)
        {
            #pragma omp simd
            for (int j = 0; j < 16; j++)
            {
                post.values[i][j] = neural_net[i].units[j];
            }   
        }
    }
    //from now pre refers to pre Re:Zero and Post past_unit
    //zeros out values for consistency
    
    //for inference so only pass previous state
    template<typename T>
    inline void inf_forward_pass(T &inputs,  std::vector<float> & tminus_1){
        #pragma omp parallel
        {
                //std::vector<float> temp(max_input_dim);
            #pragma omp for schedule(static,4)
            for (int i = 0; i < input_index.size(); i++)
            {
                neural_net[input_index[i]].units[0] = inputs[i];
            }
            connect_recurrent(tminus_1);
            for (int i = 0; i < layermap.size(); i++)
            {
                #pragma omp for schedule(dynamic,16)
                for (int j = 0; j < layermap[i].size(); j++)
                {   
                    neural_net[layermap[i][j]].units[0] = (neural_net[layermap[i][j]].isinput!=0) ? neural_net[layermap[i][j]].units[0]:0.0f;
                    /*
                    for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                    {    
                        temp[l] = states(tstep,weights[layermap[i][j]][l].x);
                    }
                    */
                    ////#pragma omp simd
                    for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                    {               
                        neural_net[layermap[i][j]].units[0] += weights[layermap[i][j]][l].y * tminus_1[weights[layermap[i][j]][l].x];
                    }
                    neural_net[layermap[i][j]].forward_pass(tminus_1[layermap[i][j]]);
                }
                #pragma omp for schedule(static,16)
                for(int j = 0; j < layermap[i].size();j++){
                    tminus_1[layermap[i][j]] = neural_net[layermap[i][j]].units[15];
                }
            }
        }
        return;
    }

    // performant code is ugly code, horrible code duplication with switch to avoid inner loop if statement
    template<typename T>
    inline void forward_pass(T &inputs, state &post, vec_of_arr<float> & states,int tstep){
        if(tstep==0)
        {
            #pragma omp parallel
            {
                //std::vector<float> temp(max_input_dim);
                #pragma omp for schedule(static,4)
                for (int i = 0; i < input_index.size(); i++)
                {
                    neural_net[input_index[i]].units[0] = inputs[i];
                }
                for (int i = 0; i < layermap.size(); i++)
                {
                    #pragma omp for schedule(dynamic,16)
                    for (int j = 0; j < layermap[i].size(); j++)
                    {   
                        /*
                        for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                        {    
                            temp[l] = states(tstep,weights[layermap[i][j]][l].x);
                        }*/
                        neural_net[layermap[i][j]].units[0] = (neural_net[layermap[i][j]].isinput==1) ? neural_net[layermap[i][j]].units[0]:0.0f;
                        ////#pragma omp simd
                        for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                        {               
                            neural_net[layermap[i][j]].units[0] += weights[layermap[i][j]][l].y * states(tstep,weights[layermap[i][j]][l].x);
                        }
                        neural_net[layermap[i][j]].forward_pass(0.0f);
                    }
                    #pragma omp for schedule(static,16)
                    for(int j = 0; j < layermap[i].size();j++){
                        states(tstep,layermap[i][j]) = neural_net[layermap[i][j]].units[15];
                    }
                }
                record_state(post);
            }
        }
        else{
            #pragma omp parallel
            {
                //std::vector<float> temp(max_input_dim);
                #pragma omp for schedule(static,4)
                for (int i = 0; i < input_index.size(); i++)
                {
                    neural_net[input_index[i]].units[0] = inputs[i];
                }
                connect_recurrent(states,tstep);
                for (int i = 0; i < layermap.size(); i++)
                {
                    #pragma omp for schedule(dynamic,16)
                    for (int j = 0; j < layermap[i].size(); j++)
                    {   
                        neural_net[layermap[i][j]].units[0] = (neural_net[layermap[i][j]].isinput!=0) ? neural_net[layermap[i][j]].units[0]:0.0f;
                        /*
                        for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                        {    
                            temp[l] = states(tstep,weights[layermap[i][j]][l].x);
                        }
                        */
                        ////#pragma omp simd
                        for (int l = 0; l < weights[layermap[i][j]].size(); l++)
                        {               
                            neural_net[layermap[i][j]].units[0] += weights[layermap[i][j]][l].y * states(tstep,weights[layermap[i][j]][l].x);
                        }
                        neural_net[layermap[i][j]].forward_pass(states(tstep-1,layermap[i][j]));
                    }
                    #pragma omp for schedule(static,16)
                    for(int j = 0; j < layermap[i].size();j++){
                        states(tstep,layermap[i][j]) = neural_net[layermap[i][j]].units[15];
                    }
                }
                record_state(post);
            }
        }
        return;
    }
    //back propagation through time, passing arguement gradients to avoid memorry allocation
    inline void bptt(vec_of_arr<float> & dloss, std::vector<state> &post, network_gradient &net_grad,vec_of_arr<float> &states, vec_of_arr<float> &gradients){
        #pragma omp parallel
        {
            std::vector<float> temp((max_input_dim > max_output_dim)?max_input_dim:max_output_dim);
            std::vector<float> temp2(max_output_dim);
            #pragma omp for simd
            for(int i = 0 ; i < gradients.vec.size(); i++){
                gradients.vec[i] = 0;
            }
            #pragma omp for schedule(static)
            for (int i = 0; i < dloss.vec_size; i++)
            {
                for (int j = 0; j < dloss.arr_size; j++)
                {
                    gradients(i,output_index[j]) = dloss(i,j);
                    dloss(i,j) = 0;
                }
            }
            for (int i = dloss.vec_size - 1; i > 0; i--)
            {
                for (int j = layermap.size() - 1; j >= 0; j--)
                {   
                    #pragma omp for schedule(dynamic,16) 
                    for (int k = 0; k < layermap[j].size(); k++)
                    {
                        const int & n = layermap[j][k];
                        for (int l = 0; l < bweights[n].size(); l++)
                        {
                            temp[l] = gradients(i,bweights[n][l].x);
                        }
                        for (int l = 0; l < bweights[n].size(); l++)
                        {
                            temp2[l] = weights[bweights[n][l].x][bweights[n][l].y].y;
                        }
                        float reduction_variable = 0;
                        #pragma omp simd reduction(+:reduction_variable)
                        for (int l = 0; l < bweights[n].size(); l++)
                        {
                            reduction_variable += temp[l] * temp2[l];
                        }
                        gradients(i,n) += reduction_variable;
                        gradients(i,n) = neural_net[n].backpropagation(gradients(i,n),post[i].values[n],net_grad.net_grads[n],gradients(i-1,n));   
                    }  
                }
                drecurrent_connect(i,gradients, states);
            }
            for (int j = layermap.size() - 1; j >= 0; j--)
            {
                #pragma omp for schedule(dynamic,16)
                for (int k = 0; k < layermap[j].size(); k++)
                {
                    const int & n = layermap[j][k];
                    for (int l = 0; l < bweights[n].size(); l++)
                    {
                        temp[l] = gradients(0,bweights[n][l].x);
                    }
                    for (int l = 0; l < bweights[n].size(); l++)
                    {
                        temp2[l] = weights[bweights[n][l].x][bweights[n][l].y].y;
                    }
                    float reduction_variable = 0;
                    #pragma omp simd reduction(+:reduction_variable)
                    for (int l = 0; l < bweights[n].size(); l++)
                    {
                        reduction_variable += temp[l] * temp2[l];
                    }
                    gradients(0,n) += reduction_variable;
                    // I didn't overload another fucntion so will just pass reduction_variable as an arguement
                    gradients(0,n) = neural_net[n].backpropagation(gradients(0,n),post[0].values[n],net_grad.net_grads[n],reduction_variable);
                }          
            }
            #pragma omp for
            for(int j = 0; j < weights.size(); j++){
                for(int i = 0 ; i < dloss.vec_size; i++){
                    for(int k = 0; k < weights[j].size(); k++){
                        temp[k] = states(i,weights[j][k].x);
                    }
                    #pragma omp simd
                    for(int k = 0; k < weights[j].size(); k++){
                        net_grad.weight_gradients[j][k] += gradients(i,j) * temp[k];
                    }
                }
            }
        }
    }
    // saving to and loading from a text file
    inline void save_to_txt(std::string file_name){
        std::ofstream file(file_name,std::fstream::trunc);
        file << "number_of_neurons:"<<"\n";
        file << neural_net.size() << "\n";
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
        file << "<recurrent_connection>"<<"\n";
        file << "no_of_connections"<<"\n";
        file << recurrent_connection.arr_size<<"\n";
        for(int i = 0; i < recurrent_connection.arr_size;i++){
            file << recurrent_connection(0,i) << " ";
        }
        file << "\n";
        for(int i = 0; i < recurrent_connection.arr_size;i++){
            file << recurrent_connection(1,i) << " ";
        }
        file << "\n";
        for(int i = 0; i < recurrent_connection.arr_size;i++){
            file << recurrent_connection(2,i) << " ";
        }
        file << "\n";
        file << "</recurrent_connection>" <<"\n";
        file << "<weights>" << "\n";
        for (int i = 0; i < neural_net.size(); i++)
        {
            file << "no_of_weights" << "\n";
            file << weights[i].size() << "\n";
            for (int j = 0; j < weights[i].size(); j++)
            {
                file << weights[i][j].x << " ";
                file << std::hexfloat
                << weights[i][j].y << "\n";
            }
            file << "---------" << "\n";
        }
        file << "</weights>" << std::endl;
        for (int i = 0; i < neural_net.size(); i++)
        {
            file << "<neuron>" << "\n";
            file << "<bias>" << "\n";
            for (int j = 0; j < 16; j++)
            {
                file << std::hexfloat
                << neural_net[i].bias[j] << "\n";
            }
            file << "</bias>" << "\n";

            file << "<param_1>" << "\n";
            file << std::hexfloat
            << neural_net[i].padding_or_param1 << "\n";
            file << "</param_1>" << "\n";
            file << "<a_f>" << "\n";
            file << neural_net[i].a_f << "\n";
            file << "</a_f>" << "\n";
            file << "<nweights>" << "\n";
            for (int j = 0; j < 9; j++)
            {
                for (int k = 0; k < 7; k++)
                {
                    file << std::hexfloat
                    << neural_net[i].weights[j][k] << "\n";
                }
                
            }
            file << "</nweights>" << "\n";
            file << "</neuron>" << "\n";
        }
        file << "<end>" << "\n";
        file.close();
    }

    NN(std::string file_name){
        std::string str_data;
        std::vector<int> output_in;
        std::vector<int> input_in;
        std::ifstream file(file_name);
        if (file.good()){;}else{std::cout<<"ERROR "<<file_name<<" does not exist"<<std::endl; std::exit(EXIT_FAILURE);}
        file >> str_data;
        file >> str_data;
        neural_net.resize(std::stoi(str_data),neuron_unit());
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
        file >> str_data;
        file >> str_data;
        int rec_arrs = std::stoi(str_data);
        recurrent_connection.arr_size = rec_arrs;
        recurrent_connection.vec_size = 3;
        recurrent_connection.vec.resize(rec_arrs*3);
        rweights.resize(recurrent_connection.arr_size);
        routput.resize(recurrent_connection.arr_size);
        for(int i=0;i<rec_arrs;i++){
            std::string data;
            file >> data;
            recurrent_connection(0,i)=std::stoi(data);
        }
        for(int i=0;i<rec_arrs;i++){
            std::string data;
            file >> data;
            recurrent_connection(1,i)=std::stoi(data);
        }
        for(int i=0;i<rec_arrs;i++){
            std::string data;
            file >> data;
            recurrent_connection(2,i)=std::stoi(data);
            neural_net[recurrent_connection(2,i)].isinput=2;
        }
        file >> str_data;
        file >> str_data;
        itr = 0;
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
                weights[itr].emplace_back(int_float(index,value));
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
                    neural_net[itr].bias[i] = b;
                }
                file >> data;
            }else if (data == "<param_1>")
            {    
                file >> data;
                float al = std::stof(data);
                neural_net[itr].padding_or_param1 = al;
                file >> data;
            }else if (data == "<a_f>")
            {
                file >> data;
                int af = std::stoi(data);
                neural_net[itr].a_f = af;
                file >> data;
            }else if (data == "<nweights>")
            {
                for(int i = 0; i < 9; i++){
                    for (int j = 0; j < 7; j++)
                    {
                        file >> data;
                        float we = std::stof(data);
                        neural_net[itr].weights[i][j] = we;
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
            neural_net[input_index[i]].isinput = 1;
        }
        bweights_sync();
        update_max_sizes();
    }

    void cut_recurrent(){
        weight_index_sort();
        for (int i = 0; i < weights.size(); i++)
        {
            for (int j = 0; j < weights[i].size(); j++)
            {
                if (weights[i][j].x > i)
                {
                    weights[i].resize(j,int_float(0,0));
                    break;
                }   
            }   
        }
    }
    // returns true if inputs can appect every output of current timestep
    // false otherwise
    bool connection_check(){
        std::vector<int> checker(neural_net.size(),0);
        for (int i = 0; i < input_index.size(); i++)
        {
            checker[input_index[i]] = 1;
            for (int j = 0; j < layermap.size(); j++)
            {
                for (int k = 0; k < layermap[j].size(); k++)
                {
                    const int & n=layermap[j][k];
                    for(int l = 0; l < weights[n].size(); l++){
                        if (weights[n][l].x > n)
                        {
                            break;
                        }
                        checker[n] = (checker[n]||checker[weights[n][l].x]);
                    }
                }
            }
            for (int j = 0; j < output_index.size(); j++)
            {
                if (!checker[output_index[j]])
                {
                    return false;
                }
                
            }
            std::fill(checker.begin(),checker.end(),0);    
        }
        return true;
    }

    // returns number of trainable parameters
    int parameter_count(){
        int count = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            count += weights[i].size();
        }
        count += neural_net.size() * (16+9*7+1);
        return count;
    }
    
    void new_weights(int m_new_weights, int connection_radius = -1){
        std::uniform_int_distribution<int> dist_to(0, neural_net.size() -1); 
        //float multiplier = 1/std::sqrt(layermap.size()) * 0.5;
        int* new_weights = new int[neural_net.size()];
        int* new_weights_limit = new int[neural_net.size()];
        long long total_available_weights = 0;
        for (int i = 0; i < neural_net.size(); i++)
        {
            new_weights[i] = 0;
        }
        for (int i = 0; i < neural_net.size(); i++)
        {
            int remain = ((connection_radius>0)&&(connection_radius<i)) ? connection_radius:i;
            new_weights_limit[i] = remain - weights[i].size();
            total_available_weights += new_weights_limit[i];
        }
        if (m_new_weights > total_available_weights)
        {
            m_new_weights = total_available_weights;
        }
        for (int i = 0; i < m_new_weights; i++)
        {
            int index_to = dist_to(ttttt);
            if (neural_net[index_to].isinput || (new_weights[index_to] == new_weights_limit[index_to]))
            {
                i--;   
            }
            else{
                new_weights[index_to]++;
            }
        }
        weight_index_sort();
        #pragma omp parallel for
        for (int i = 0; i < neural_net.size(); i++)
        {
            weights[i].reserve(weights[i].size() + new_weights[i]);
            int remain = ((connection_radius>0)&&(connection_radius<i)) ? connection_radius:i;
            for (int j = 0; j < new_weights[i]; j++)
            {
                int index_from = std::floor(zero_to_one(ttttt) * (remain - 1 -weights[i].size()));      //indexing starts at 0
                index_from += (i - remain);
                int pos = 0;
                for(int k = 0; k < weights[i].size(); k++)
                {
                    if (index_from >= weights[i][k].x)
                    {
                        index_from++;
                        pos++;
                        continue;
                    }
                    break;
                }
                if (((index_from-(i - remain)) >= remain))
                {
                    index_from = ((index_from-(i - remain)) % remain) + (i - remain);
                }
                weights[i].insert(weights[i].begin() + pos,int_float(index_from,0));
            }
        }
        delete[] new_weights;
        delete[] new_weights_limit;
        layermap_sync();
        bweights_sync();
    }

    /*
    void mem_reinit_sync(){
        for(int i = 0; i < weights.size(); i++){
            if(neural_net[i].a_f==3){
                for(int j = 0; j < weights[i].size();j++){
                    weights[i][j].y = 0;
                }
            }
        }
    }*/
};

struct NNclone
{
    struct training_essentials{
        vec_of_arr<float> dloss;
        std::vector<NN::state> post;
        NN::network_gradient f;
        vec_of_arr<float> gradients;
        vec_of_arr<float> states;
        training_essentials(NN& n):
            dloss(1,n.output_index.size()),
            post(1,NN::state(n.neural_net.size())),
            f(n),
            gradients(1,n.neural_net.size()),
            states(1,n.neural_net.size()){}
        void resize(int tsteps){
            NN::state ff(gradients.arr_size);
            gradients.resize(tsteps);
            dloss.resize(tsteps);
            post.resize(tsteps,ff);
            states.resize(tsteps);
        }
        void append(){
            resize(gradients.vec_size+1);
        }
    };
    training_essentials gradientsandmore;
    std::vector<neuron_unit_copy> neural_net;
    std::vector<float> rweights;
    std::vector<float> routput;
    NNclone(NN & n)
    :gradientsandmore(n)
    ,neural_net(n.neural_net.size())
    ,rweights(n.neural_net.size(),0)
    ,routput(n.neural_net.size(),0)
    {}
        //for inference so only pass previous state
        inline void record_state(NN::state &post){
            #pragma omp for simd collapse(2)
            for (int i = 0; i < neural_net.size(); i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    post.values[i][j] = neural_net[i].units[j];
                }   
            }
        }
        inline void connect_recurrent(NN &n,vec_of_arr<float> & states, int tstep){
            if(n.recurrent_connection.vec.size()==0){}
            else{
                #pragma omp for nowait
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    rweights[i] = std::tanh(states(tstep-1,n.recurrent_connection(0,i)));
                } 
                #pragma omp for 
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    routput[i] = sigmoid(states(tstep-1,n.recurrent_connection(1,i)));
                } 
                #pragma omp for 
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    neural_net[n.recurrent_connection(2,i)].units[0]=routput[i]*rweights[i];
                }
            }
        }
        inline void connect_recurrent(NN &n,std::vector<float> & tminus_1){
            if(n.recurrent_connection.vec.size()==0){}
            else{
                #pragma omp for nowait
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    rweights[i] = std::tanh(tminus_1[n.recurrent_connection(0,i)]);
                } 
                #pragma omp for
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    routput[i] = sigmoid(tminus_1[n.recurrent_connection(1,i)]);
                } 
                #pragma omp for 
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    neural_net[n.recurrent_connection(2,i)].units[0]=routput[i]*rweights[i];
                }
            }
        }
        inline void drecurrent_connect(NN &n, int tstep, vec_of_arr<float> &gradients, vec_of_arr<float> &states){
            if(n.recurrent_connection.vec.size()==0){}
            else{
                #pragma omp for
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    rweights[i] = std::tanh(states(tstep-1,n.recurrent_connection(0,i)));
                } 
                #pragma omp for
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    routput[i] = sigmoid(states(tstep-1,n.recurrent_connection(1,i)));
                } 
                #pragma omp for
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    gradients(tstep-1,n.recurrent_connection(1,i)) += gradients(tstep,n.recurrent_connection(2,i)) * rweights[i] * routput[i] * (1-routput[i]);
                }
                #pragma omp for simd
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    rweights[i] *= rweights[i];
                    rweights[i] = 1-rweights[i];
                    rweights[i] *= routput[i];
                }
                #pragma omp for
                for(int i = 0; i < n.recurrent_connection.arr_size;i++){
                    gradients(tstep-1,n.recurrent_connection(0,i)) += gradients(tstep,n.recurrent_connection(2,i)) * rweights[i];
                }
            }
        }
        template<typename T>
        inline void inf_forward_pass(NN & n,T &inputs, std::vector<float> & tminus_1){
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < n.input_index.size(); i++)
                {
                    neural_net[n.input_index[i]].units[0] = inputs[i];
                }
                connect_recurrent(n,tminus_1);
                for (int i = 0; i < n.layermap.size(); i++)
                {
                    #pragma omp for
                    for (int j = 0; j < n.layermap[i].size(); j++)
                    {   
                        const int & index = n.layermap[i][j];
                        neural_net[index].units[0] = (n.neural_net[index].isinput!=0) ? neural_net[index].units[0]:0.0f;
                        for (int l = 0; l < n.weights[index].size(); l++)
                        {               
                            neural_net[index].units[0] += n.weights[index][l].y * tminus_1[n.weights[index][l].x];
                        }
                        neural_net[index].forward_pass(n.neural_net[index],tminus_1[index]);
                        tminus_1[index] = neural_net[index].units[15];
                    }
                }
            }
            return;
        }

        // performant code is ugly code, horrible code duplication with switch to avoid inner loop if statement
        template<typename T>
        inline void forward_pass(NN &n,T &inputs, NN::state &post, vec_of_arr<float> & states,int tstep){
            if(tstep==0)
            {
                #pragma omp parallel
                {
                    //std::vector<float> temp(max_input_dim);
                    #pragma omp for
                    for (int i = 0; i < n.input_index.size(); i++)
                    {
                        neural_net[n.input_index[i]].units[0] = inputs[i];
                    }
                    for (int i = 0; i < n.layermap.size(); i++)
                    {
                        #pragma omp for
                        for (int j = 0; j < n.layermap[i].size(); j++)
                        {   
                            const int & index = n.layermap[i][j];
                            neural_net[index].units[0] = (n.neural_net[index].isinput==1) ? neural_net[index].units[0]:0.0f;
                            for (int l = 0; l < n.weights[index].size(); l++)
                            {               
                                neural_net[index].units[0] += n.weights[index][l].y * states(tstep,n.weights[index][l].x);
                            }
                            neural_net[index].forward_pass(n.neural_net[index],0.0f);
                            states(tstep,index) = neural_net[index].units[15];
                        }
                    }
                    record_state(post);
                }
            }
            else{
                #pragma omp parallel
                {
                    //std::vector<float> temp(max_input_dim);
                    #pragma omp for 
                    for (int i = 0; i < n.input_index.size(); i++)
                    {
                        neural_net[n.input_index[i]].units[0] = inputs[i];
                    }
                    connect_recurrent(n,states,tstep);
                    for (int i = 0; i < n.layermap.size(); i++)
                    {
                        #pragma omp for schedule(dynamic,16)
                        for (int j = 0; j < n.layermap[i].size(); j++)
                        {   
                            const int & index = n.layermap[i][j];
                            neural_net[index].units[0] = (n.neural_net[index].isinput!=0) ? neural_net[index].units[0]:0.0f;
                            ////#pragma omp simd
                            for (int l = 0; l < n.weights[index].size(); l++)
                            {               
                                neural_net[index].units[0] += n.weights[index][l].y * states(tstep,n.weights[index][l].x);
                            }
                            neural_net[index].forward_pass(n,states(tstep-1,index));
                            states(tstep,index) = neural_net[index].units[15];
                        }
                    }
                    record_state(post);
                }
            }
            return;
        }

    //back propagation through time, passing arguement gradients to avoid memorry allocation
    inline void bptt(NN & n,vec_of_arr<float> & dloss, std::vector<NN::state> &post, NN::network_gradient &net_grad,vec_of_arr<float> &states, vec_of_arr<float> &gradients){
        #pragma omp parallel
        {
            #pragma omp for simd
            for(int i = 0 ; i < gradients.vec.size(); i++){
                gradients.vec[i] = 0;
            }
            #pragma omp for simd
            for (int i = 0; i < dloss.vec_size; i++)
            {
                for (int j = 0; j < dloss.arr_size; j++)
                {
                    gradients(i,n.output_index[j]) = dloss(i,j);
                }
            }
            #pragma omp for simd
            for(int i = 0 ; i < dloss.vec.size(); i++){
                dloss.vec[i] = 0;
            }
            for (int i = dloss.vec_size - 1; i > 0; i--)
            {
                for (int j = n.layermap.size() - 1; j >= 0; j--)
                {   
                    #pragma omp for 
                    for (int k = 0; k < n.layermap[j].size(); k++)
                    {
                        const int & index = n.layermap[j][k];
                        float reduction_variable = 0;
                        for (int l = 0; l < n.bweights[index].size(); l++)
                        {
                            gradients(i,index) += gradients(i,n.bweights[index][l].x)*n.weights[n.bweights[index][l].x][n.bweights[index][l].y].y;
                        }
                        gradients(i,index) = neural_net[index].backpropagation(n.neural_net[index],gradients(i,index),post[i].values[index],net_grad.net_grads[index],gradients(i-1,index));   
                    }  
                }
                drecurrent_connect(n,i,gradients, states);
            }
            for (int j = n.layermap.size() - 1; j >= 0; j--)
            {
                #pragma omp for schedule(dynamic,16)
                for (int k = 0; k < n.layermap[j].size(); k++)
                {
                    const int & index = n.layermap[j][k];
                    for (int l = 0; l < n.bweights[index].size(); l++)
                    {
                        gradients(0,index) += gradients(0,n.bweights[index][l].x) * n.weights[n.bweights[index][l].x][n.bweights[index][l].y].y;
                    }
                    float nothing;
                    // I didn't overload another fucntion so will just pass a float as an arguement
                    gradients(0,index) = neural_net[index].backpropagation(n.neural_net[index],gradients(0,index),post[0].values[index],net_grad.net_grads[index],nothing);
                }          
            }
            #pragma omp for collapse(2)
            for(int j = 0; j < n.weights.size(); j++){
                for(int i = 0 ; i < dloss.vec_size; i++){
                    for(int k = 0; k < n.weights[j].size(); k++){
                        net_grad.weight_gradients[j][k] += gradients(i,j) * states(i,n.weights[j][k].x);
                    }
                }
            }
        }
    }
};

// higher order weights from output to input ??? or a specific set of neurons


