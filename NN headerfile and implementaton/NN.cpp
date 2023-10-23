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
#include <filesystem>
#include"NN.hpp"


std::random_device rand_dev;                          
std::mt19937 twister(rand_dev());
std::uniform_real_distribution<float> zero_to_one(0.0,1.0);

NN::index_value_pair::index_value_pair(int x, float y)
    :index(x)
    ,value(y)
{}

//constructor for neuron class
NN::neuron::neuron(float init_val, float init_bias, std::vector<index_value_pair> init_weights,std::string act_func)
    :output(init_val)
    ,bias(init_bias)
    ,weights(init_weights)
    ,activation_function_v(1)                                   //defaulting to ReLU
    ,memory(false)
    ,input_neuron(false)
    ,alpha(0)
{
        if(act_func == "ReLU6"){
            activation_function_v = 1;
        }
        else if (act_func == "GELU")
        {
            activation_function_v = 0;
        }
        else{
            std::cout<<"invalid string for activation function, please input \"ReLU6\" or, \"GELU\" only";
            std::cout<<" note no spaces allowed";
            std::exit(EXIT_FAILURE);
        }
}

void NN::layermap_sync()
{
    layermap.clear();
    dependency.reserve(neural_net.size());
    dependency.resize(neural_net.size());
    //std::vector<bool> index_label(neural_net.size(),true);          
    //this is to help label the neurons (not included yet in layermap = true)
    bool* index_label = new bool[neural_net.size()]; 
    std::vector<std::vector<int>> input_tree(neural_net.size());
    for (int i = 0; i < neural_net.size(); i++)
    {
        index_label[i] = true;
    }
    /**/
    for (int i = 0; i < dependency.size(); i++)
    {
        dependency[i].resize(neural_net.size(),false);
        std::fill(dependency[i].begin(),dependency[i].end(),false);
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if (neural_net[i].weights[j].index > i)
            {
                continue;
            }
            dependency[i][neural_net[i].weights[j].index] = true;  //set union may be faster test for use case
        }
    }
    //std::cout<<std::endl;
    std::vector<int> layermap_layer_candidate;
    int initial_neuron = 0;                                         //the neuron to be included into layermap with highest priority at beginning is at index 0, order of neuron index is order of firing
    int mapped_n = 0;
    while(true)                                 
    {
        if (neural_net.size() == 0)
        {
            std::cout<<"ERROR, neural net with 0 neurons"<<std::endl;
            std::exit(EXIT_FAILURE);
        }
                                  
        layermap_layer_candidate.clear();
        layermap_layer_candidate.reserve(neural_net.size() - mapped_n);
        for (int i = initial_neuron; i < neural_net.size(); i++)
        {   
            if(index_label[i])
            {
                bool independent = true;
                for (int j = initial_neuron; j < neural_net.size(); j++)
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
            for (int i = 0; i < neural_net.size(); i++)
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
        for (int i = initial_neuron; i < neural_net.size(); i++)
        {
            if(index_label[i])
            {
                initial_neuron = i;
                break;
            }
            if (i == (neural_net.size() - 1))                       //if all neurons are now in layermap the function has completed
            {
                delete[] index_label;
                return;
            }       
        }
    }
    delete[] index_label;
}

//potentially variance is too large, perhaps have the variance of the weights as an adjustable hyper parameter, I'm not proficient enough at multivariable calc adn linear algebra so IDK how to derive anything suitable
float NN::He_initialisation(int n, float a){
    float w_variance = 2/ (n * (1 + a*a));  //I understand that this is absolutely not the correct use of this initialisation strategy but alas it might work
    return w_variance;
}

float sign_of(float x){
    return((std::signbit(x) * -2) + 1);
}

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

// pdf is f(x) = m(a^2*e^(a^2)) where a(x) = 10(x-0.5), m is a constant approx =11.28379...., in the interval 0 < x < 1
// has a shape with 2 humps around 0.5
float custom_dist(){
    float x = zero_to_one(twister);
    x = 1 - 2*x;
    x = 0.1 * (5 - approx_erfinv(x));
    return x;
}

float quadractic_dist(){
    float x = zero_to_one(twister);
    x = 0.5 * (std::cbrt(2 * x - 1) + 1);  
    return x;   //x will be distributed according to quadractic pdf 12(t - 0.5)^2 between (0,1)     
}

float uniform_dist(){
    float x = zero_to_one(twister);
    return x;
}

float normal_dist(){
    std::normal_distribution<float> dis(0.5,0.1);
    float x = dis(twister);
    x = (x>1) ? 1:x;
    x = (x<0) ? 0:x;
    return x; 
}

//constructor for neural net struct
NN::NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, std::vector<int> memory_neurons, float connection_density, float connection_sd)
    :neural_net(size,neuron(0,0,{},"ReLU6"))
    ,pre_activations(size,0)
    ,t_minus1(size,0)
    ,f_x(size,0)
    ,alpha_g(size,0)
    ,momentumW(size)
    ,momentumB(size,0)
    ,momentumA(size,0)
    ,input_index(input_neurons)
    ,output_index(output_neurons)
    ,memory_index(memory_neurons)
{                         //rng using the standard library's mersenne twister
    std::normal_distribution<float> connection_dist(connection_density, connection_sd);
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net[input_index[i]].input_neuron = true;
    }
    for (int i = 0; i < memory_index.size(); i++)
    {
        neural_net[memory_index[i]].memory = true;
    }
    for (int i = 0; i < size; i++)
    {
        if (neural_net[i].input_neuron)
        {
            continue;
        }
        
        float density = connection_dist(twister);                            //number of connections for each neuron is distributed normally around connection_density with standard deviation connection_sd
        if(density < 0){
            density = 0;
        }
        else if (density > 1)
        {
            density = 1;
        }                                                           //here we truncated off any possible values that are negative or above 1 since that doesn't make any sense, (beware this may result in neurons with no input connections)
        int connect_n =  std::floor(density * (size - 1));          //this is the number of input connections 
        momentumW[i].clear();
        momentumW[i].reserve(connect_n);
        neural_net[i].weights.reserve(connect_n);
        std::set<int> input_i = {};
        for (int j = 0; j < connect_n; j++)
        {
            int remaining = size - 2 - neural_net[i].weights.size();
            int input_neuron_index = std::floor((remaining) * (custom_dist() - 0.5));
            int wrap_around  = (input_neuron_index < 0) * (remaining + input_neuron_index);
            wrap_around = (wrap_around > input_neuron_index) ? wrap_around:input_neuron_index;
            input_neuron_index = (wrap_around % (remaining));
            int pos = 0;
            bool gapped = true;
            for(int k = 0; k < neural_net[i].weights.size(); k++)
            {
                if ((input_neuron_index >= i) && gapped)
                {
                    gapped = false;
                    input_neuron_index++;
                }
                if (input_neuron_index >= neural_net[i].weights[k].index)
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
            neural_net[i].weights.insert(neural_net[i].weights.begin() + pos,index_value_pair(input_neuron_index,0));
            momentumW[i].emplace_back(0);    
        }
    }
    weights_g = momentumW;
    bias_g = momentumB;
    layermap_sync();
    float multiplier = 1.0/std::sqrt(layermap.size());       //I don't fully understand Fixup and how to scale skip connections, hopefully this is adequate
    for (int i = 0; i < size; i++)
    {   //not really the correct way to use He initialisation but eh
        int input_layer_size = neural_net[i].weights.size();
        std::normal_distribution<float> weight_init_dist(0 , sqrt(He_initialisation(input_layer_size)));
        for(int j = 0; j < input_layer_size; j++)
        {
            //if it is a memory neuron to stop things exploding its input weights will all initially be zero, the other weights will break the symmetry, surely?
            if (neural_net[i].memory)
            {
                neural_net[i].weights[j].value = 0;
            }
            else{
                if (neural_net[i].weights[j].index > i)
                {
                    neural_net[i].weights[j].value = 0;
                }
                else{
                    neural_net[i].weights[j].value = weight_init_dist(twister) /*multiplier*/;
                }
            }
        } 
    }
}

NN::NN()
{

}

void NN::neural_net_clear(){
    for (int i = 0; i < neural_net.size(); i++)
    {
        neural_net[i].output = 0;
    } 
    std::fill(t_minus1.begin(),t_minus1.end(),0);
    std::fill(f_x.begin(),f_x.end(),0);
    std::fill(pre_activations.begin(),pre_activations.end(),0);
}

void NN::momentum_clear(){
    for (int i = 0; i < momentumW.size(); i++)
    {
        std::fill(momentumW[i].begin(),momentumW[i].end(),0);
    }
    std::fill(momentumB.begin(),momentumB.end(),0);
    std::fill(momentumA.begin(),momentumA.end(),0);
}

void NN::gradient_clear(){
    std::fill(bias_g.begin(),bias_g.end(),0);
    std::fill(alpha_g.begin(),alpha_g.end(),0);
    for (int i = 0; i < weights_g.size(); i++)
    {
        std::fill(weights_g[i].begin(),weights_g[i].end(),0);
    }
}

float GELU(float pre_activation){
    //approximating near 0 with x*σ(1.702x), where σ is the sigmoid function 1/(e^-x  +  1)
    if (std::abs(pre_activation) < 30)
    {
        double ex = std::exp(-1.702 * pre_activation) + 1;
        ex = 1.0/ex;          
        return (pre_activation * ex);
    }
    else{   //when far from 0 approximate with ReLU, i.e. x>0, f(x) = x; x<0, f(x) = 0
        return pre_activation * (!std::signbit(pre_activation));
    }
}

/*
void NN::neuron_activation(int n, float a){
    for (int i = 0; i < neural_net[n].weights.size(); i++)
    {
        neural_net[n].output += neural_net[n].weights[i].value * neural_net[neural_net[n].weights[i].index].output;         //the previous timestep is maintained, I wonder if this will lead to problems, we'll see IDK how to approach theoretically,
        //the pipe dream is that this will work similiar to LSTM as from what I understand the core essense of LSTM is to record the values when needed and that is regulated by gates
    }
    neural_net[n].output += neural_net[n].bias;
    neural_net[n].activation_function(a); 
}*/

float NN::loss(std::vector<float> & output, std::vector<float> & target, int loss_f){
    switch (loss_f)
    {
    default:    //Mean squared error as default
        float MSE = 0;
        for (int i = 0; i < output.size(); i++)
        {
            float error = output[i] - target[i];
            MSE += (error * error);
        }
        MSE = MSE/output.size();
        return MSE;
        break;
    }
}

float activation_function(float output,const int & activation_function_v,const float & a){
    switch (activation_function_v)
    {
    case 0:
        return GELU(output);
    case 1:
        //ReLU6,, as a hard sigmoid function from -3 to 3
        output = (output > 6) ? 6 /* +  ((output - ) * a))*/:output;
        output = (output > 0) ? output:0/* + (a * (output + 3))*/;
        return output;
    case 2:
        return output;
    case 3:
        return std::tanh(output);
    default:
        output = (output > 0) ? output:a * output;
        return output;
    }   
}

inline float dGELU(float x){
    //near 0 we approximate with the derivative of x*σ(1.702x)
    if(std::abs(x) < 30){
        double ex = std::exp(1.702 * x); 
        double denominator = (ex + 1) * (ex + 1);
        double a = (1.702 * ex * x) + (ex * ex) + ex;
        return (a/denominator);
    }
    else{   //when far from 0 we use the derivative of ReLU, x>0, f'(x) = 1; x<0, f'(x) = 0 
        return !(std::signbit(x));
    }
}


inline float dReLU(float x, float a){
    return (x > 0) ? 1:a;
}

inline float dReLU6(float x){
    x = (x >= 6) ? 0:1;
    x = (x <= 0) ? 0:1;
    return x;
}

float act_func_derivative(float x,const int &  activation_function_v,const float & a){
    switch (activation_function_v)
        {
        case 0:
            return dGELU(x);
            break;
        case 1:
            return (dReLU6(x));
            break;
        case 2:
            return 1;       //linear do nothing, this is for a potential softmax implementation for the output neuron
        case 3:
            return 1 - (std::tanh(x) * std::tanh(x));
        default:
            return (dReLU(x,a));
            break;
        }
}

//uses ReLU capped at 7 hardcoded in, I need all the performance I can get, I just need this thing to work for now
void NN::forward_pass(std::vector<float> &inputs, float a){
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net[input_index[i]].output = inputs[i];
    }
    for (int i = 0; i < layermap.size(); i++)
    {
        for (int j = 0; j < layermap[i].size(); j++)
        {   
            neural_net[layermap[i][j]].output *= (neural_net[layermap[i][j]].memory || neural_net[layermap[i][j]].input_neuron);
            for (int l = 0; l < neural_net[layermap[i][j]].weights.size(); l++)
            {
                const int in_indx = neural_net[layermap[i][j]].weights[l].index;
                const float in = (in_indx > layermap[i][j]) ? t_minus1[in_indx] : neural_net[in_indx].output;
                //apologies for the naming scheme
                neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].weights[l].value * in;
            }
            neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].bias;
            const float pre_activation = neural_net[layermap[i][j]].output;
            const float al = neural_net[layermap[i][j]].alpha;
            neural_net[layermap[i][j]].output = (pre_activation) + (al * activation_function(neural_net[layermap[i][j]].output,neural_net[layermap[i][j]].activation_function_v,a));
        }
    } 
    #pragma omp simd
    for (int i = 0; i < neural_net.size(); i++)
    {
        t_minus1[i] = neural_net[i].output;
    }
}

void NN::forward_pass_s_pa(std::vector<float> &inputs, float a){
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net[input_index[i]].output = inputs[i];
    }
    for (int i = 0; i < layermap.size(); i++)
    {
        for (int j = 0; j < layermap[i].size(); j++)
        {   
            if (neural_net[layermap[i][j]].memory || neural_net[layermap[i][j]].input_neuron)
            {
                neural_net[layermap[i][j]].output = neural_net[layermap[i][j]].output;
            }
            else{
                neural_net[layermap[i][j]].output = 0;
            } 
            for (int l = 0; l < neural_net[layermap[i][j]].weights.size(); l++)
            {
                const int in_indx = neural_net[layermap[i][j]].weights[l].index;
                const float in = (in_indx > layermap[i][j]) ? t_minus1[in_indx] : neural_net[in_indx].output;
                //apologies for the naming scheme
                neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].weights[l].value * in;
            }
            neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].bias;
            pre_activations[layermap[i][j]] = neural_net[layermap[i][j]].output;
            const float al = neural_net[layermap[i][j]].alpha;
            neural_net[layermap[i][j]].output =  (activation_function(neural_net[layermap[i][j]].output,neural_net[layermap[i][j]].activation_function_v,a));
            f_x[layermap[i][j]] = neural_net[layermap[i][j]].output;
            neural_net[layermap[i][j]].output = (pre_activations[layermap[i][j]]) + (al * (f_x[layermap[i][j]]));
        }
    } 
    #pragma omp simd
    for (int i = 0; i < neural_net.size(); i++)
    {
        t_minus1[i] = neural_net[i].output;
    }
}

float standard_normal_pdf(float x){
    const double sqrt1_2pi = 0.3989422804014327;
    return (sqrt1_2pi * std::exp(x * x * -0.5));
}

//back propgation through time, no weight updates, only gradient, timestep is the timestep to start back prorpagating from, forwardpass_pa are the pre activations
void NN::bptt(int timestep, std::vector<std::vector<float>> &forwardpass_states,std::vector<std::vector<float>> &forwardpass_pa,std::vector<std::vector<float>> &forwardpass_fx, std::vector<std::vector<float>> &target_output_loss, float ReLU_leak, float gradient_limit)
{
    weights_gradient.resize(neural_net.size());
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        weights_gradient[i].resize(neural_net[i].weights.size());
        std::fill(weights_gradient[i].begin(),weights_gradient[i].end(),0);
    }
    bias_gradient.resize(neural_net.size());
    std::fill(bias_gradient.begin(),bias_gradient.end(),0);
    neuron_gradient.resize((forwardpass_states.size() >=neuron_gradient.size()) ? forwardpass_states.size():neuron_gradient.size());
    for (int i = 0; i < timestep; i++)
    {
        neuron_gradient[i].resize(neural_net.size());
        std::fill(neuron_gradient[i].begin(),neuron_gradient[i].end(),0);
    }
    alpha_gradient.resize(neural_net.size());
    std::fill(alpha_gradient.begin(),alpha_gradient.end(),0);
    //for loop descending starting from the most recent timestep
    for (int i = timestep - 1; i > 0; i--)
    {
        for(int j = 0; j < target_output_loss[i].size(); j++){
            neuron_gradient[i][output_index[j]] += target_output_loss[i][j];
        }
        for (int j = layermap.size() - 1; j >= 0  ; j--)
        {
            for (int k = 0; k < layermap[j].size(); k++)
            {
                const float  a = neural_net[layermap[j][k]].alpha;
                float dldz = neuron_gradient[i][layermap[j][k]] * (1 + (act_func_derivative(forwardpass_pa[i][layermap[j][k]],neural_net[layermap[j][k]].activation_function_v,ReLU_leak) * (a)));
                //dldz = (std::abs(dldz) > gradient_limit) ? (sign_of(dldz) * gradient_limit):dldz;
                alpha_gradient[layermap[j][k]] += (forwardpass_fx[i][layermap[j][k]]) * neuron_gradient[i][layermap[j][k]];
                bias_gradient[layermap[j][k]] += dldz;
                //surely this won't lead to exploding gradients, right??
                neuron_gradient[i-1][layermap[j][k]] = (neural_net[layermap[j][k]].memory) ? neuron_gradient[i-1][layermap[j][k]] + dldz:neuron_gradient[i-1][layermap[j][k]];
                for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                {
                    if(layermap[j][k] > neural_net[layermap[j][k]].weights[l].index){       //if greater then must be same time step, if less must be previous timestep
                        neuron_gradient[i][neural_net[layermap[j][k]].weights[l].index] += dldz * neural_net[layermap[j][k]].weights[l].value;
                        weights_gradient[layermap[j][k]][l] += dldz * forwardpass_states[i][neural_net[layermap[j][k]].weights[l].index];
                    }
                    else{
                        neuron_gradient[i-1][neural_net[layermap[j][k]].weights[l].index] += dldz * neural_net[layermap[j][k]].weights[l].value;
                        weights_gradient[layermap[j][k]][l] += dldz * forwardpass_states[i-1][neural_net[layermap[j][k]].weights[l].index];
                    }
                }
            }  
        }
    }
    for(int i = 0; i < target_output_loss[0].size(); i++){
        neuron_gradient[0][output_index[i]] += target_output_loss[0][i];
    }
    for (int j = layermap.size() - 1; j >= 0 ; j--)
    {
        for (int k = 0; k < layermap[j].size(); k++)
        {
            const float a = neural_net[layermap[j][k]].alpha;
            float dldz = neuron_gradient[0][layermap[j][k]] * (1 + (act_func_derivative(forwardpass_pa[0][layermap[j][k]],neural_net[layermap[j][k]].activation_function_v,ReLU_leak) * (a)));
            //dldz = (std::abs(dldz) > gradient_limit) ? (sign_of(dldz) * gradient_limit):dldz;
            alpha_gradient[layermap[j][k]] += (forwardpass_fx[0][layermap[j][k]]) * neuron_gradient[0][layermap[j][k]];
            bias_gradient[layermap[j][k]] += dldz;
            for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                {
                if(layermap[j][k] > neural_net[layermap[j][k]].weights[l].index){       //if greater then must be same time step, if less must be previous timestep
                    neuron_gradient[0][neural_net[layermap[j][k]].weights[l].index] += dldz * neural_net[layermap[j][k]].weights[l].value;        
                    weights_gradient[layermap[j][k]][l] += dldz * forwardpass_states[0][neural_net[layermap[j][k]].weights[l].index];
                }
                else{
                    weights_gradient[layermap[j][k]][l] += dldz * forwardpass_states[0][neural_net[layermap[j][k]].weights[l].index];
                }
            }
        }    
    }

    for (int i = 0; i < bias_gradient.size(); i++)
    {
        if (std::isnan(bias_gradient[i]) || std::isinf(bias_gradient[i]) || (std::abs(bias_gradient[i]) > gradient_limit))
        {
            bias_gradient[i] = sign_of(bias_gradient[i]) * gradient_limit;
        }
        bias_g[i] += bias_gradient[i];  
    }

    for (int i = 0; i < alpha_g.size(); i++)
    {
        if (std::isnan(alpha_gradient[i]) || std::isinf(alpha_gradient[i]) || (std::abs(alpha_gradient[i]) > gradient_limit))
        {
            alpha_gradient[i] = sign_of(alpha_gradient[i]) * gradient_limit;
        }
        alpha_g[i] += alpha_gradient[i]; 
    }

    for (int i = 0; i < weights_gradient.size(); i++)
    {
        for (int j = 0; j < weights_gradient[i].size(); j++)
        {
            if (std::isnan(weights_gradient[i][j]) || std::isinf(weights_gradient[i][j]) || (std::abs(weights_gradient[i][j]) > gradient_limit))
            {
                weights_gradient[i][j] = sign_of(weights_gradient[i][j]) * gradient_limit;  
            }
            weights_g[i][j] += weights_gradient[i][j];      
        }
    }
}

void NN::update_momentum(float momentum){
    #pragma omp simd
    for (int i = 0; i < momentumB.size(); i++)
    {
        momentumB[i] = (momentumB[i] * momentum) + (bias_g[i] * (1 - momentum));
    }
    for (int i = 0; i < momentumW.size(); i++)
    {
        #pragma omp simd
        for (int j = 0; j < momentumW[i].size(); j++)
        {
            momentumW[i][j] = (momentumW[i][j] * momentum) + (weights_g[i][j] * (1 - momentum));
        }
    }
    #pragma omp simd
    for (int i = 0; i < momentumA.size(); i++)
    {
        momentumA[i] = (momentumA[i] * momentum) + (alpha_g[i] * (1 - momentum));
    }
}

void NN::update_parameters(float learning_rate, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i] || neural_net[i].memory)
        {
            continue;
        }
        neural_net[i].bias -= learning_rate * momentumB[i];
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        neural_net[i].alpha -= learning_rate * momentumA[i];
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        #pragma omp simd
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value -= learning_rate * momentumW[i][j];
        }        
    }
}

//weights = weights - h_param * signof(weights)
void NN::l1_reg(float h_param, std::vector<bool> freeze_neuron){
    if(freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value -= h_param * sign_of(neural_net[i].weights[j].value);
            neural_net[i].weights[j].value = (std::abs(neural_net[i].weights[j].value) > h_param) ? neural_net[i].weights[j].value:0; 
        }
        
    }
    
}

//weights -= weights * w_decay
void NN::l2_reg(float w_decay, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value -= neural_net[i].weights[j].value * w_decay;
        }
        
    }
    
}

void NN::weight_noise(float sigma, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    std::normal_distribution<float> noise(0,sigma);
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value += noise(twister);
        }
        
    }
    
}

//the new weights will be initialised to 0, also resets momentum to 0, uniform distribution, we assumed weights are already nearly sorted in ascending index (bubble sort is called)
void NN::new_weights_s(int m_new_weights, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
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
        new_weights_limit[i] = neural_net.size() - 1 - neural_net[i].weights.size();
        if (neural_net[i].input_neuron || freeze_neuron[i])
        {
            continue;
        }
        
        total_available_weights += new_weights_limit[i];
    }
    if (m_new_weights > total_available_weights)
    {
        m_new_weights = total_available_weights;
    }
    for (int i = 0; i < m_new_weights; i++)
    {
        int index_to = dist_to(twister);
        if (neural_net[index_to].input_neuron || freeze_neuron[index_to] || (new_weights[index_to] == new_weights_limit[index_to]))
        {
            i--;   
        }
        else{
            new_weights[index_to]++;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < neural_net.size(); i++)
    {
        for (int j = 1; j < neural_net[i].weights.size(); j++)
        {
            if (neural_net[i].weights[j-1].index > neural_net[i].weights[j].index)
            {
                std::swap(neural_net[i].weights[j-1],neural_net[i].weights[j]);
            }
        }
        
    }
    
    #pragma omp parallel for
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        neural_net[i].weights.reserve(neural_net[i].weights.size() + new_weights[i]);
        momentumW[i].reserve(neural_net[i].weights.size() + new_weights[i]);
        for (int j = 0; j < new_weights[i]; j++)
        {
            int index_from = std::floor(zero_to_one(twister) * (neural_net.size() -2 - neural_net[i].weights.size()));      //indexing starts at 0
            int pos = 0;
            bool gapped = true;
            for(int k = 0; k < neural_net[i].weights.size(); k++)
            {
                if ((index_from >= i) && gapped)
                {
                    gapped = false;
                    index_from++;
                }
                if (index_from >= neural_net[i].weights[k].index)
                {
                    index_from++;
                    pos++;
                    continue;
                }
                if ((index_from >= i) && gapped)
                {
                    gapped = false;
                    index_from++;
                    continue;
                }
                break;
            }
            if ((index_from >= i) && gapped)
            {
                index_from++;
            }
            neural_net[i].weights.insert(neural_net[i].weights.begin() + pos,index_value_pair(index_from,0));
            momentumW[i].emplace_back(0);
            weights_g[i].emplace_back(0);
        }
    }
    delete[] new_weights;
    delete[] new_weights_limit;
    momentum_clear();
    layermap_sync();
}

//initialise m_new_weights number of new recurrent weights to 0
void NN::new_rweights_s(int m_new_weights, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    std::uniform_int_distribution<int> dist_to(0, neural_net.size() - 1); 
    //float multiplier = 1/std::sqrt(layermap.size()) * 0.5;
    int* new_weights = new int[neural_net.size()];
    int* new_weights_limit = new int[neural_net.size()];
    int* rsize = new int[neural_net.size()];
    long long total_available_weights = 0;

    bool swappos = true;
    while (swappos)
    {
        swappos = false;
        #pragma omp parallel for
        for (int i = 0; i < neural_net.size(); i++)
        {
            for (int j = 1; j < neural_net[i].weights.size(); j++)
            {
                if (neural_net[i].weights[j-1].index > neural_net[i].weights[j].index)
                {
                    std::swap(neural_net[i].weights[j-1],neural_net[i].weights[j]);
                    swappos = true;
                }
            }
            
        }   
    }

    for (int i = 0; i < neural_net.size(); i++)
    {
        new_weights[i] = 0;
        rsize[i] = 0;
        for(int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if (neural_net[i].weights[j].index>i)
            {
               rsize[i] = neural_net[i].weights.size() - j;
               break;
            }
        }
    }

    for (int i = 0; i < neural_net.size(); i++)
    {
        new_weights_limit[i] = neural_net.size() - 1 - rsize[i] - i;
        if (neural_net[i].input_neuron || freeze_neuron[i])
        {
            continue;
        }
        
        total_available_weights += new_weights_limit[i];
    }
    if (m_new_weights > total_available_weights)
    {
        m_new_weights = total_available_weights;
    }
    for (int i = 0; i < m_new_weights; i++)
    {
        int index_to = dist_to(twister);
        if (neural_net[index_to].input_neuron || freeze_neuron[index_to] || (new_weights[index_to] == new_weights_limit[index_to]))
        {
            i--;   
        }
        else{
            new_weights[index_to]++;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        neural_net[i].weights.reserve(neural_net[i].weights.size() + new_weights[i]);
        momentumW[i].reserve(neural_net[i].weights.size() + new_weights[i]);
        for (int j = 0; j < new_weights[i]; j++)
        {
            int og;
            int index_from = i + std::floor(zero_to_one(twister) * (neural_net.size() - 2 - rsize[i] - i));
            og = index_from;
            int pos = neural_net[i].weights.size() - rsize[i];
            if (index_from == i)
            {
                index_from++;
            }
            for(int k = (neural_net[i].weights.size() - rsize[i]); k < neural_net[i].weights.size(); k++)
            {
                if (index_from >= neural_net[i].weights[k].index)
                {
                    index_from++;
                    pos++;
                    continue;
                }
                break;
            }
            neural_net[i].weights.insert(neural_net[i].weights.begin() + pos,index_value_pair(index_from,0));
            momentumW[i].emplace_back(0);
            weights_g[i].emplace_back(0);
            rsize[i]++;
        }
    }
    delete[] new_weights;
    delete[] new_weights_limit;
    delete[] rsize;
    momentum_clear();
    layermap_sync();
}

//initialise m_new_weights number of new weights to the "memory" neurons
void NN::new_imweights_s(int m_new_weights, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    std::uniform_int_distribution<int> dist_to(0, memory_index.size() - 1); 
    //float multiplier = 1/std::sqrt(layermap.size()) * 0.5;
    int* new_weights = new int[memory_index.size()];
    int* new_weights_limit = new int[memory_index.size()];
    long long total_available_weights = 0;
    for (int i = 0; i < memory_index.size(); i++)
    {
        new_weights[i] = 0;
    }
    for (int i = 0; i < memory_index.size(); i++)
    {
        new_weights_limit[i] = neural_net.size() - 1 - neural_net[memory_index[i]].weights.size();
        if (neural_net[memory_index[i]].input_neuron || freeze_neuron[memory_index[i]])
        {
            continue;
        }
        
        total_available_weights += new_weights_limit[i];
    }
    if (m_new_weights > total_available_weights)
    {
        m_new_weights = total_available_weights;
    }
    for (int i = 0; i < m_new_weights; i++)
    {
        int index_to = dist_to(twister);
        if (neural_net[memory_index[index_to]].input_neuron || freeze_neuron[memory_index[index_to]] || (new_weights[index_to] == new_weights_limit[index_to]))
        {
            i--;   
        }
        else{
            new_weights[index_to]++;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < neural_net.size(); i++)
    {
        for (int j = 1; j < neural_net[i].weights.size(); j++)
        {
            if (neural_net[i].weights[j-1].index > neural_net[i].weights[j].index)
            {
                std::swap(neural_net[i].weights[j-1],neural_net[i].weights[j]);
            }
        }
        
    }
    
    #pragma omp parallel for
    for (int i = 0; i < memory_index.size(); i++)
    {
        if (freeze_neuron[memory_index[i]])
        {
            continue;
        }
        neural_net[memory_index[i]].weights.reserve(neural_net[memory_index[i]].weights.size() + new_weights[i]);
        momentumW[memory_index[i]].reserve(neural_net[memory_index[i]].weights.size() + new_weights[i]);
        for (int j = 0; j < new_weights[i]; j++)
        {
            int index_from = std::floor(zero_to_one(twister) * (neural_net.size() -2 - neural_net[memory_index[i]].weights.size()));      //indexing starts at 0
            int pos = 0;
            bool gapped = true;
            for(int k = 0; k < neural_net[memory_index[i]].weights.size(); k++)
            {
                if ((index_from >= memory_index[i]) && gapped)
                {
                    gapped = false;
                    index_from++;
                }
                if (index_from >= neural_net[memory_index[i]].weights[k].index)
                {
                    index_from++;
                    pos++;
                    continue;
                }
                if ((index_from >= memory_index[i]) && gapped)
                {
                    gapped = false;
                    index_from++;
                    continue;
                }
                break;
            }
            if ((index_from >= memory_index[i]) && gapped)
            {
                index_from++;
            }
            
            neural_net[memory_index[i]].weights.insert(neural_net[memory_index[i]].weights.begin() + pos,index_value_pair(index_from,0));
            momentumW[memory_index[i]].emplace_back(0);
            weights_g[memory_index[i]].emplace_back(0);
        }
    }
    delete[] new_weights;
    delete[] new_weights_limit;
    momentum_clear();
    layermap_sync();
}

//initialise m_new_weights number of new from the "memory" neurons
void NN::new_omweights_s(int m_new_weights, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    std::uniform_int_distribution<int> dist_to(0, neural_net.size() -1); 
    //float multiplier = 1/std::sqrt(layermap.size()) * 0.5;
    int* new_weights = new int[neural_net.size()];
    int* new_weights_limit = new int[neural_net.size()];
    int* msize = new int[neural_net.size()];
    long long total_available_weights = 0;
        //sort the weights
    bool swappos = true;
    while (swappos)
    {
        swappos = false;
        #pragma omp parallel for
        for (int i = 0; i < neural_net.size(); i++)
        {
            for (int j = 1; j < neural_net[i].weights.size(); j++)
            {
                if (neural_net[i].weights[j-1].index > neural_net[i].weights[j].index)
                {
                    std::swap(neural_net[i].weights[j-1],neural_net[i].weights[j]);
                    swappos = true;
                }
            }
            
        }   
    }

    for (int i = 0; i < neural_net.size(); i++)
    {
        new_weights[i] = 0;
        msize[i] = 0;
        for(int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if (neural_net[neural_net[i].weights[j].index].memory)
            {
               msize[i]++;
            }
        }
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        msize[i] = (neural_net[i].memory) ? (msize[i] + 1):msize[i] ;
        new_weights_limit[i] = memory_index.size() - msize[i];
        if (neural_net[i].input_neuron || freeze_neuron[i])
        {
            continue;
        }
        
        total_available_weights += new_weights_limit[i];
    }
    if (m_new_weights > total_available_weights)
    {
        m_new_weights = total_available_weights;
    }
    //can be a very inefficient loop as neural netowrk approaches full connectivity
    for (int i = 0; i < m_new_weights; i++)
    {
        int index_to = dist_to(twister);
        if (neural_net[index_to].input_neuron || freeze_neuron[index_to] || (new_weights[index_to] == new_weights_limit[index_to]))
        {
            i--;   
        }
        else{
            new_weights[index_to]++;
        }
    }
    //#pragma omp parallel for
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        neural_net[i].weights.reserve(neural_net[i].weights.size() + new_weights[i]);
        momentumW[i].reserve(neural_net[i].weights.size() + new_weights[i]);
        for (int j = 0; j < new_weights[i]; j++)
        {
            int index_from = std::floor(zero_to_one(twister) * (memory_index.size() - msize[i] - 1));
            int pos = 0;
            for(int k = 0; k < neural_net[i].weights.size(); k++)
            {
                if (memory_index[index_from] == i)
                {
                    index_from++;
                }
                if (neural_net[i].weights[k].index == memory_index[index_from])
                {
                    index_from++;
                }
                if (memory_index[index_from] == i)
                {
                    index_from++;
                }
            }
            index_from = memory_index[index_from];
            for (int k = 0; k < neural_net[i].weights.size(); k++)
            {
                if (index_from > neural_net[i].weights[j].index)
                {
                    pos++;
                }
                
            }
            
            neural_net[i].weights.insert(neural_net[i].weights.begin() + pos,index_value_pair(index_from,0));
            momentumW[i].emplace_back(0);
            weights_g[i].emplace_back(0);
            msize[i]++;
        }
    }
    delete[] new_weights;
    delete[] new_weights_limit;
    delete[] msize;
    momentum_clear();
    layermap_sync();
}




//if below weights_cutoff, weights are removed, also reset momentum to 0 
void NN::prune_weights(float weights_cutoff, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    for (int i = 0; i < neural_net.size(); i++)
    {
        if (freeze_neuron[i])
        {
            continue;
        }
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if(std::abs(neural_net[i].weights[j].value) < weights_cutoff){
                neural_net[i].weights.erase(neural_net[i].weights.begin() + j);
                momentumW[i].pop_back();
                weights_g[i].pop_back();
                j--;
            }       
        }      
    }
    layermap_sync();
    momentum_clear();
}

//Warning momentum is not saved
//and yes I am, trying to serialise and deserialise myself, I didn't use boost because it doesn't come with gcc, and I don't want to go through the hassle of installing it
void NN::save(std::string file_name){
    std::ofstream file(file_name,std::fstream::trunc);
    file << "number_of_neurons:" << "\n";
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
    file << "memory_index" << "\n";
    for (int i = 0; i < memory_index.size(); i++)
    {
        file << memory_index[i] << " ";
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
    file << "<neurons>" <<"\n";
    for (int i = 0; i < neural_net.size(); i++)
    {
        file << "neuron" << "\n";
        file << "bias" << "\n";
        file << neural_net[i].bias << "\n";
        file << "alpha" << "\n";
        file << neural_net[i].alpha << "\n";
        file << "<weights>" << "\n";
        file << "no_of_weights"<< "\n";
        file << neural_net[i].weights.size() << "\n";
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            file << neural_net[i].weights[j].index << " ";
            file << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) 
            << neural_net[i].weights[j].value << "\n"; 
        }
        file << "</weights>" << "\n";
        file << "activation_function_v" << "\n";
        file << neural_net[i].activation_function_v << "\n";
    }
    file << "</neurons>" << "\n";
    file.close();
}

bool good_file(std::string file_name){
    std::ifstream file(file_name);
    return file.good();
}

NN::NN(std::string file_name)
    :neural_net()
    ,momentumW()
    ,momentumB()
    ,input_index()
    ,layermap()
    ,output_index()     
{
    std::string str_data;
    std::vector<int> output_in;
    std::vector<int> input_in;
    std::ifstream file(file_name);
    if (file.good()){;}else{std::cout<<"ERROR "<<file_name<<" does not exist"<<std::endl; std::exit(EXIT_FAILURE);}
    file >> str_data;
    file >> str_data;
    neural_net.resize(std::stoi(str_data),neuron(0,0,{},"ReLU6"));
    pre_activations.resize(std::stoi(str_data),0);
    t_minus1.resize(std::stoi(str_data),0);
    alpha_g.resize(std::stoi(str_data),0);
    momentumA.resize(std::stoi(str_data),0);
    f_x.resize(std::stoi(str_data),0);
    for (int i = 0; i < neural_net.size(); i++)
    {
        neural_net[i].alpha = 1;
    }
    
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
        if(data == "memory_index"){
            break;
        }
        output_index.emplace_back(std::stoi(data));
    }
    while(true){
        std::string data;
        file >> data;
        if(data == "number_of_layers"){
            break;
        }
        memory_index.emplace_back(std::stoi(data));
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
    while (true)
    {
        std::string data;
        file >> data;
        if (data == "</neurons>")
        {
            break;
        }
        else if (data == "neuron")
        {
            while(true)
            {
                file >> data;
                if(data == "bias"){
                    file >> data;   //bias value
                    neural_net[itr].bias = std::stof(data);
                }
                else if (data == "alpha")
                {
                    file >> data;   //alpha value
                    neural_net[itr].alpha = std::stof(data);
                }
                else if (data == "no_of_weights")
                {
                    file >> data;   //no_of weights value
                    neural_net[itr].weights.reserve(std::stoi(data));
                }
                else if (data == "<weights>")
                {
                    ;
                } 
                else if(data == "</weights>"){
                    break;
                }
                else{
                    int index = std::stoi(data);
                    file >> data;
                    float value= std::stof(data);
                    neural_net[itr].weights.emplace_back(index_value_pair(index,value));
                }
            }
            file >> data;
            file >> data;
            neural_net[itr].activation_function_v = std::stoi(data);
            itr ++;
        }
        else{
            std::cout<< "ERROR: something wrong with file";
            std::exit(EXIT_FAILURE);
        }
    }
    file.close();
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net[input_index[i]].input_neuron = true;
    }
    for (int i = 0; i < memory_index.size(); i++)
    {
        neural_net[memory_index[i]].memory = true;
    }
    
    momentumB.resize(neural_net.size(),0);
    momentumW.resize(neural_net.size(),{});
    for (int i = 0; i < momentumW.size(); i++)
    {
        momentumW[i].reserve(neural_net[i].weights.size());
        momentumW[i].resize(neural_net[i].weights.size(),0);
    }
    bias_g = momentumB;
    weights_g = momentumW;
}

void NN::sleep(){

}

bool NN::input_connection_check(bool outputc, bool memoryc, bool rc){
    neural_net_clear();
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net[input_index[i]].output = 1;
        for (int j = 0; j < layermap.size(); j++)
        {
            for (int k = 0; k < layermap[j].size(); k++)
            {
                for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                {
                    if(neural_net[neural_net[layermap[j][k]].weights[l].index].output != 0){
                        neural_net[layermap[j][k]].output = 1;
                        break;
                    }    //just to see if input will influence all outputs, easiest way to do this no activation function and set all weights to +1
                }
            }
        }
        //2 forward passes to see if a recurrent connection has effect
        if (rc)
        {
            for (int j = 0; j < layermap.size(); j++)
            {
                for (int k = 0; k < layermap[j].size(); k++)
                {
                    for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                    {
                        if(neural_net[neural_net[layermap[j][k]].weights[l].index].output != 0){
                        neural_net[layermap[j][k]].output = 1;
                        break;
                        }
                    }
                }
            }
        }
        if (outputc)
        {
            for (int m = 0; m < output_index.size(); m++)
            {
                if (neural_net[output_index[m]].output == 0)
                {
                    neural_net_clear();
                    return false;
                }
            }
        }
        if (memoryc)
        {
            for (int m = 0; m < memory_index.size(); m++)
            {
                if (neural_net[memory_index[m]].output == 0)
                {
                    neural_net_clear();
                    return false;
                }
                
            }
        }
        neural_net_clear();
    }
    return true;
}

bool NN::memory_connection_check(bool rc)
{
    neural_net_clear();
    for (int i = 0; i < memory_index.size(); i++)
    {
        neural_net[memory_index[i]].output = 1;
        for (int j = 0; j < layermap.size(); j++)
        {
            for (int k = 0; k < layermap[j].size(); k++)
            {
                for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                {
                        if(neural_net[neural_net[layermap[j][k]].weights[l].index].output != 0){
                        neural_net[layermap[j][k]].output = 1;
                        break;
                    } 
                }
            }
        }
        //2 forward passes to see if a recurrent connection has effect
        if (rc)
        {
            for (int j = 0; j < layermap.size(); j++)
            {
                for (int k = 0; k < layermap[j].size(); k++)
                {
                    for (int l = 0; l < neural_net[layermap[j][k]].weights.size(); l++)
                    {
                        if(neural_net[neural_net[layermap[j][k]].weights[l].index].output != 0){
                            neural_net[layermap[j][k]].output = 1;
                            break;
                        }  
                    }
                }
            }
        }
        for (int m = 0; m < output_index.size(); m++)
        {
            if (neural_net[output_index[m]].output == 0)
            {
                neural_net_clear();
                return false;
            }
        }
        neural_net_clear();
    }
    return true;
}

void NN::ensure_connection(int c_step_size ,bool io, bool im, bool mo, bool rc){
    if (io || im)
    {
        while (!input_connection_check(io,im,rc))
        {
            new_weights_s(c_step_size);
        }
    }
    if (mo)
    {
        while (!memory_connection_check(rc))
        {
            new_weights_s(c_step_size);
        }
    }
}

/*
void NN::depth_check(){
    int min_depth = 0;
    int max_depth = 0;
    int avg_depth = 0;
    int denominator = input_index.size() * output_index.size();
    for (int i = 0; i < input_index.size(); i++)
    {
        neural_net_clear();
        neural_net[input_index[i]].output = 1;
        for (int j = 0; j < neural_net.size(); j++)
        {
            for (int k = 0; k < neural_net[j].weights.size(); k++)
            {
                if(neural_net[j].weights[k].index > j){continue;}
                neural_net[j].output = (neural_net[j].output > neural_net[neural_net[j].weights[k].index].output) ? (neural_net[j].output):(neural_net[neural_net[j].weights[k].index].output + 1);
            }
        }
        if (i == 0)
        {
            min_depth = neural_net[output_index[0]].output;
            max_depth = neural_net[output_index[0]].output;
        }
        for (int j = 0; j < output_index.size(); j++)
        {
            min_depth = (min_depth < neural_net[output_index[j]].output) ? min_depth:neural_net[output_index[j]].output;
            max_depth = (max_depth > neural_net[output_index[j]].output) ? max_depth:neural_net[output_index[j]].output;
            avg_depth += neural_net[output_index[j]].output;
        }
    }
    std::cout<<"at least "<<min_depth<<" \"layers\""<<std::endl;
    std::cout<<"at most "<<max_depth<<" \"layers\""<<std::endl;
    std::cout<<"on average "<<avg_depth/denominator<<" \"layers\""<<std::endl;
}
*/






NNclone::NNclone(){}

NNclone::NNclone(const NN &cloned)
:neuron_states(cloned.neural_net.size(),0)
,pre_activations(cloned.neural_net.size(),0)
,f_x(cloned.neural_net.size(),0)
,t_minus1(cloned.neural_net.size(),0)
,weights_g(cloned.neural_net.size())
,bias_g(cloned.neural_net.size(),0)
,alpha_g(cloned.neural_net.size(),0)
,neuron_gradient(0)
,weights_gradient(cloned.neural_net.size())
,bias_gradient(cloned.neural_net.size(),0)
,alpha_gradient(cloned.neural_net.size(),0)
{
    for (int i = 0; i < cloned.neural_net.size(); i++)
    {
        weights_g[i].resize(cloned.neural_net[i].weights.size(),0);
        weights_gradient[i].resize(cloned.neural_net[i].weights.size(),0);
    }
}

void NNclone::neural_net_clear(){
    std::fill(neuron_states.begin(),neuron_states.end(),0);
    std::fill(t_minus1.begin(),t_minus1.end(),0);
    std::fill(pre_activations.begin(),pre_activations.end(),0);
    std::fill(f_x.begin(),f_x.end(),0);
}

void NNclone::gradient_clear(){
    std::fill(bias_g.begin(),bias_g.end(),0);
    std::fill(alpha_g.begin(),alpha_g.end(),0);
    for (int i = 0; i < weights_g.size(); i++)
    {
        std::fill(weights_g[i].begin(),weights_g[i].end(),0);
    }
}

void NNclone::act_func_derivatives_clear(){
    std::fill(pre_activations.begin(),pre_activations.end(),0);
}

void NNclone::forward_pass(const NN &cloned, std::vector<float> &inputs, float a){
    for (int i = 0; i < cloned.input_index.size(); i++)
    {
        neuron_states[cloned.input_index[i]] = inputs[i];
    }
    for (int i = 0; i < cloned.layermap.size(); i++)
    {
        for (int j = 0; j < cloned.layermap[i].size(); j++)
        {   
            neuron_states[cloned.layermap[i][j]] *= (cloned.neural_net[cloned.layermap[i][j]].memory || cloned.neural_net[cloned.layermap[i][j]].input_neuron);
            #pragma omp simd
            for (int l = 0; l < cloned.neural_net[cloned.layermap[i][j]].weights.size(); l++)
            {
                const int in_indx = cloned.neural_net[cloned.layermap[i][j]].weights[l].index;
                const float in  = (in_indx>cloned.layermap[i][j]) ? t_minus1[in_indx]:neuron_states[in_indx];
                //apologies for the naming scheme
                neuron_states[cloned.layermap[i][j]] += cloned.neural_net[cloned.layermap[i][j]].weights[l].value * in;
            }
            neuron_states[cloned.layermap[i][j]] += cloned.neural_net[cloned.layermap[i][j]].bias;
            const float pre_activation = neuron_states[cloned.layermap[i][j]];
            const float al = cloned.neural_net[cloned.layermap[i][j]].alpha;
            neuron_states[cloned.layermap[i][j]] = pre_activation + (al * activation_function(pre_activation,cloned.neural_net[cloned.layermap[i][j]].activation_function_v,a));
        }
    } 
    #pragma omp simd
    for (int i = 0; i < neuron_states.size(); i++)
    {
        t_minus1[i] = neuron_states[i];
    }
}

void NNclone::forward_pass_s_pa(const NN &cloned, std::vector<float> &inputs, float a){
    for (int i = 0; i < cloned.input_index.size(); i++)
    {
        neuron_states[cloned.input_index[i]] = inputs[i];
    }
    std::vector<float> pre_norm_mmean;
    for (int i = 0; i < cloned.layermap.size(); i++)
    {
        float memory_count = 0;
        pre_norm_mmean.resize(cloned.layermap[i].size());
        for (int j = 0; j < cloned.layermap[i].size(); j++)
        {   
            neuron_states[cloned.layermap[i][j]] *= (cloned.neural_net[cloned.layermap[i][j]].memory || cloned.neural_net[cloned.layermap[i][j]].input_neuron);
            #pragma omp simd
            for (int l = 0; l < cloned.neural_net[cloned.layermap[i][j]].weights.size(); l++)
            {
                const int in_indx = cloned.neural_net[cloned.layermap[i][j]].weights[l].index;
                const float in  = (in_indx>cloned.layermap[i][j]) ? t_minus1[in_indx]:neuron_states[in_indx];
                //apologies for the naming scheme
                neuron_states[cloned.layermap[i][j]] += cloned.neural_net[cloned.layermap[i][j]].weights[l].value * in;
            }
            neuron_states[cloned.layermap[i][j]] += cloned.neural_net[cloned.layermap[i][j]].bias;
            pre_activations[cloned.layermap[i][j]] = neuron_states[cloned.layermap[i][j]];
            const float al = cloned.neural_net[cloned.layermap[i][j]].alpha;
            neuron_states[cloned.layermap[i][j]] = activation_function(neuron_states[cloned.layermap[i][j]],cloned.neural_net[cloned.layermap[i][j]].activation_function_v,a);
            f_x[cloned.layermap[i][j]] = neuron_states[cloned.layermap[i][j]];
            neuron_states[cloned.layermap[i][j]] = (f_x[cloned.layermap[i][j]] * al) + (pre_activations[cloned.layermap[i][j]]);
        }
    }
    #pragma omp simd
    for (int i = 0; i < neuron_states.size(); i++)
    {
        t_minus1[i] = neuron_states[i];
    } 
}

void NNclone::bptt(const NN &cloned,const int timestep,std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &forwardpass_pa,std::vector<std::vector<float>> &forwardpass_fx, std::vector<std::vector<float>> &target_output_loss ,const float ReLU_leak,const float gradient_limit)
{
    weights_gradient.resize(cloned.neural_net.size());
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        weights_gradient[i].resize(cloned.neural_net[i].weights.size());
        std::fill(weights_gradient[i].begin(),weights_gradient[i].end(),0);
    }
    bias_gradient.resize(cloned.neural_net.size());
    std::fill(bias_gradient.begin(),bias_gradient.end(),0);
    neuron_gradient.reserve(forwardpass_states.size());
    neuron_gradient.resize((forwardpass_states.size() >=neuron_gradient.size()) ? forwardpass_states.size():neuron_gradient.size());
    for (int i = 0; i < timestep; i++)
    {
        neuron_gradient[i].resize(cloned.neural_net.size());
        std::fill(neuron_gradient[i].begin(),neuron_gradient[i].end(),0);
    }
    alpha_gradient.resize(cloned.neural_net.size());
    std::fill(alpha_gradient.begin(),alpha_gradient.end(),0);
    //for loop descending starting from the most recent timestep
    for (int i = timestep - 1; i > 0; i--)
    {
        for(int j = 0; j < target_output_loss[i].size(); j++){
            neuron_gradient[i][cloned.output_index[j]] += target_output_loss[i][j];
        }
        for (int j = cloned.layermap.size() - 1; j >= 0  ; j--)
        {
            for (int k = 0; k < cloned.layermap[j].size(); k++)
            {
                //neuron_gradient[i][cloned.layermap[j][k]] = (std::abs(neuron_gradient[i][cloned.layermap[j][k]]) > gradient_limit) ? (sign_of(neuron_gradient[i][cloned.layermap[j][k]]) * gradient_limit):(neuron_gradient[i][cloned.layermap[j][k]]); 
                const float a = cloned.neural_net[cloned.layermap[j][k]].alpha;
                float dldz = neuron_gradient[i][cloned.layermap[j][k]] * (1 + (act_func_derivative(forwardpass_pa[i][cloned.layermap[j][k]],cloned.neural_net[cloned.layermap[j][k]].activation_function_v,ReLU_leak)*a));
                alpha_gradient[cloned.layermap[j][k]] += (forwardpass_fx[i][cloned.layermap[j][k]]) * neuron_gradient[i][cloned.layermap[j][k]];
                bias_gradient[cloned.layermap[j][k]] += dldz;
                //surely this won't lead to exploding gradients, right??
                neuron_gradient[i-1][cloned.layermap[j][k]] = (cloned.neural_net[cloned.layermap[j][k]].memory) ? (neuron_gradient[i-1][cloned.layermap[j][k]] + dldz):neuron_gradient[i-1][cloned.layermap[j][k]];
                #pragma omp simd
                for (int l = 0; l < cloned.neural_net[cloned.layermap[j][k]].weights.size(); l++)
                {
                    int t = (cloned.layermap[j][k] > cloned.neural_net[cloned.layermap[j][k]].weights[l].index) ? i:(i-1);
                    neuron_gradient[t][cloned.neural_net[cloned.layermap[j][k]].weights[l].index] += dldz * cloned.neural_net[cloned.layermap[j][k]].weights[l].value;
                    weights_gradient[cloned.layermap[j][k]][l] += dldz * forwardpass_states[i][cloned.neural_net[cloned.layermap[j][k]].weights[l].index];
                }
            }  
        }
    }
    for(int i = 0; i < target_output_loss[0].size(); i++){
        neuron_gradient[0][cloned.output_index[i]] += target_output_loss[0][i];
    }
    for (int j = cloned.layermap.size() - 1; j >= 0 ; j--)
    {
        for (int k = 0; k < cloned.layermap[j].size(); k++)
        {
            //neuron_gradient[0][cloned.layermap[j][k]] = (std::abs(neuron_gradient[0][cloned.layermap[j][k]]) > gradient_limit) ? (sign_of(neuron_gradient[0][cloned.layermap[j][k]]) * gradient_limit):(neuron_gradient[0][cloned.layermap[j][k]]); 
            const float a = cloned.neural_net[cloned.layermap[j][k]].alpha;
            float dldz = neuron_gradient[0][cloned.layermap[j][k]] * (1 + (act_func_derivative(forwardpass_pa[0][cloned.layermap[j][k]],cloned.neural_net[cloned.layermap[j][k]].activation_function_v,ReLU_leak)*(a)));
            alpha_gradient[cloned.layermap[j][k]] += (forwardpass_fx[0][cloned.layermap[j][k]]) * neuron_gradient[0][cloned.layermap[j][k]];
            bias_gradient[cloned.layermap[j][k]] += dldz;
            for (int l = 0; l < cloned.neural_net[cloned.layermap[j][k]].weights.size(); l++)
                {
                if(cloned.layermap[j][k] > cloned.neural_net[cloned.layermap[j][k]].weights[l].index){       //if greater then must be same time step, if less must be previous timestep
                    neuron_gradient[0][cloned.neural_net[cloned.layermap[j][k]].weights[l].index] += dldz * cloned.neural_net[cloned.layermap[j][k]].weights[l].value;        
                    weights_gradient[cloned.layermap[j][k]][l] += dldz * forwardpass_states[0][cloned.neural_net[cloned.layermap[j][k]].weights[l].index];
                }
                else{
                    weights_gradient[cloned.layermap[j][k]][l] += dldz * forwardpass_states[0][cloned.neural_net[cloned.layermap[j][k]].weights[l].index];
                }
            }
        }    
    }
    for (int i = 0; i < bias_gradient.size(); i++)
    {
        if (std::isnan(bias_gradient[i]) || std::isinf(bias_gradient[i]) || (std::abs(bias_gradient[i]) > gradient_limit))
        {
            bias_gradient[i] = sign_of(bias_gradient[i]) * gradient_limit;
        }
        bias_g[i] += bias_gradient[i];  
    }

    for (int i = 0; i < alpha_g.size(); i++)
    {
        if (std::isnan(alpha_gradient[i]) || std::isinf(alpha_gradient[i]) || (std::abs(alpha_gradient[i]) > gradient_limit))
        {
            alpha_gradient[i] = sign_of(alpha_gradient[i]) * gradient_limit;
        }
        alpha_g[i] += alpha_gradient[i]; 
    }

    for (int i = 0; i < weights_gradient.size(); i++)
    {
        for (int j = 0; j < weights_gradient[i].size(); j++)
        {
            if (std::isnan(weights_gradient[i][j]) || std::isinf(weights_gradient[i][j]) || (std::abs(weights_gradient[i][j]) > gradient_limit))
            {
                weights_gradient[i][j] = sign_of(weights_gradient[i][j]) * gradient_limit;  
            }
            weights_g[i][j] += weights_gradient[i][j];      
        }
    }
}

void NNclone::sync(const NN &cloned)
{
    neuron_states.resize(cloned.neural_net.size());
    pre_activations.resize(cloned.neural_net.size());
    f_x.resize(cloned.neural_net.size());
    t_minus1.resize(cloned.neural_net.size());
    weights_g.resize(cloned.neural_net.size());
    bias_g.resize(cloned.neural_net.size());
    alpha_g.resize(cloned.neural_net.size());
    alpha_gradient.resize(cloned.neural_net.size());
    neuron_gradient.resize(cloned.neural_net.size());
    weights_gradient.resize(cloned.neural_net.size());
    bias_gradient.resize(cloned.neural_net.size());
    for (int i = 0; i < cloned.neural_net.size(); i++)
    {
        weights_g[i].resize(cloned.neural_net[i].weights.size(),0);
        weights_gradient[i].resize(cloned.neural_net[i].weights.size(),0);
    }
    gradient_clear();
    neural_net_clear();
}

