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


std::random_device rdev;                          
std::mt19937 twister(rdev());


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

bool NN::neuron::isnt_input(int neuron_index){
    for (int i = 0; i < weights.size(); i++)
    {
        if(weights[i].index == neuron_index)
        {
            return false;
        }
    }
    return true;
}

inline bool contain_int(std::vector<int> &vec, int x){
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] == x)
        {
            return true;
        }
    }
    return false;
}

void NN::layermap_sync()
{
    layermap.clear();
    std::vector<std::set<int>> dependency;
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
    /*
    for (int i = 0; i < input_tree.size(); i++)
    {
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if (i < neural_net[i].weights[j].index){;}
            else{input_tree[i].emplace_back(neural_net[i].weights[j].index);}
        }
        std::set<int> set(input_tree[i].begin(),input_tree[i].end());        
        for (int j = 0; j < input_tree[i].size(); j++)
        {
            for (int k = 0; k < input_tree[input_tree[i][j]].size(); k++)
            {
                set.insert(input_tree[input_tree[i][j]][k]);
            }
        } 
    } */
    for (int i = 0; i < dependency.size(); i++)
    {
        std::vector<int> w;
        w.reserve(neural_net[i].weights.size());
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if (i > neural_net[i].weights[j].index){w.emplace_back(neural_net[i].weights[j].index);}
        }
        for (int j = 0; j < w.size(); j++)
        {
            dependency[i].insert(w[j]);
            dependency[i].insert(dependency[w[j]].begin(),dependency[w[j]].end());  //set union may be faster test for use case
            //std::set<int> un;
            //std::set_union(dependency[i].begin(),dependency[i].end(),dependency[w[j]].begin(),dependency[w[j]].end(), std::inserter(un,un.end()));
            //dependency[i] = un;
        }
        //std::cout<<dependency[i].size()<<std::endl;
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
            if(index_label[i] && (dependency[i].count(initial_neuron) == 0))
            {
                layermap_layer_candidate.emplace_back(i);
            }
        }
        for (int i = 0; i < layermap_layer_candidate.size(); i++)                       
        {
            for (int j = i; j < layermap_layer_candidate.size(); j++)                   //getting rid of the neurons that have the neuron at i index as input
            {
                //remove neurons of higher index that contain some connection to other neurons in the layer 
                if(dependency[layermap_layer_candidate[j]].count(layermap_layer_candidate[i]) || (!neural_net[layermap_layer_candidate[i]].isnt_input(layermap_layer_candidate[j]))) //if it is an input, double negation
                {
                    layermap_layer_candidate.erase(layermap_layer_candidate.begin() + j);
                    j--;                                                                //as the for loop will execute j++ and the element at index j just got removed, the new element at index j is the old one at j+1
                }
            }
        }
        mapped_n += layermap_layer_candidate.size();
        for (int i = 0; i < layermap_layer_candidate.size(); i++)
        {
            index_label[layermap_layer_candidate[i]] = false;
            dependency[i].clear();
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

//constructor for neural net struct
NN::NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, std::vector<int> memory_neurons, float connection_density, float connection_sd)
    :neural_net(size,neuron(0,0,{},"ReLU6"))
    ,momentumW(size)
    ,momentumB(size,0)
    ,input_index(input_neurons)
    ,output_index(output_neurons)
    ,memory_index(memory_neurons)
{                         //rng using the standard library's mersenne twister
    std::normal_distribution<float> connection_dist(connection_density, connection_sd);
    std::uniform_int_distribution<int> rand_neuron(0,size - 2);     //uniform distribution to pick a random neuron excluding the index of current neuron
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
        for (int j = 0; j < connect_n; j++)
        {
            int input_neuron_index = rand_neuron(twister);
            if((input_neuron_index >= i))
            {
                continue;           //Do not initiallise with recurrent connections, can be added later
            }
            if(neural_net[i].isnt_input(input_neuron_index)){
                neural_net[i].weights.emplace_back(index_value_pair(input_neuron_index,0));
                momentumW[i].emplace_back(0);
            }
            else{
                j--;
            }
        }
    }
    weights_g = momentumW;
    bias_g = momentumB;
    layermap_sync();
    float multiplier = 1/std::sqrt(layermap.size()) * 1/size;       //I don't fully understand Fixup and how to scale skip connections, hopefully this is adequate
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
                    neural_net[i].weights[j].value = weight_init_dist(twister) /* multiplier*/;
                }
            }
        } 
    }
}


NN::NN()
    :neural_net(0,neuron(0,0,{},"ReLU6"))
    ,momentumW(0)
    ,momentumB(0,0)
    ,input_index(0)
    ,output_index(0)
{

}

void NN::neural_net_clear(){
    for (int i = 0; i < neural_net.size(); i++)
    {
        neural_net[i].output = 0;
    } 
}

void NN::momentum_clear(){
    for (int i = 0; i < momentumW.size(); i++)
    {
        for(int j = 0; j < momentumW[i].size(); j++){
            momentumW[i][j] = 0;
        }
    }
    for(int i = 0; i < momentumB.size(); i++){
        momentumB[i] = 0;
    }
}

void NN::gradient_clear(){
    for (int i = 0; i < bias_g.size(); i++)
    {
        bias_g[i] =0;
    }
    
    for (int i = 0; i < weights_g.size(); i++)
    {
        for(int j = 0; j< weights_g[i].size(); j++){
            weights_g[i][j] = 0;
        }
    }
    
}

float GELU(float pre_activation){
    //float x = pre_activation; 
    //float y = M_SQRT1_2 * (x + (0.044715 * x * x * x)); 
    //return (0.5 * x * (1 + std::tanh(y)));
    float CSND = 0.5 * (1 + std::erf(pre_activation * 0.707106781186547524402));     //0.70710678118654752440 = 1/sqrt(2)          
    return (CSND * pre_activation);
}

void NN::neuron::activation_function(float a){
    switch (activation_function_v)
    {
    case 0:
        output = GELU(output);
        break;
    case 1:
        //ReLU6, capping the ReLU output at 6
        output = (output > 6) ? 6:output;
        output = (output > 0) ? output:a * output;
        break;
    case 2:
        output = output;    //linear y = x
        break;
    default:
        output = (output > 0) ? output:a * output;
        break;
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
            if (neural_net[layermap[i][j]].memory || neural_net[layermap[i][j]].input_neuron)
            {
                neural_net[layermap[i][j]].output = neural_net[layermap[i][j]].output;
            }
            else{
                neural_net[layermap[i][j]].output = 0;
            } 
            for (int l = 0; l < neural_net[layermap[i][j]].weights.size(); l++)
            {
                //apologies for the naming scheme
                neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].weights[l].value * neural_net[neural_net[layermap[i][j]].weights[l].index].output;
            }
            neural_net[layermap[i][j]].output += neural_net[layermap[i][j]].bias;
            neural_net[layermap[i][j]].activation_function(a);
        }
    } 
}

float standard_normal_pdf(float x){
    const double sqrt1_2pi = 0.3989422804014327;
    return (sqrt1_2pi * std::exp(x * x * -0.5));
}

inline float dGELU(float x){
    float CSND = 0.5 * (1 + std::erf(x * 0.70710678118654752440)); 
    return (CSND + (x * standard_normal_pdf(x)));
}

inline float dReLU(float x, float a){
    return (x > 0) ? 1:a;
}

inline float dReLU6(float x, float a){
    x = (x >= 6) ? a:1;
    x = (x <= 0) ? a:1;
    return x;
}

float NN::neuron::act_func_derivative(float x, float a){
    switch (activation_function_v)
        {
        case 0:
            return dGELU(x);
            break;
        case 1:
            return (dReLU6(x,a));
            break;
        case 2:
            return 1;       //linear do nothing, this is for a potential softmax implementation for the output neuron
        default:
            return (dReLU(x,a));
            break;
        }
}

//back propgation through time, no weight updates, only gradient
void NN::bptt(std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &target_output_loss, float ReLU_leak, float gradient_limit)
{
    weights_gradient.resize(neural_net.size());
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        weights_gradient[i].reserve(neural_net[i].weights.size());
        weights_gradient[i].resize(neural_net[i].weights.size());
        std::fill(weights_gradient[i].begin(),weights_gradient[i].end(),0);
    }
    bias_gradient.reserve(neural_net.size());
    bias_gradient.resize(neural_net.size(),0);
    std::fill(bias_gradient.begin(),bias_gradient.end(),0);
    neuron_gradient.reserve(forwardpass_states.size());
    neuron_gradient.resize((forwardpass_states.size() >=neuron_gradient.size()) ? forwardpass_states.size():neuron_gradient.size());
    for (int i = 0; i < forwardpass_states.size(); i++)
    {
        neuron_gradient[i].reserve(neural_net.size());
        neuron_gradient[i].resize(neural_net.size(),0);
        std::fill(neuron_gradient[i].begin(),neuron_gradient[i].end(),0);
    }
    //for loop descending starting from the most recent timestep
    for (int i = forwardpass_states.size() - 1; i > 0; i--)
    {
        for(int j = 0; j < target_output_loss[i].size(); j++){
            neuron_gradient[i][output_index[j]] += target_output_loss[i][j];
        }
        for (int j = layermap.size() - 1; j >= 0  ; j--)
        {
            for (int k = 0; k < layermap[j].size(); k++)
            {
                float dldz = neuron_gradient[i][layermap[j][k]] 
                * neural_net[layermap[j][k]].act_func_derivative(forwardpass_states[i][layermap[j][k]],ReLU_leak);
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
            float dldz = neuron_gradient[0][layermap[j][k]] * neural_net[layermap[j][k]].act_func_derivative(forwardpass_states[0][layermap[j][k]],ReLU_leak);
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
        if (std::abs(bias_gradient[i]) > gradient_limit)
        {
            if (bias_gradient[i] > 0){
                bias_gradient[i] = gradient_limit;
            }
            else{
                bias_gradient[i] = -1 * gradient_limit;
            }
        }
        else if (std::isnan(bias_gradient[i]) || std::isinf(bias_gradient[i]))
        {
            if (std::signbit(bias_gradient[i]))
            {
                bias_gradient[i] = -1 * gradient_limit;
            }
            else{
                bias_gradient[i] = gradient_limit;
            }
        }
        bias_g[i] += bias_gradient[i];   
    }
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        for (int j = 0; j < weights_gradient[i].size(); j++)
        {
            
            if (std::abs(weights_gradient[i][j]) > gradient_limit)
            {
                if (weights_gradient[i][j] > 0)
                {
                    weights_gradient[i][j] = gradient_limit;
                }
                else{
                    weights_gradient[i][j] = -1 * gradient_limit;
                }
            }
            else if (std::isnan(weights_gradient[i][j]) || std::isinf(weights_gradient[i][j]))
            {
                if (std::signbit(weights_gradient[i][j]))
                {
                    weights_gradient[i][j] = -1 * gradient_limit;
                }
                else
                {
                    weights_gradient[i][j] = gradient_limit;
                }
                
            }
            weights_g[i][j] += weights_gradient[i][j];      
        }
    }
}


void NN::update_momentum(float momentum){
    for (int i = 0; i < momentumB.size(); i++)
    {
        momentumB[i] = (momentumB[i] * momentum) + (bias_g[i] * (1 - momentum));
    }
    for (int i = 0; i < momentumW.size(); i++)
    {
        for (int j = 0; j < momentumW[i].size(); j++)
        {
            momentumW[i][j] = (momentumW[i][j] * momentum) + (weights_g[i][j] * (1 - momentum));
        }
        
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
        if (freeze_neuron[i])
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
            neural_net[i].weights[j].value -= h_param * neural_net[i].weights[j].value; 
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


//the new weights will be initialised to 0, also resets momentum to 0
void NN::new_weights(int m_new_weights, std::vector<bool> freeze_neuron){
    if (freeze_neuron.size() == 0)
    {
        freeze_neuron.reserve(neural_net.size());
        freeze_neuron.resize(neural_net.size(),false);
    }
    std::uniform_int_distribution<int> dist_to(0, neural_net.size() -1); 
    std::uniform_int_distribution<int> dist_from(0, neural_net.size() -2);
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
            int index_from = dist_from(twister);
            if (index_from >= i)
            {
                index_from++;
            }
            
            if (neural_net[i].isnt_input(index_from))
            {
                neural_net[i].weights.emplace_back(index_from,0);
                momentumW[i].emplace_back(0);
                weights_g[i].emplace_back(0);
            }
            else{
                j--;
            }
        }
    }
    delete[] new_weights;
    delete[] new_weights_limit;
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
    if (std::filesystem::exists(file_name)){;}
    else{std::cout<<"ERROR "<<file_name<<" does not exist"<<std::endl; std::exit(EXIT_FAILURE);}
    std::ifstream file(file_name);
    file >> str_data;
    file >> str_data;
    neural_net.resize(std::stoi(str_data),neuron(0,0,{},"ReLU6"));
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
            file >> data;
            file >> data;
            neural_net[itr].bias = std::stof(data);
            file >> data;
            file >> data;
            file >> data;
            neural_net[itr].weights.reserve(std::stoi(data));
            while(true)
            {
                file >> data;
                if(data == "</weights>"){
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
                    neural_net[layermap[j][k]].output += 1 * neural_net[neural_net[layermap[j][k]].weights[l].index].output;    //just to see if input will influence all outputs, easiest way to do this no activation function and set all weights to +1
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
                        neural_net[layermap[j][k]].output += 1 * neural_net[neural_net[layermap[j][k]].weights[l].index].output;    
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
                    neural_net[layermap[j][k]].output += 1 * neural_net[neural_net[layermap[j][k]].weights[l].index].output;    //easiest way to do this no activation function and set all weights to +1
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
                        neural_net[layermap[j][k]].output += 1 * neural_net[neural_net[layermap[j][k]].weights[l].index].output;    
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
            new_weights(c_step_size);
        }
    }
    if (mo)
    {
        while (!memory_connection_check(rc))
        {
            new_weights(c_step_size);
        }
    }
}


