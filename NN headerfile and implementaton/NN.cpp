#include<vector>
#include<cmath>
#include<iostream>
#include<cstdlib>
#include<string>
#include<random>
#include<fstream>
#include<limits>
#include<iomanip>
#include<algorithm>


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
        if(act_func == "ReLU"){
            activation_function_v = 1;
        }
        else if (act_func == "leaky ReLU")
        {
           activation_function_v = 2;
        }
        else if (act_func == "GELU")
        {
            activation_function_v = 3;
        }
        else{
            std::cout<<"invalid string for activation function, please input \"ReLU\", \"leaky ReLU\" or, \"GELU\" only";
            std::cout<<"note no spaces allowed";
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

void NN::layermap_sync()
{
    layermap.clear();
    //std::vector<bool> index_label(neural_net.size(),true);          
    //this is to help label the neurons (not included yet in layermap = true)
    bool* index_label = new bool[neural_net.size()]; 
    for (int i = 0; i < neural_net.size(); i++)
    {
        index_label[i] = true;
    }
    
    std::vector<int> layermap_layer_candidate;
    int initial_neuron = 0;                                         //the neuron to be included into layermap with highest priority at beginning is at index 0, order of neuron index is order of firing
    int mapped_n = 0;
    int counter = 0;
    for(long limit = 0; limit < 214748364; limit++)         //avoiding problems with infinite while loops even if there is a bug
    {                                                       //for most practical intents and purposes this is a while loop
        layermap_layer_candidate.clear();
        layermap_layer_candidate.reserve(neural_net.size() - mapped_n);
        for (int i = initial_neuron; i < neural_net.size(); i++)
        {   if(index_label[i] && (neural_net[i].isnt_input(initial_neuron)))
            {
                layermap_layer_candidate.emplace_back(i);
            }
        }
        for (int i = 0; i < layermap_layer_candidate.size(); i++)                       
        {
            for (int j = i; j < layermap_layer_candidate.size(); j++)                   //getting rid of the neurons that have the neuron at i index as input
            {
                counter++;
                if(!neural_net[layermap_layer_candidate[j]].isnt_input(layermap_layer_candidate[i])) //if it is an input, double negation
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
    :neural_net(size,neuron(0,0,{},"ReLU"))
    ,momentumW(size)
    ,momentumB(size,0)
    ,input_index(input_neurons)
    ,output_index(output_neurons)
    ,memory_index(memory_neurons)
{
    std::cout<<"creating neural net"<< "\n";                                //rng using the standard library's mersenne twister
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
        momentumW[i].resize(connect_n,0);
        neural_net[i].weights.reserve(connect_n);
        for (int j = 0; j < connect_n; j++)
        {
            int input_neuron_index = rand_neuron(twister);
            if(input_neuron_index >= i)
            {
                input_neuron_index++;   //shifts the distribution to take out the gap
            }
            if(neural_net[i].isnt_input(input_neuron_index)){
                neural_net[i].weights.emplace_back(index_value_pair(input_neuron_index,0));
            }
            else{
                j--;
            }
        }
    }
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
                neural_net[i].weights[j].value = weight_init_dist(twister);
            }
            else{
                neural_net[i].weights[j].value = weight_init_dist(twister) /* multiplier*/;
            }
        } 
    }
}

/*
NN::NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons)
    :neural_net(size,neuron(0,0,{},"ReLU"))
    ,momentumW(size)
    ,momentumB(size,0)
    ,input_index(input_neurons)
    ,output_index(output_neurons)
{

}*/

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
        //ReLU7, capping the ReLU output at 7, basically ReLU6 but I want to be different
        output = (output > 7) ? 7:output;
        output = (output > 0) ? output:a * output;
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

inline float dReLU7(float x, float a){
    x = (x >= 7) ? a:1;
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
            return (dReLU7(x,a));
            break;
        default:
            return (dReLU(x,a));
            break;
        }
}

//back propgation through time
void NN::bptt(std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &target,float learning_rate, float momentum_param, float ReLU_leak, float gradient_limit)
{
    std::vector<std::vector<float>> neuron_gradient(forwardpass_states.size());
    neuron_gradient.shrink_to_fit();
    for (int i = 0; i < neuron_gradient.size(); i++)
    {
        neuron_gradient[i].reserve(neural_net.size());
        neuron_gradient[i].resize(neural_net.size(),0);
    }
    
    std::vector<std::vector<float>> weights_gradient(neural_net.size());
    weights_gradient.shrink_to_fit();
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        weights_gradient[i].reserve(neural_net[i].weights.size());
        weights_gradient[i].resize(neural_net[i].weights.size(),0);
    }
    std::vector<float> bias_gradient(neural_net.size(),0);
    bias_gradient.shrink_to_fit();
    //for loop descending starting from the most recent timestep
    for (int i = forwardpass_states.size() - 1; i > 0; i--)
    {
        for(int j = 0; j < target[i].size(); j++){
            neuron_gradient[i][output_index[j]] += forwardpass_states[i][output_index[j]] - target[i][j];
        }
        for (int j = layermap.size() - 1; j >= 0  ; j--)
        {
            for (int k = 0; k < layermap[j].size(); k++)
            {
                float dldz = neuron_gradient[i][layermap[j][k]] 
                * neural_net[layermap[j][k]].act_func_derivative(forwardpass_states[i][layermap[j][k]],ReLU_leak);
                bias_gradient[layermap[j][k]] += dldz;
                //surely this won't lead to exploding gradients, right??
                //neuron_gradient[i-1][layermap[j][k]] = (neural_net[layermap[j][k]].memory) ? neuron_gradient[i-1][layermap[j][k]] + dldz:neuron_gradient[i-1][layermap[j][k]];
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
    for(int i = 0; i < target[0].size(); i++){
        neuron_gradient[0][output_index[i]] += forwardpass_states[0][output_index[i]] - target[0][i];
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
    neuron_gradient.clear();
    neuron_gradient.shrink_to_fit();
    for (int i = 0; i < bias_gradient.size(); i++)
    {
            
        if (std::abs(bias_gradient[i]) > gradient_limit)
        {
            if (bias_gradient[i] > 0){
                bias_gradient[i] = gradient_limit;
            }
            else{
                bias_gradient[i] = -gradient_limit;
            }
        }   
        momentumB[i] = momentumB[i] * momentum_param + (bias_gradient[i] * (1 - momentum_param));
        if (neural_net[i].memory || neural_net[i].input_neuron)
        {
            continue;
        }
        neural_net[i].bias -= learning_rate * momentumB[i];
    }
    bias_gradient.clear();
    bias_gradient.shrink_to_fit();
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
                    weights_gradient[i][j] = -gradient_limit;
                }
                    
            }
            momentumW[i][j] = momentumW[i][j] * momentum_param + (weights_gradient[i][j] * (1 - momentum_param));
            neural_net[i].weights[j].value -= learning_rate * momentumW[i][j];
        }
    }
}


//weights = weights - h_param * signof(weights)
void NN::l1_reg(int h_param){
    for (int i = 0; i < neural_net.size(); i++)
    {
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value -= h_param * std::signbit(neural_net[i].weights[j].value); 
        }
        
    }
    
}

//weights -= weights * w_decay
void NN::l2_reg(int w_decay){
    for (int i = 0; i < neural_net.size(); i++)
    {
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            neural_net[i].weights[j].value -= neural_net[i].weights[j].value * w_decay;
        }
        
    }
    
}

//the new weights will be initialised to 0, also resets momentum to 0
void NN::new_weights(int m_new_weights){
    std::uniform_int_distribution<int> dist_to(0, neural_net.size() -1); 
    std::uniform_int_distribution<int> dist_from(0, neural_net.size() -2);
    float multiplier = 1/std::sqrt(layermap.size()) * 0.5;
    int* new_weights = new int[neural_net.size()];
    momentum_clear();
    for (int i = 0; i < neural_net.size(); i++)
    {
        new_weights[i] = 0;
    }
    

    for (int i = 0; i < m_new_weights; i++)
    {
        int index_to = dist_to(twister);
        if (neural_net[index_to].input_neuron)
        {   
        }
        else{
            new_weights[index_to]++;
        }
    }
    
    for (int i = 0; i < neural_net.size(); i++)
    {
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
            }
        }
    }
    delete[] new_weights;
    layermap_sync();
}

//if below weights_cutoff, weights are removed, also reset momentum to 0 
void NN::prune_weights(float weights_cutoff){
    for (int i = 0; i < neural_net.size(); i++)
    {
        for (int j = 0; j < neural_net[i].weights.size(); j++)
        {
            if(std::abs(neural_net[i].weights[j].value) < weights_cutoff){
                neural_net[i].weights.erase(neural_net[i].weights.begin() + j);
                momentumW[i].pop_back();
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
    file >> str_data;
    file >> str_data;
    neural_net.resize(std::stoi(str_data),neuron(0,0,{},"ReLU"));
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
}

void NN::sleep(){

}

void NN::backward_pass(std::vector<float> &forwardpass_past,std::vector<float> &forwardpass_current, std::vector<float> &target,float learning_rate, float momentum_param,float ReLU_leak){
    std::vector<float> neuron_gradient(neural_net.size(),0);
    neuron_gradient.shrink_to_fit();
    std::vector<std::vector<float>> weights_gradient(neural_net.size());
    weights_gradient.shrink_to_fit();
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        weights_gradient[i].reserve(neural_net[i].weights.size());
        weights_gradient[i].resize(neural_net[i].weights.size(),0);
    }
    std::vector<float> bias_gradient(neural_net.size(),0);
    bias_gradient.shrink_to_fit();

    for (int i = 0; i < target.size(); i++)
    {
        neuron_gradient[output_index[i]] += forwardpass_current[output_index[i]] - target[i];
    }
    for (int i = layermap.size() - 1; i >= 0; i--)
    {
        for (int j = 0; j < layermap[i].size(); j++)
       {
            for (int k = 0; k < neural_net[layermap[i][j]].weights.size(); k++){
		        float dldz = neuron_gradient[layermap[i][j]] * neural_net[layermap[i][j]].act_func_derivative(forwardpass_current[layermap[i][j]],ReLU_leak);
                bias_gradient[layermap[i][j]] += dldz;
                
		    if(layermap[i][j] > neural_net[layermap[i][j]].weights[k].index){
                    weights_gradient[layermap[i][j]][k] += dldz * forwardpass_current[neural_net[layermap[i][j]].weights[k].index];
                    neuron_gradient[neural_net[layermap[i][j]].weights[k].index] += dldz * neural_net[layermap[i][j]].weights[k].value;
                }
                else{   //we truncate the backpropagation here
                    weights_gradient[layermap[i][j]][k] += dldz * forwardpass_past[neural_net[layermap[i][j]].weights[k].index];
                }
            }
       }    
    }
    neuron_gradient.clear();
    neuron_gradient.shrink_to_fit();
    for (int i = 0; i < bias_gradient.size(); i++)
    {
        momentumB[i] = momentumB[i] * momentum_param + (bias_gradient[i] * (1 - momentum_param));
        neural_net[i].bias -= learning_rate * momentumB[i];
    }
    bias_gradient.clear();
    bias_gradient.shrink_to_fit();
    for (int i = 0; i < weights_gradient.size(); i++)
    {
        for (int j = 0; j < weights_gradient[i].size(); j++)
        {
            momentumW[i][j] = momentumW[i][j] * momentum_param + (weights_gradient[i][j] * (1 - momentum_param));
            neural_net[i].weights[j].value -= learning_rate * momentumW[i][j];
        }
    }
}







