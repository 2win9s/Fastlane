#pragma once

//the following code will be very verbose in terms of comments

#include<vector>
#include<string>
#include<set>

float GELU(float pre_activation);
float dGELU(float x);
float dReLU(float x, float a);

//the neural net struct
struct NN
{
    //this is a struct for convenience where something needs both an integer for indexing and a float
    struct index_value_pair
    {
        int index;
        float value;
        index_value_pair(int x, float y);
    };
    struct neuron
    {
        float output;                               //the state of the neuron
        float bias;                                 //the value of the bias applied
        std::vector<index_value_pair> weights;      //the index label of the neurons connected by weights and the value of these weights
        int activation_function_v;                  //the value of the interger determines the activation function used check the impemented function
        bool memory;                                //will the previous state be added to the pre activation or not, default false
        bool input_neuron;                          //is it an input neuron?, default false
        //constructor, NOTE TO SELF CHECK IF EMPTY WEIGHTS DEFAULT ARGUEMENT
        //activation functions "ReLU", "leaky ReLU" or "GELU"
        neuron(float init_val, float init_bias, std::vector<index_value_pair> init_weights,std::string act_func);

        //approximated
        void activation_function(float a = 0);
        float act_func_derivative(float x, float a = 0);

        bool isnt_input(int neuron_index);                     
    };


    std::vector<neuron> neural_net;         //firing order is order of index by default
    std::vector<float> act_func_derivatives;
    std::vector<float> p_mmean;             //pre activation minus mean for layernorm
    std::vector<int> input_index;           //indexing recording input neurons
    std::vector<int> output_index;          //indexing recording output neurons
    std::vector<int> memory_index;          //indexing recording the neurons that add on their previous state
    
    std::vector<std::vector<float>> momentumW;
    std::vector<float> momentumB;  
    std::vector<std::vector<float>> weights_g;
    std::vector<float> bias_g;   
    std::vector<std::vector<float>> neuron_gradient;
    
    std::vector<std::vector<float>> weights_gradient;   //reducing memory allocations and deallocations
    std::vector<float>  bias_gradient;                  //reducing memory allocations and deallocations
    
    std::vector<std::vector<int>> layermap; //separating vector of neurons into layers with no weights connecting the neurons in each layer
    //only for forward

    void layermap_sync();    //this is to create/ update the layermap, this will be called an run after initialisation, regularisation and adding more weights
    
    //dunno if it works for this, no idea how to derive one
    float He_initialisation(int n, float a = 0);

    public:
    /*
    biases will initialise to 0
    initially determinea the density and number of connections to each neuron
    with function arguements size of neural network, connection density between zero and 1 representing on average the percentage
    of other neurons connected to each neuron(as input),
    connection_sd represents the standard deviation from the average number of connections again as a proportion(will be normally distributed)  
    truncated at <0 and >1
    */
    NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons, std::vector<int> memory_neurons, float connection_density, float connection_sd);
    //NN(int size, std::vector<int> input_neurons, std::vector<int> output_neurons);
    //NN() = delete;      //no default constructor with no arguements, is this bad practice?
    NN();
    //will initialise from data in file_name
    NN(std::string file_name);
    //will save object to text file
    void save(std::string file_name);


    void neural_net_clear();
    void momentum_clear();
    void gradient_clear();
    void act_func_derivatives_clear();

    //activation of neuron at index n
    void neuron_activation(int n, float a = 0);
    float loss(std::vector<float> & output, std::vector<float> & target, int loss_f = 0);
    void forward_pass(std::vector<float> &inputs, float a = 0,bool layer_norm);
    void forward_pass_s_pa(std::vector<float> &inputs, float a = 0,bool layer_norm);    //save the pre activation

    //back propgation through time, no weight updates, only gradient
    void bptt(int timestep,std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &forwardpass_d, std::vector<std::vector<float>> &target_output_loss , float ReLU_leak = 0, float gradient_limit = 10);
    
    //passes the gradients through a softsign function before updating momentum and weights, stochastic gradient descent, weights updated each iteration
    void bptt_softsign_gradient(std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &target_output_loss,float learning_rate, float momentum_param = 0.9 , float ReLU_leak = 0, float gradient_limit = 5, std::vector<bool> freeze_neuron = {});
    void update_momentum(float momentum = 0.9);
    void update_parameters(float learning_rate, std::vector<bool> freeze_neuron = {});
    void l1_reg(float h_param, std::vector<bool> freeze_neuron = {});
    void l2_reg(float w_decay, std::vector<bool> freeze_neuron = {});
    void weight_noise(float sigma, std::vector<bool> freeze_neuron ={});

    void new_weights(int n_new_weights, std::vector<bool> freeze_neuron ={});
    void prune_weights(float weights_cutoff, std::vector<bool> freeze_neuron = {});

    void sleep();
    void backward_pass(std::vector<float> &forwardpass_past,std::vector<float> &forwardpass_current, std::vector<float> &target,float learning_rate, float momentum_param = 0.9, float ReLU_leak = 0);
    
    //does the value of every input have influence all output neurons
    bool input_connection_check(bool outputc = true, bool memoryc = true, bool rc = true);  //rc is checking for recurrent connections within 1 timestep, of course there may be a more complex pathway that involves many timesteps, write your own function if needed
    //does the value of every "memory neuron" influence all output neurons
    bool memory_connection_check(bool rc = true);

    //c_step_size is the number of connections added before each check, io check input to output, im check input to memeory, mo memory to output, and rc whether to consider recurrent connections or not
    void ensure_connection(int c_step_size ,bool io = true, bool im = true, bool mo = true, bool rc = true);
};

struct NNclone
{
    std::vector<float> neuron_states;
    std::vector<float> p_mmean; 
    std::vector<float> act_func_derivatives;
    std::vector<std::vector<float>> weights_g;
    std::vector<float> bias_g;   
    std::vector<std::vector<float>> neuron_gradient;
    
    std::vector<std::vector<float>> weights_gradient;   //reducing memory allocations and deallocations
    std::vector<float>  bias_gradient;                  //reducing memory allocations and deallocations
    
    NNclone();
    NNclone(const NN &cloned);


    void neural_net_clear();
    void gradient_clear();
    void act_func_derivatives_clear();

    void forward_pass(const NN &cloned,std::vector<float> &inputs, float a = 0,bool layer_norm);
    void forward_pass_s_pa(const NN &cloned,std::vector<float> &inputs, float a = 0,bool layer_norm);    //save the pre activation
    void bptt(const NN &cloned,int timestep,std::vector<std::vector<float>> &forwardpass_states, std::vector<std::vector<float>> &forwardpass_d, std::vector<std::vector<float>> &target_output_loss , float ReLU_leak = 0, float gradient_limit = 10);

    void sync(const NN &cloned);
};












