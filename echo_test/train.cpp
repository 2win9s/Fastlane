/*Getting this artificial ner



#include"NN.hpp"
#include<windows.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>

std::random_device rd;                          
std::mt19937 mtwister(rd());
std::uniform_real_distribution<float> rand06(0,6);      //the neural net is using ReLU capped at 7 as the activation function for all neurons, so this range is to ensure it can be expressed

//the name of the neural network is hopeless, just like its architecture
NN hopeless("echo2.txt"); //inputs {0,1], output{20}


void parameter_check(){
    double w_mean = 0;
    double w_variance = 0;
    double b_mean = 0;
    double b_variance = 0;
    int w_count = 0;
    int b_count = hopeless.neural_net.size();
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        for (int j = 0; j < hopeless.neural_net[i].weights.size(); j++)
        {
            w_count += 1;
            w_mean += hopeless.neural_net[i].weights[j].value;
        }
        b_mean += hopeless.neural_net[i].bias;
    }
    b_mean = b_mean/b_count;
    w_mean = w_mean/w_count;
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        for (int j = 0; j < hopeless.neural_net[i].weights.size(); j++)
        {
            w_variance += (hopeless.neural_net[i].weights[j].value - w_mean)
            * (hopeless.neural_net[i].weights[j].value - w_mean);
        }
        b_variance += (hopeless.neural_net[i].bias - b_mean)
        * (hopeless.neural_net[i].bias - b_mean);
    } 
    w_variance = w_variance/w_count;
    b_variance = b_variance/b_count;
    std::cout<<"parameters"<<"\n";
    std::cout<<"mean weights "<<w_mean;
    std::cout<<" variance of weights "<<w_variance<<"\n";
    std::cout<<"mean bias "<<b_mean;
    std::cout<<" variance of bias "<<b_variance<<"\n";
    std::cout<<"number of weights"<<w_count<<std::endl;
}

void momentum_check(){
    double w_moment = 0;
    double b_moment = 0;
    for (int i = 0; i < hopeless.momentumW.size(); i++)
    {
        b_moment += hopeless.momentumB[i];
        for(int j = 0; j < hopeless.momentumW[i].size(); j++){
            w_moment += hopeless.momentumW[i][j];
        }
    }
    std::cout<<w_moment<<std::endl;
    std::cout<<b_moment<<std::endl;
}

//an iteration to measure the loss
float iteration(int record_timestep, int recall_timestep,bool print = false){
    float loss;
    std::vector<float> inputs(3,0);
    std::vector<float> target(1,0);
    std::vector<float> output(1,0);
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);       //the value we want the neural net to remember
            inputs[1] = 1;                      //this is to tell the neural net to remember
            target[0] = inputs[0];
            inputs[2] = 0;
        }
        else if (i == record_timestep - 2)
        {
            inputs[0] = 0;
            inputs[1] = 0;                      
            inputs[2] = 6;
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;                      //to tell the neural net to spit out the recorded value
            inputs[2] = 0;
        }
        else{
            std::fill(inputs.begin(),inputs.end(),0);
        }
        hopeless.forward_pass(inputs);
    }
    output[0] = hopeless.neural_net[hopeless.output_index[0]].output;
    loss = hopeless.loss(output,target,0);
    if (print)
    {
        std::cout<<"output "<<output[0]<<"\n";
        std::cout<<"target "<<target[0]<<"\n";
        std::cout<<std::endl;
    }
    
    return loss;
}

void tr_iteration(int record_timestep, int recall_timestep, float a, float re_leak){
    std::vector<float> inputs(3,0);
    std::vector<std::vector<float>> target(recall_timestep);
    std::vector<std::vector<float>> forwardpass_states(recall_timestep);
    std::vector<std::vector<float>> dl(recall_timestep);
    for (int i = 0; i < forwardpass_states.size(); i++)
    {
        forwardpass_states[i].reserve(hopeless.neural_net.size());
        forwardpass_states[i].resize(hopeless.neural_net.size(),0);
    }
    for (int i = 0; i < target.size(); i++)
    {
        target[i].reserve(1);
        target[i].resize(1,0);
        dl[i].reserve(1);
        dl[i].resize(1,0);
    }
    
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);
            inputs[1] = 1;
            target[recall_timestep - 1][0] = inputs[0];
            inputs[2] = 0; 
        }
        else if (i == record_timestep - 2)
        {
            inputs[0] = 0;
            inputs[1] = 0;                      
            inputs[2] = 6;
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;
            inputs[2] = 0;
        }
        else
        {
            std::fill(inputs.begin(),inputs.end(),0);
        }
        hopeless.forward_pass(inputs,a);
        for (int j = 0; j < hopeless.neural_net.size(); j++)
        {
            forwardpass_states[i][j] = hopeless.neural_net[j].output;
        }
    }
    for (int i = 0; i < target.size(); i++)
    {
        if (i == recall_timestep - 1)
        {
            dl[i][0] += (forwardpass_states[i][hopeless.output_index[0]] - target[i][0]) * 4;  
        } 
    }  
    hopeless.bptt(forwardpass_states,dl);
}

void bias_reg(float param){
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        hopeless.neural_net[i].bias -= param * hopeless.neural_net[i].bias;
    }
    
}

void bias_noise(float sigma){
    std::normal_distribution<float> ra(0,sigma);
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        hopeless.neural_net[i].bias += ra(mtwister);
    }
    
}
/*
void random_search(std::uniform_int_distribution<int> re_tsp_dist,std::uniform_int_distribution<int> rtimestep, int timestep_gap){
    float ball_park_l = 0;
    for (int i = 0; i < 10000000; i++)
    {
        NN hope("seed.txt");
        hopeless = hope;
        hopeless.weight_noise(0.01);
        //bias_noise(0.0001);
        float avg_loss20000 = 0;
        for (int j = 0; j < 20000; j++)
        {
            int recordtimestp = re_tsp_dist(mtwister);
            int recalltimstp = recordtimestp + rtimestep(mtwister);
            avg_loss20000 += iteration(recordtimestp,recalltimstp);    
        }
        avg_loss20000 = avg_loss20000 / 20000;
        ball_park_l += avg_loss20000;
        if (i % 1000 == 999)
        {
            std::cout<<ball_park_l / 1000<<std::endl;
            ball_park_l = 0;
        }
        
        if (avg_loss20000 < 0.9)
        {
            for (int k = 0; k < 10; k++)
            {
                int recordtimestp = re_tsp_dist(mtwister);
                int recalltimstp = recordtimestp + timestep_gap;
                iteration(recordtimestp,recalltimstp,true);    
            }
            parameter_check();
            std::cout<<"average loss over 10000 iterations "<<avg_loss20000<<std::endl;    
            std::string save_filename;
            std::cout<<"Enter name of file to save to"<<std::endl;
            std::cin>>save_filename;
            hopeless.save(save_filename);
        }
    }
    
}
*/

float avg_loss(std::uniform_int_distribution<int> re_tsp_dist,std::uniform_int_distribution<int> rtimestep, int timestep_gap){
    float avg_loss10000 = 0;
    for (int i = 0; i < 10000; i++)
    {
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimstp = recordtimestp + timestep_gap;
        avg_loss10000 += iteration(recordtimestp,recalltimstp);        
    }  
    avg_loss10000 = avg_loss10000 / 10000;
    return avg_loss10000;
}

void prune_lowest(){
    float lowest = 20000000;
    int l_index;
    int neuron;
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        for (int j = 0; j < hopeless.neural_net[i].weights.size(); j++)
        {
            if(std::abs(hopeless.neural_net[i].weights[j].value) < lowest){
                lowest = std::abs(hopeless.neural_net[i].weights[j].value);
                l_index = hopeless.neural_net[i].weights[j].index;
                neuron = i;
            }
        }
    }
    hopeless.neural_net[neuron].weights.erase(hopeless.neural_net[neuron].weights.begin()  + l_index);
    hopeless.momentumW[neuron].pop_back();
    hopeless.weights_gradient[neuron].pop_back();
    hopeless.layermap_sync();
    hopeless.momentum_clear();
}

int main(){
    int timestep_gap;
    std::uniform_int_distribution<int> re_tsp_dist(3,10);
    int cycles;
    float learning_rate;
    float a = 0;
    float re_leak = 0;
    std::cout<<"number of cycles"<<std::endl;
    std::cin>>cycles;
    std::cout<<"learning rate"<<std::endl;
    std::cin>>learning_rate;
    std::cout<<"timesteps between record and recall"<<std::endl;
    std::cin>>timestep_gap;
    std::uniform_int_distribution<int> rtimestep(0,timestep_gap);
    //random_search(re_tsp_dist,rtimestep,timestep_gap);
    parameter_check();
    //prune_lowest();
    for (int i = 0; i < cycles; i++)
    {
        //learning_rate = learning_rate * 0.9995;
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimestp = recordtimestp + rtimestep(mtwister);    
        tr_iteration(recordtimestp,recalltimestp,a,re_leak);
        if ((i < cycles/2 )&& (i % 100 == 50))
        {
            hopeless.l1_reg(0.00001);
        }
        
        if (true)
        {
            if (i == 0)
            {
                for (int j = 0; j < hopeless.bias_gradient.size(); j++)
                {
                    hopeless.momentumB[j] = hopeless.bias_gradient[j];
                }
                for (int ii = 0; ii < hopeless.weights_gradient.size(); ii++)
                {
                    for (int j = 0; j < hopeless.weights_gradient[ii].size(); j++)
                    {
                        hopeless.momentumW[ii][j] = hopeless.weights_gradient[ii][j];
                    }
                    
                }
            }
            else
            {
                hopeless.update_momentum(0.9);
                hopeless.update_parameters(learning_rate);
                //hopeless.weight_noise(0.00001);
            }
        }
    }  

    float avg_loss10000 = 0;
    for (int i = 0; i < 10000; i++)
    {
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimstp = recordtimestp + timestep_gap;
        avg_loss10000 += iteration(recordtimestp,recalltimstp);
        if (i > 9990)
        {
            iteration(recordtimestp,recalltimstp,true);
        }
        
    }
    parameter_check();
    avg_loss10000 = avg_loss10000 / 10000;
    std::cout<<"average loss over 10000 iterations "<<avg_loss10000<<std::endl;    
    std::string save_filename;
    while (true)
    {
        std::cout<<"save? (y/n)"<<std::endl;
        char yn;
        std::cin>>yn;
        if (yn == 'y')
        {
            std::cout<<"Enter name of file to save to"<<std::endl;
            std::cin>>save_filename;
            hopeless.save(save_filename);
            return 0;
        }
        else if (yn == 'n')
        {
            return 0;
        }
        else{
            std::cout<<"ERROR, enter y or n"<<std::endl;
            Sleep(150);
        }
    }
}
