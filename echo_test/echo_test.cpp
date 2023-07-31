#include"NN.hpp"

#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>

std::random_device rd;                          
std::mt19937 mtwister(rd());
std::uniform_real_distribution<float> rand06(0,6);      //the neural net is using ReLU capped at 7 as the activation function for all neurons, so this range is to ensure it can be expressed

//the name of the neural network is hopeless, just like its architecture
NN hopeless("testing.txt"); //inputs {0,1], output{20}


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
float iteration(int record_timestep, int recall_timestep){
    float loss;
    std::vector<float> inputs(2,0);
    std::vector<float> target(1,0);
    std::vector<float> output(1,0);
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);       //the value we want the neural net to remember
            inputs[1] = 1;                      //this is to tell the neural net to remember
            target[0] = inputs[0];
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;                      //to tell the neural net to spit out the recorded value
        }
        else{
            std::fill(inputs.begin(),inputs.end(),0);
        }
        hopeless.forward_pass(inputs);
    }
    output[0] = hopeless.neural_net[20].output;
    hopeless.neural_net_clear();
    loss = hopeless.loss(output,target,0);
    return loss;
}

void tr_iteration(int record_timestep, int recall_timestep, float learning_rate, float a, float re_leak){
    std::vector<float> inputs(2,0);
    std::vector<std::vector<float>> target(recall_timestep);
    std::vector<std::vector<float>> forwardpass_states(recall_timestep);
    for (int i = 0; i < forwardpass_states.size(); i++)
    {
        forwardpass_states[i].reserve(hopeless.neural_net.size());
        forwardpass_states[i].resize(hopeless.neural_net.size(),0);
    }
    for (int i = 0; i < target.size(); i++)
    {
        target[i].reserve(1);
        target[i].resize(1,0);
    }
    
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);
            inputs[1] = 1;
            target[recall_timestep - 1][0] = inputs[0];
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;
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
    hopeless.neural_net_clear();
    hopeless.bptt(forwardpass_states,target,learning_rate);
}

int main(){
    int timestep_gap;
    std::uniform_int_distribution<int> re_tsp_dist(0,3);
    int cycles;
    float learning_rate;
    float a;
    float re_leak;
    std::cout<<"number of cycles"<<std::endl;
    std::cin>>cycles;
    std::cout<<"learning rate"<<std::endl;
    std::cin>>learning_rate;
    std::cout<<"timesteps between record and recall"<<std::endl;
    std::cin>>timestep_gap;
    for (int i = 0; i < cycles; i++)
    {
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimestp = recordtimestp + timestep_gap;
        tr_iteration(recordtimestp,recalltimestp,learning_rate,a,re_leak);
    }    
    parameter_check();
    float avg_loss1000 = 0;
    for (int i = 0; i < 1000; i++)
    {
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimstp = recordtimestp + timestep_gap;
        avg_loss1000 += iteration(recordtimestp,recalltimstp);
    }
    avg_loss1000 = avg_loss1000 / 1000;
    std::cout<<"average loss over 1000 iterations "<<avg_loss1000<<std::endl;    
    std::string save_filename;
    std::cout<<"Enter name of file to save to"<<std::endl;
    std::cin>>save_filename;
    hopeless.save(save_filename);
    return 0;
}