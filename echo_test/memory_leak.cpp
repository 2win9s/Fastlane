/*Very, very simple neural network that when given {x,1} as input will remember the number x and then recall back 
the number when given {0,2} , on the other timesteps the neural network is given {0,0} and outputs 0
although this is an execptionally simple recalling task, without even a need to ever "forget" a value
this neural network demonstrates that this architecture can indeed retain information 
over many timesteps and also that it can be trained using stochastic gradient descent (with great difficulty)*/

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
NN hopeless("memory_leak.txt"); //inputs {0,1], output{20}

//an iteration to measure the loss
float iteration(int record_timestep, int recall_timestep){
    float loss = 0;
    std::vector<float> inputs(2,0);
    std::vector<float> target(1,0);
    std::vector<float> output(1,0);
    float x;   //the number to be recalled
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);       //the value we want the neural net to remember
            inputs[1] = 1;                      //this is to tell the neural net to remember
            x = inputs[0];
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;                      //to tell the neural net to spit out the recorded value
            target[0] = x;
        }
        else{
            inputs[0] = 0;
            inputs[1] = 0;
            target[0] = 0;
        }
        hopeless.forward_pass(inputs);
        output[0] = hopeless.neural_net[hopeless.output_index[0]].output;
        loss += hopeless.loss(output,target,0);
    }
    //std::cout<<hopeless.neural_net[20].output<<std::endl;
    //std::cout<<target[0]<<std::endl;
    hopeless.neural_net_clear();
    return loss;
}

int main(){
    int timestep_gap;
    std::uniform_int_distribution<int> re_tsp_dist(0,100);        
    float avg_loss1000 = 0; 
    std::cout<<"timesteps between record and recall"<<std::endl;
    std::cin>>timestep_gap;
    for (int i = 0; i < 10000; i++)
    {
        int recordtimestp = re_tsp_dist(mtwister);
        int recalltimstp = recordtimestp + timestep_gap;
        avg_loss1000 += iteration(recordtimestp,recalltimstp);
    }
    avg_loss1000 = avg_loss1000 / 10000;
    std::cout<<"average loss over 1000 iterations "<<avg_loss1000<<std::endl;  
}