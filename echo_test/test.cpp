/*input neurons are 0,1,2, the output neuron is {21} and the "memory" neuron is {11}
when the input is {x,1,0}, the neural net remembers x and the output will be x when the input is {0,2,0}
to forget the number x set the input to {0,0,6} before {y,1,0} is given as input to forget x and then record y*/


#include"NN.hpp"
#include<windows.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>

std::random_device rd;                          
std::mt19937 mtwister(rd());
std::uniform_real_distribution<float> rand06(0,6);      //the neural net is using ReLU capped at 6 as the activation function for all neurons, so this range is to ensure it can be expressed

//the name of the neural network is hopeless, just like its architecture
NN hopeless("echo.txt"); 

//an iteration to measure the loss
float iteration(int record_timestep, int recall_timestep,bool print = false){
    float loss;
    std::vector<float> inputs(3,0);
    std::vector<float> target(1);
    std::vector<float> output(1,0);
    std::uniform_int_distribution<int> randint10(0,10);
    
    int forget_timestep = randint10(mtwister);  //choosing a random time before recording the input to forget the one currently remembered
    
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);       //the value we want the neural net to remember
            inputs[1] = 1;                      //this is to tell the neural net to remember
            target[0] = inputs[0];
            inputs[2] = 0;
        }
        else if (i == record_timestep - 2 - forget_timestep)
        {
            inputs[0] = 0;
            inputs[1] = 0;                      //here it should "forget"
            inputs[2] = 6;
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;                      //to tell the neural net to spit out the recorded value
            inputs[2] = 0;
        }
        else{
            inputs[0] = 0;       //neural net does nothing
            inputs[1] = 0;
            inputs[2] = 0;

        }

        //this is the forward pass
        hopeless.forward_pass(inputs);


    }

    //reading the output
    output[0] = hopeless.neural_net[hopeless.output_index[0]].output;
    loss = (output[0] - target[0]) * (output[0] - target[0]);  //squared error
    if (print)
    {
        std::cout<<"output "<<output[0]<<"\n";
        std::cout<<"target "<<target[0]<<"\n";
        std::cout<<std::endl;
    }
    
    return loss;
}

float avg_loss(std::uniform_int_distribution<int> re_tsp_dist, int timestep_gap){
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

int main(){
    //a random timestep to decide when after the start of an iteration to record a value
    std::uniform_int_distribution<int> re_tsp_dist(12,20);
    float a = 0;
    float re_leak = 0;

    //the number of timesteps between recording and recalling
    int timestep_gap;
    std::cout<<"timesteps between record and recall"<<std::endl;
    std::cin>>timestep_gap;
    
    
    float loss = avg_loss(re_tsp_dist,timestep_gap);
    std::cout<<"average loss over 10000 iterations "<<loss<<std::endl;
    std::cout<<"Hit ENTER to close"<<std::endl;
    std::cin.sync(); 
    std::cin.get();
    return 0;
}
