#include"NN.hpp"
#include<iostream>
#include<random>

std::random_device rd;                          
std::mt19937 mtwister(rd());


void parameter_check(NN &hopeless){
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


float He_initialisation(int n, float a = 0){
    float w_variance = 2/ (n * (1 + a*a));  //I understand that this is absolutely not the correct use of this initialisation strategy but alas it might work
    return w_variance;
}

int main(){
    std::vector<int> input_vec(70,0);
    for(int i = 0; i < input_vec.size(); i++){
        input_vec[i] = i;
    }
    std::vector<int> mem_vec(100,0);
    for (int i = 0; i < mem_vec.size(); i++)
    {
        mem_vec[i] = 2023 + i;
    }
    NN network(4277,input_vec,{4274,4275,4276},mem_vec,0.003,0.0015);
    network.ensure_connection(100,true,true,true,false);
    for (int i = 0; i < network.neural_net.size(); i++)
    {
        if (network.neural_net[i].memory)
        {
            network.neural_net[i].activation_function_v = 1;        //ReLU6
            continue;
        }
        network.neural_net[i].activation_function_v = 0;            //GELU
        int fan_in = 0;
        for (int j = 0; j < network.neural_net[i].weights.size(); j++)
        {
            if (network.neural_net[i].weights[j].index > i)
            {
                continue;
            }
            
            fan_in++;
        }
        std::normal_distribution<float> weight_dist(0,std::sqrt(He_initialisation(fan_in)));
        for (int j = 0; j < network.neural_net[i].weights.size(); j++)
        {
            if (network.neural_net[i].weights[j].index > i)
            {
                continue;
            }
            
            network.neural_net[i].weights[j].value = weight_dist(mtwister);
        }
    }
    for (int i = 0; i < network.output_index.size(); i++)
    {
        network.neural_net[network.output_index[i]].activation_function_v = 2;  //linear (softmax layer)
    }
    
    parameter_check(network);
    network.save("model.txt");
    return 0;
}