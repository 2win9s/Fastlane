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
    std::cout<<"number of weights "<<w_count<<std::endl;
}

/*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (Kaiming He et Al.) : https://arxiv.org/abs/1502.01852v1*/
float He_initialisation(int n, float a = 0){
    float w_variance = 2/ (n * (1 + a*a));  //I understand that this is absolutely not the correct use of this initialisation strategy but alas it might work
    return w_variance;
}

int main(){
    std::vector<int> input_vec(380,0);
    for(int i = 0; i < input_vec.size(); i++){
        input_vec[i] = i;
    }
    std::vector<int> mem_vec ={};
    for (int i = 7677; i < 7777; i++)
    {
        mem_vec.emplace_back(i);
    }
    NN network(7777,input_vec,{7674,7675,7676},mem_vec,0,0);
    std::cout<<"\n"<<std::endl;
    for (int i = input_vec.size(); i < (input_vec.size() + 400); i++)
    {
        for (int j = 0; j < input_vec.size(); j++)
        {
            network.neural_net[i].weights.emplace_back(j,0);
        }
    }
    for (int i = input_vec.size() + 400; i < 7674; i += 400)
    {
        for (int j = i; j < (((i + 400)< 7674) ? (i + 400):7674); j++)
        {
            for(int k = i - 400; k < i; k++){
                network.neural_net[j].weights.emplace_back(k,0);
            }
        }
    }
    for (int i = 7674; i < 7677; i++)
    {
        for (int j = (7674 - 400); j < 7674; j++)
        {
            network.neural_net[i].weights.emplace_back(j,0);
        }
    }
    network.new_weights_s(300000);
    network.new_rweights_s(300000);
    network.new_imweights_s(20000);
    network.new_omweights_s(20000);
    for (int i = 0; i < network.neural_net.size(); i++)
    {
        network.neural_net[i].activation_function_v = 1;                 //GeLU
        int fan_in = network.neural_net[i].weights.size();
        std::normal_distribution<float> weight_dist(0,std::sqrt(He_initialisation(fan_in)));
        //Understanding the difficulty of training deep feedforward neural networks (Glorot et. al) https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        std::uniform_real_distribution<float> xavier_dist(-(1.0/network.neural_net[i].weights.size()),(1.0/network.neural_net[i].weights.size()));
        //float multiplier = 1.0 / std::sqrt(network.layermap.size());
        for (int j = 0; j < network.neural_net[i].weights.size(); j++)
        {
            network.neural_net[i].weights[j].value = xavier_dist(mtwister);
        }
    }
    for (int i = 0; i < network.output_index.size(); i++)
    {
        network.neural_net[network.output_index[i]].activation_function_v = 2;
    }
    network.layermap_sync();
    parameter_check(network);
    network.save("alter.txt");
    return 0;
}