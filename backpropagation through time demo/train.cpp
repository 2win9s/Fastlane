//training on the SAheart data

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
#include<filesystem>
#include<variant>
#include<omp.h>
#include<array>
#include"version1.hpp"
#include<sstream>


float dmse(float output, float target){
    return 2*(output - target);
}

// testing short term memory capabilities by making it do addition
float iteration(relu_neural_network &genos, relu_neural_network::network_gradient &past_grad){
    std::uniform_int_distribution<int> one_to_ten(20,30);
    const int steps = one_to_ten(ttttt);
    vec_of_arr<float> dloss(steps,genos.output_index.size());
    neural_net_record ff(genos.relu_net.size());
    std::vector<neural_net_record> pre(steps,ff);
    std::vector<neural_net_record> post(steps,ff);
    relu_neural_network::network_gradient f(genos);
    std::vector<relu_neural_network::network_gradient> net_grad(steps,f);
    vec_of_arr<float> states(steps,genos.relu_net.size());
    float target = 0;
    // mean 0 variance 1
    std::uniform_real_distribution<float> rand_num(-1.7320508,1.7320508);
    float mse = 0;
    for (int i = 0; i < steps; i++)
    {
        std::vector<float> input(1,0);
        input[0] = rand_num(ttttt);
        target += input[0];

        genos.fpass(input,pre[i],post[i],states,i);
        if (isnan(genos.relu_net[genos.output_index[0]].units[15]))
        {
            std::cout<<input[0]<<std::endl;
            std::cout<<std::endl;
            for (int j = 0; j < 5; j++)
            {
                std::cout<<genos.relu_net[j].units[0]<<std::endl;
                std::cout<<genos.relu_net[j].units[15]<<std::endl;
                std::cout<<post[i].values[j][0]<<std::endl;
                std::cout<<std::endl;
            }
            
            std::exit(0);
        }
        
        mse += (dmse(genos.relu_net[genos.output_index[0]].units[15],target) * dmse(genos.relu_net[genos.output_index[0]].units[15],target)) * 0.25;
        dloss(i,0) = dmse(genos.relu_net[genos.output_index[0]].units[15],target);
    }
    genos.bptt(dloss,pre,post,net_grad,states);
    for (int i = 1; i < net_grad.size(); i++)
    {
        for (int j = 0; j < net_grad[i].net_grads.size(); j++)
        {
            net_grad[0].net_grads[j].add(net_grad[i].net_grads[j]);
        }
        for (int j = 0; j < net_grad[i].weight_gradients.size(); j++)
        {
            for (int k = 0; k < net_grad[i].weight_gradients[j].size(); k++)
            {
                net_grad[0].weight_gradients[j][k] += net_grad[i].weight_gradients[j][k];
            }
            
        }
        
    }
    net_grad[0].restrict();
    past_grad.sgd_with_momentum(genos,0.000001,0.9,net_grad[0]);
    return mse;
}


int main(){
    relu_neural_network genos("huh1.txt");
    relu_neural_network::network_gradient past_grad(genos);
    for (int i = 0; i < 100; i++)
    {
        float cum_err = 0;
        for (int j = 0; j < 10000; j++)
        {
            cum_err += iteration(genos,past_grad);
        }
        std::cout<<"batch "<<i+1<<" MSE "<<cum_err<<std::endl;
    }
    for (int i = 0; i < past_grad.weight_gradients.size(); i++)
    {
        for (int j = 0; j < past_grad.weight_gradients[i].size(); j++)
        {
            if (past_grad.weight_gradients[i][j] == 0)
            {
                std::cout<<"problem"<<std::endl;
            }
            
        }
        
    }
    
    genos.save_to_txt("huh.txt");
    return 0;
}   
