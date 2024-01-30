// Quick test to see if everything is implemnted correctly and see the function approximating capability of neuron units

#include<vector>
#include<cmath>
#include<iostream>
#include<random>
#include <filesystem>
#include "version0.hpp"
#include <variant>
#include<array>
std::random_device ran;                          
std::mt19937 mtwister(ran());


float dmse(float out, float target){
    return (2 * (out - target));
}

float test_batch(relu_neuron &n){
    float mse = 0;
    std::array<float,16> dacts;
    std::array<float,16> mem;
    neuron_gradients grads;
    neuron_gradients new_grads;
    grads.valclear();
    new_grads.valclear();
    for (int i = 0; i < 20000; i++)
    {
        std::uniform_real_distribution<float> r1t10(-1,1);
        float a = r1t10(mtwister);


        /* function of a to approximate note to self sqrt(negative number) = NaN*/
        float target =  std::sqrt(std::abs(a)) + std::sin(a) - std::cos(a);
        forwardpass(n,a,dacts);
        for (int j = 0; j < 16; j++)
        {
            mem[j] = n.units[j];
        }
        mse += (n.units[15] - target) * (n.units[15] - target);
        
        //for seeing if things go 
        bool flag = false;
        for (int i = 0; i < 16; i++)
        {
            if (std::isnan(mem[i]))
            {
                flag = true;
                std::cout<<"nan in unit "<<i<<std::endl;
            }
            
        }
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 7; j++)
            {
                if (std::isnan(n.weights[i][j]))
                {
                    flag = true;
                    std::cout<<"nan in weights"<<std::endl;
                }
                
            }
            
        }
        if (flag)
        {
            for (int i = 0; i < 9; i++)
            {
                float mean =0;
                float var  =0;
                std::cout<<"[";
                for (int j = 0; j < 6; j++)
                {
                    mean += n.weights[i][j];
                    std::cout<<n.weights[i][j]<<",";
                }
                mean +=n.weights[i][6];
                mean = mean /7;
                std::cout<<n.weights[i][6];
                for (int j = 0; j < 7; j++)
                {
                    var += (mean - n.weights[i][j])*(mean - n.weights[i][j]);
                }
                var = var/7;
                std::cout<<"]"<<"("<<mean<<","<<var<<")"<<std::endl;
            }
            std::exit(0);
        }
        backprop(n,dmse(n.units[15],target),mem,dacts,new_grads);
        grads.sgd_with_momentum(n,0.01,0.7,new_grads);
        new_grads.valclear();
    }
    return mse/2000;
}

int main(){
    relu_neuron genos;
    std::cout<<"initial weights"<<std::endl;
    relu_neuron start_point = genos;
    for (int i = 0; i < 9; i++)
        {
            float mean =0;
            float var  =0;
            std::cout<<"[";
            for (int j = 0; j < 6; j++)
            {
                mean += start_point.weights[i][j];
                std::cout<<start_point.weights[i][j]<<",";
            }
            mean +=start_point.weights[i][6];
            mean = mean /7;
            std::cout<<start_point.weights[i][6];
            for (int j = 0; j < 7; j++)
            {
                var += (mean - start_point.weights[i][j])*(mean - start_point.weights[i][j]);
            }
            var = var/7;
            std::cout<<"]"<<"("<<mean<<","<<var<<")"<<std::endl;
    }
    std::cout<<"----------"<<std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout<<test_batch(genos)<<" mse of batch "<<i+1<<std::endl;
    }
    return 0;
}