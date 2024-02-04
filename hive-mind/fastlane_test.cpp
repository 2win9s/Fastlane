// Quick test to see if everything is implemnted correctly and see the function approximating capability of neuron units

#include<vector>
#include<cmath>
#include<iostream>
#include<random>
#include <filesystem>
#include "version1.hpp"
#include <variant>
#include<array>
std::random_device ran;                          
std::mt19937 mtwister(ran());


float dmse(float out, float target){
    return (2 * (out - target));
}

float test_batch(relu_neural_network &n){
    float mse = 0;
    relu_neural_network::network_gradient momentum(n);
    relu_neural_network::network_gradient current(n);
    neural_net_record pre(2);
    neural_net_record post(2);
    for (int i = 0; i < 20000; i++)
    {
        std::uniform_real_distribution<float> r1t10(-1,1);
        float a = r1t10(mtwister);
        /* function of a to approximate note to self sqrt(negative number) = NaN*/
        float target =  std::sqrt(std::abs(a)) + std::sin(a) - std::cos(a);
        std::vector<float> input(1,a);
        n.sforwardpass(input,pre,post);
        std::vector<float> dloss(1,0);
        dloss[0] = 2*(n.relu_net[1].units[15]-target);
        mse += (n.relu_net[1].units[15]-target) * (n.relu_net[1].units[15]-target);
        //std::cout<<n.relu_net[1].units[15]<<std::endl;
        //std::cout<<target<<std::endl;
        //std::cout<<"---"<<std::endl;
        n.sbackpropagation(dloss,pre,post,current);
        momentum.sgd_with_momentum(n,0.001,0.9,current);
        current.valclear();
    }
    return mse/2000;
}

int main(){
    relu_neural_network genos("save.txt");
    std::normal_distribution<float> randw(0,1);
    float weight = randw(ttttt);
    genos.weights[1].emplace_back(relu_neural_network::index_value_pair(0,weight));
    genos.layermap_sync();
    std::cout<<"----------"<<std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout<<test_batch(genos)<<" mse of batch "<<i+1<<std::endl;
    }
    genos.save_to_txt("save.txt");
    return 0;
}
