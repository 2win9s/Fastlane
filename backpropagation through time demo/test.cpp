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

relu_neural_network genos("huh.txt");

int main(){
    vec_of_arr<float> states(1,genos.relu_net.size());
    int tstep = 0;
    float target = 0;
    while (true)
    {
        float in;
        std::cout<<"enter number to add"<<std::endl;
        std::cin>>in;
        target += in;
        std::vector<float> input(1,0);
        input[0] = in;
        genos.fpass(input,states,tstep);
        tstep+=1;
        std::vector<float> next(genos.relu_net.size(),0);
        states.app_arr(next);
        std::cout<<"target "<<target<<" ";
        std::cout<<"output "<<genos.relu_net[genos.output_index[0]].units[15]<<std::endl;
        std::cout<<std::endl;
    }
    return 0;
}