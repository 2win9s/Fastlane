#include"NN.hpp"

#include<windows.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>
#include<fstream>
#include<algorithm>
#include<numeric>
#include<omp.h>




NN stoopid("alter.txt");

int main(){
    stoopid.new_weights_s(100000);
    stoopid.save("alter.txt");
    return 0;
}