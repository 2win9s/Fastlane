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
#include"version0.hpp"
#include<sstream>

struct dataset
{
    std::vector<float> sbp;
    std::vector<float> tobacco;
    std::vector<float> ldl;
    std::vector<float> adiposity;
    std::vector<float> typea;
    std::vector<float> obesity;
    std::vector<float> alcohol;
    std::vector<float> age;
    std::vector<float> chd; 
    dataset(std::string file_name){
        std::ifstream file(file_name);
        if (file.good()){;}else{std::cout<<"ERROR "<<file_name<<" does not exist"<<std::endl; std::exit(EXIT_FAILURE);}
        std::string line;
        std::getline(file, line); //skip col names
        while (std::getline(file, line))
        {
            std::istringstream lstream(line);
            std::string text;
            getline(lstream,text,',');
            getline(lstream,text,',');
            sbp.emplace_back(stof(text));

            getline(lstream,text,',');
            tobacco.emplace_back(stof(text));

            getline(lstream,text,',');
            ldl.emplace_back(stof(text));

            getline(lstream,text,',');
            adiposity.emplace_back(stof(text));

            getline(lstream,text,',');
            typea.emplace_back(stof(text));

            getline(lstream,text,',');
            obesity.emplace_back(stof(text));

            getline(lstream,text,',');
            alcohol.emplace_back(stof(text));

            getline(lstream,text,',');
            age.emplace_back(stof(text));

            getline(lstream,text,',');
            chd.emplace_back(stof(text));            
        }
    }
};

//Fisher-Yates shuffle
void shuffle_index(std::vector<int> &vec){
    std::uniform_int_distribution<int> rint(0,vec.size()-2);
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for(int i = 0; i < vec.size(); i++){
        int swappos = rint(mtwister);
        swappos = (swappos >= i) ? swappos+1:swappos;
        std::swap(vec[i],vec[swappos]);
    }
}

inline void target_vec(float chd, std::vector<float> &target_vec){
    if (chd > 0) //chd == 1, positive
    {
        target_vec[0] = 0;
        target_vec[1] = 1;
    }
    else        //negative
    {
        target_vec[0] = 1;
        target_vec[1] = 0;
    }
}

inline void input_vec(int index, std::vector<float> &input, dataset &data){
        input[0] = data.sbp[index];
        input[1] = data.tobacco[index];
        input[2] = data.ldl[index];
        input[3] = data.adiposity[index];
        input[4] = data.typea[index];
        input[5] = data.obesity[index];
        input[6] = data.alcohol[index];
        input[7] = data.age[index];
}

// softmax will be applied to the last layer for results
inline float soft_max_batch(relu_neural_network &model, dataset &data, relu_neural_network::network_gradient &momentum){
    std::vector<int> data_indice(data.alcohol.size());
    float cu_loss = 0;
    for (int i = 0; i < data_indice.size(); i++)
    {
        data_indice[i] = i;
    }
    shuffle_index(data_indice);
    std::vector<float> input(8,0);
    std::vector<float> output(2,0);
    std::vector<float> target(2,0);
    std::vector<float> dloss(2,0);
    relu_neural_network::network_gradient current(model);
    neural_net_record pre(model.relu_net.size());
    neural_net_record post(model.relu_net.size());
    for (int i = 0; i < data_indice.size(); i++)
    {
        input_vec(data_indice[i],input,data);
        target_vec(data.chd[data_indice[i]],target);
        model.sforwardpass(input,pre,post);
        for (int i = 0; i < model.output_index.size(); i++)
        {
            output[i]=model.relu_net[model.output_index[i]].units[15];
        }
        //std::cout<<output[0]<<std::endl;
        soft_max(output);
        //std::cout<<output[0]<<std::endl;
        dsoft_max(output,target,dloss);
        cu_loss += cross_entrophy(target,output);
        model.sbackpropagation(dloss,pre,post,current);
        //current.valclear();
    }
    momentum.sgd_with_momentum(model,0.00000001,0.96,current);
    return cu_loss/data_indice.size();
}


int main(){
    dataset train("SAheart.csv");
    relu_neural_network model("model1.txt");
    relu_neural_network::network_gradient momentum(model);
    for (int i = 0; i < 5000000; i++)
    {
        float s = soft_max_batch(model,train,momentum);
        if (i%5000 == 0)
        {
            std::cout<<s<<" mean cross entrophy of batch "<<i+1<<std::endl;
        }
        if (i%50000 == 0)
        {
            model.save_to_txt("model2.txt");
        }
        
    }
    model.save_to_txt("model2.txt");
    return 0;
}   