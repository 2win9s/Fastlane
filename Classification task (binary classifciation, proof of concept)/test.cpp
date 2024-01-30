//test on the SAheart data

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
inline void testset(relu_neural_network &model, dataset &data, std::vector<float> &results){
    std::fill(results.begin(),results.end(),0);
    // results stores 4 numbers, average loss, accuracy, sensitivity and specificity
    float t_pos = 0;
    float t_neg = 0;
    std::vector<float> input(8,0);
    std::vector<float> output(2,0);
    std::vector<float> target(2,0);
    float cu_loss = 0;
    for (int i = 0; i < data.chd.size(); i++)
    {
        input_vec(i,input,data);
        target_vec(data.chd[i],target);
        model.sforwardpass(input);
        for (int i = 0; i < model.output_index.size(); i++)
        {
            output[i]=model.relu_net[model.output_index[i]].units[15];
        }
        soft_max(output);
        int guess = prediction(output);
        if (data.chd[i] > 0)
        {
            t_pos +=1;
        }
        else{
            t_neg +=1;
        }
        if (guess)
        {
            if (guess == data.chd[i])
            {
                results[1] += 1;
                results[2] += 1;    //true positive
            }
            else{
                ;
            }
        }
        else{
            if (guess == data.chd[i])
            {
                results[1] += 1;
                results[3] += 1;    //true negative
            }
            else{
                ;
            }
        }

        
        results[0] += cross_entrophy(target,output);
    }
    results[0] = results[0] / data.chd.size();
    results[1] = results[1] / data.chd.size();
    results[2] = results[2] / t_pos;    // true positives/ all positives, all positives = true positives + false negatives
    results[3] = results[3] / t_neg;    // same logic fot specificity
}


int main(){
    dataset test("test.csv");
    relu_neural_network model("model1.txt");
    std::vector<float> results(4,0);
    testset(model,test,results);
    std::cout<<"The average cross entrophy loss is "<<results[0]<<std::endl;
    std::cout<<"The accuracy is "<< results[1]<<std::endl;
    std::cout<<"The sensitivity is "<< results[2]<<std::endl;
    std::cout<<"The specificity is "<< results[3]<<std::endl;
    return 0;
}   