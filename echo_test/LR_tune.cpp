#include"NN.hpp"

#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>
#include<fstream>


std::random_device rd;                          
std::mt19937 mtwister(rd());
//this generates a value for the Neural Net to Recall
std::uniform_real_distribution<float> rand06(0,6);      //the neural net is using ReLU capped at 7 as the activation function for all neurons, so this range is to ensure it can be expressed

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

void momentum_check(NN &hopeless){
    double w_moment = 0;
    double b_moment = 0;
    for (int i = 0; i < hopeless.momentumW.size(); i++)
    {
        b_moment += hopeless.momentumB[i];
        for(int j = 0; j < hopeless.momentumW[i].size(); j++){
            w_moment += hopeless.momentumW[i][j];
        }
    }
    std::cout<<w_moment<<std::endl;
    std::cout<<b_moment<<std::endl;
}

//an iteration to measure the loss
float iteration(int record_timestep, int recall_timestep, NN &hopeless ,float a){
    float loss;
    std::vector<float> inputs(2,0);
    std::vector<float> target(1,0);
    std::vector<float> output(1,0);
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);       //the value we want the neural net to remember
            inputs[1] = 1;                      //this is to tell the neural net to remember
            target[0] = inputs[0];
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;                      //to tell the neural net to spit out the recorded value
        }
        else{
            std::fill(inputs.begin(),inputs.end(),0);
        }
        hopeless.forward_pass(inputs, a);
    }
    output[0] = hopeless.neural_net[20].output;
    hopeless.neural_net_clear();
    loss = hopeless.loss(output,target,0);
    return loss;
}

void tr_iteration(int record_timestep, int recall_timestep, float learning_rate, float a, float re_leak, NN &hopeless){
    std::vector<float> inputs(2,0);
    std::vector<std::vector<float>> target(recall_timestep);
    std::vector<std::vector<float>> forwardpass_states(recall_timestep);
    for (int i = 0; i < forwardpass_states.size(); i++)
    {
        forwardpass_states[i].reserve(hopeless.neural_net.size());
        forwardpass_states[i].resize(hopeless.neural_net.size(),0);
    }
    for (int i = 0; i < target.size(); i++)
    {
        target[i].reserve(1);
        target[i].resize(1,0);
    }
    
    for (int i = 0; i < recall_timestep; i++)
    {
        if (i == record_timestep - 1)
        {
            inputs[0] = rand06(mtwister);
            inputs[1] = 1;
            target[recall_timestep - 1][0] = inputs[0];
        }
        else if (i == recall_timestep - 1)
        {
            inputs[0] = 0;
            inputs[1] = 2;
        }
        else
        {
            std::fill(inputs.begin(),inputs.end(),0);
        }
        hopeless.forward_pass(inputs,a);
        for (int j = 0; j < hopeless.neural_net.size(); j++)
        {
            forwardpass_states[i][j] = hopeless.neural_net[j].output;
        }
    }
    hopeless.neural_net_clear();
    hopeless.bptt(forwardpass_states,target,learning_rate);
}

void save_lr_loss(std::vector<float> & loss_vec, std::vector<float> &lr_vec , std::string hyperparameter_s){
    std::ofstream file("lr_range_data.txt");
    file << "learning rate and loss data" << "\n";
    file << hyperparameter_s;
    file << "format: learning rate , loss"<<"\n";
    for (int i = 0; i < loss_vec.size(); i++)
    {
        file << loss_vec[i] << " , " << lr_vec[i] << "\n";
    }
}

void lR_range_tester(int step_number, float starting_rate, std::string linear_or_exponential, float step_parameter, int time_step_gap , int cycle_per_LR, float Re_leak){
    if (linear_or_exponential == "linear")
    {
        std::vector<float> loss_vec(step_number,0);
        std::vector<float> learning_rate(step_number,0);
        loss_vec.shrink_to_fit();
        learning_rate.shrink_to_fit();
        float current_lr = starting_rate;
        std::cout<<"Testing Begins...";
        for (int i = 0; i < step_number; i++)
        {
            std::cout<<"\r"<<std::flush;
            std::cout<<"Step "<<i + 1<< " of "<< step_number<<std::flush;
            NN hopeless("testing.txt"); //inputs {0,1], output{20} , fully connected
            learning_rate[i] = current_lr;
            current_lr += step_parameter * starting_rate;       //linear increase
            for (int j = 0; j < cycle_per_LR; j++)
            {
                std::uniform_int_distribution<int> re_tsp_dist(0,5);
                int record_tsp = re_tsp_dist(mtwister);
                int recall_tsp = record_tsp + time_step_gap;
                tr_iteration(record_tsp,recall_tsp,current_lr,Re_leak, Re_leak, hopeless);
            }
            float average_loss1000 = 0;
            for (int j = 0; j < 1000; j++)
            {
                std::uniform_int_distribution<int> re_tsp_dist(0,5);
                int record_tsp = re_tsp_dist(mtwister);
                int recall_tsp = record_tsp + time_step_gap;
                average_loss1000 += iteration(record_tsp,recall_tsp,hopeless, Re_leak);
            }
            average_loss1000 = average_loss1000 / 1000;
            loss_vec[i] = average_loss1000;
        }
        std::string hyperparameter_summary = "";
        hyperparameter_summary = "steps " + std::to_string(step_number) + ", base learning rate " + std::to_string(starting_rate)
         + ",step_type " + linear_or_exponential + "\n" + "step_size " + std::to_string(step_parameter) + ", timestep_gap " + 
         std::to_string(time_step_gap) + ", iterations at each step " + std::to_string(cycle_per_LR) + "\n" + "Leaky_ReLU_param " 
         + std::to_string(Re_leak) + "\n";
        save_lr_loss(loss_vec,learning_rate, hyperparameter_summary);
    }
    else if (linear_or_exponential == "exponential")
    {
        std::vector<float> loss_vec(step_number,0);
        std::vector<float> learning_rate(step_number,0);
        loss_vec.shrink_to_fit();
        learning_rate.shrink_to_fit();
        float current_lr = starting_rate;
        std::cout<<"Testing Begins...";
        for (int i = 0; i < step_number; i++)
        {
            std::cout<<"\r"<<std::flush;
            std::cout<<"Step "<<i + 1<< " of "<< step_number<<std::flush;
            NN hopeless("testing.txt"); //inputs {0,1], output{20} , fully connected
            learning_rate[i] = current_lr;
            current_lr += (step_parameter + 1);       //linear increase
            for (int j = 0; j < cycle_per_LR; j++)
            {
                std::uniform_int_distribution<int> re_tsp_dist(0,5);
                int record_tsp = re_tsp_dist(mtwister);
                int recall_tsp = record_tsp + time_step_gap;
                tr_iteration(record_tsp,recall_tsp,current_lr,Re_leak,Re_leak,hopeless);
            }
            float average_loss1000 = 0;
            for (int j = 0; j < 1000; j++)
            {
                std::uniform_int_distribution<int> re_tsp_dist(0,5);
                int record_tsp = re_tsp_dist(mtwister);
                int recall_tsp = record_tsp + time_step_gap;
                average_loss1000 += iteration(record_tsp,recall_tsp,hopeless,Re_leak);
            }
            average_loss1000 = average_loss1000 / 1000;
            loss_vec[i] = average_loss1000;
        }
        std::string hyperparameter_summary = "";
        hyperparameter_summary = "steps " + std::to_string(step_number) + ", base learning rate " + std::to_string(starting_rate)
         + ",step_type " + linear_or_exponential + "\n" + "step_size " + std::to_string(step_parameter) + ", timestep_gap " + 
         std::to_string(time_step_gap) + ", iterations at each step " + std::to_string(cycle_per_LR) + "\n" + "Leaky_ReLU_param " 
         + std::to_string(Re_leak) + "\n";
        save_lr_loss(loss_vec,learning_rate, hyperparameter_summary);
    }
    else
    {
        std::cout<<"ERROR: invalid argument";
        std::exit(EXIT_FAILURE);
    }
    
}




int main(){
    int steps = 1000;
    int base_rate = 0.00001;
    std::string step_type = "exponential";
    int step_size = 0.01;
    int recall_gap = 100;
    int cycles = 1000000;
    float Releak = 0.01;
    lR_range_tester(steps,base_rate,step_type,step_size,recall_gap,cycles,Releak);
    return 1;
}

