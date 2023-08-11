#include"NN.hpp"

#include<windows.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<random>
#include<fstream>
#include<algorithm>

#include<omp.h>

std::ofstream report;
float a = 0.01;
float re_leak = 0.01;

const float max_val = 10000000000;
const float n_zero = 0.0001; //try and stop things from freaking out 

std::vector<std::string> characters = {" ","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","z","{","|","}","~"};

std::vector<std::string> data_points(60000);
std::vector<std::string> labels(60000);

std::uniform_int_distribution<int> randtrain(0,40000);   //the first 40thousand data points will be training data
std::uniform_int_distribution<int> randval(40001,59999);//this is the validation data

void load_data(){
    std::ifstream data("text.txt");
    std::ifstream label("label.txt");
    for (int i = 0; i < 60000; i++)
    {
        std::getline(data,data_points[i]);
    }
    data.close();
    for (int i = 0; i < 60000; i++)
    {
        std::getline(label,labels[i]);
    }
    label.close();
}

uint8_t input_convert(std::string s){
    for (uint8_t i = 0; i < characters.size(); i++)
    {
        if (s == characters[i])
        {
            return i;
        }    
    }
    return 0;   //default to whitespace if it isn't printable ascii    
}

void chara_convert7(bool (&chara)[7],uint8_t x){
    chara[0] = (1 & (x >> 0));
    chara[1] = (1 & (x >> 1));
    chara[2] = (1 & (x >> 2));
    chara[3] = (1 & (x >> 3));
    chara[4] = (1 & (x >> 4));
    chara[5] = (1 & (x >> 5));
    chara[6] = (1 & (x >> 6));
}

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

    report<<"parameters"<<"\n";
    report<<"mean weights "<<w_mean;
    report<<" variance of weights "<<w_variance<<"\n";
    report<<"mean bias "<<b_mean;
    report<<" variance of bias "<<b_variance<<"\n";
    report<<"number of weights"<<w_count<<"\n";
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

bool broken_float(float x){
    if (std::isnan(x)){
        return true;
    }
    else if (std::isinf(x)){
        return true;
    }
    else{
        return false;
    }
}

void soft_max(std::vector<float> &output){
    double denominator = 0;
    std::vector<float> expout(3);
    for (int i = 0; i < 3; i++)
    {
        expout[i] = std::exp(output[i]);
        if (broken_float(expout[i]))
        {
            expout[i] = max_val;
        }
        denominator += expout[i]; 
    }
    denominator = 1 / denominator; 
    for (int i = 0; i < 3; i++)
    {
        output[i] = expout[i] * denominator;
    }
}

void dl_softmax(std::vector<float> &output, std::vector<float> &target, std::vector<float> &dl){
    dl[0] = output[0] - target[0];
    dl[1] = output[1] - target[1];
    dl[2] = output[2] - target[2]; 

}

bool wow(std::vector<float> &output, std::vector<float> &target){
    int guess;
    int answer;
    guess = std::distance(output.begin(),std::max_element(output.begin(),output.end()));
    answer = std::distance(target.begin(),std::max_element(target.begin(),target.end()));
    if (guess == answer){return true;}
    else{return false;}
}

//what is the difference between cross entrophy and log likelihood? at this point I'm too afraid to ask
float soft_max_loss(std::vector<float> &output, std::vector<float> &target){
    double loss = 0;
    if (output[0] <= n_zero)
    {
        output[0] = n_zero;          
    }
    if (output[1] <= n_zero)
    {
        output[1] = n_zero;          
    }
    if (output[2] <= n_zero)
    {
        output[2] = n_zero;          
    }
    loss += target[0] * std::log(output[0]);
    loss += target[1] * std::log(output[1]);
    loss += target[2] * std::log(output[2]);
    loss  = -1 * loss;
    return loss;
}

float loss_iteration(int textindex, NN &hopeless){
    hopeless.neural_net_clear();
    float len = data_points[textindex].length();
    float loss;
    std::vector<float> inputs(70,0);
    std::vector<float> target(3,0);
    std::vector<float> output(3,0);
    if (labels[textindex] == "Positive")
    {
        target[0] = 1;
        target[1] = 0;
        target[2] = 0;
    }
    else if (labels[textindex] == "Negative")
    {
        target[0] = 0;
        target[1] = 1;
        target[2] = 0;
    }
    else if (labels[textindex] == "Neutral")
    {
        target[0] = 0;
        target[1] = 0;
        target[2] = 1;
    }
    else{
        std::cout<<"ERROR!!!!!!, label"<<std::endl;
        std::cout<<labels[textindex]<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < len; i+=10)
    {
        std::fill(inputs.begin(),inputs.end(),0);       //defaults to space if there is no character/ isn't ascii
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 10) < len) ? (i + 10):len); j++)
        {
            bool chara[7];
            std::string chra = "";
            chra += data_points[textindex][i];
            chara_convert7(chara,input_convert(chra));
            inputs[index + 0] = chara[0];
            inputs[index + 1] = chara[1];
            inputs[index + 2] = chara[2];
            inputs[index + 3] = chara[3];
            inputs[index + 4] = chara[4];
            inputs[index + 5] = chara[5];
            inputs[index + 6] = chara[6];
            index += 7;
        }
        hopeless.forward_pass(inputs,a);
    }
    for (int i = 0; i < hopeless.output_index.size(); i++)
    {
        output[i] = hopeless.neural_net[hopeless.output_index[i]].output;
    }
    soft_max(output);
    loss = soft_max_loss(target,output); 
    return loss;   
}

bool acc_iteration(int textindex, NN &hopeless){
    hopeless.neural_net_clear();
    float len = data_points[textindex].length();
    float loss = 0;
    std::vector<float> inputs(70,0);
    std::vector<float> target(3,0);
    std::vector<float> output(3,0);
    if (labels[textindex] == "Positive")
    {
        target[0] = 1;
        target[1] = 0;
        target[2] = 0;
    }
    else if (labels[textindex] == "Negative")
    {
        target[0] = 0;
        target[1] = 1;
        target[2] = 0;
    }
    else if (labels[textindex] == "Neutral")
    {
        target[0] = 0;
        target[1] = 0;
        target[2] = 1;
    }
    else{
        std::cout<<"ERROR!!!!!!, label"<<std::endl;
        std::cout<<labels[textindex]<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < len; i+=10)
    {
        std::fill(inputs.begin(),inputs.end(),0);       //defaults to space if there is no character/ isn't ascii
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 10) < len) ? (i + 10):len); j++)
        {
            bool chara[7];
            std::string chra = "";
            chra += data_points[textindex][i];
            chara_convert7(chara,input_convert(chra));
            inputs[index + 0] += chara[0];
            inputs[index +1] += chara[1];
            inputs[index +2] += chara[2];
            inputs[index +3] += chara[3];
            inputs[index +4] += chara[4];
            inputs[index +5] += chara[5];
            inputs[index +6] += chara[6];
            index += 7;
        }
        hopeless.forward_pass(inputs,a);
    }
    for (int i = 0; i < hopeless.output_index.size(); i++)
    {
        output[i] = hopeless.neural_net[hopeless.output_index[i]].output;
    }
    soft_max(output);
    bool correct;
    correct = wow(target,output);
    return correct;   
}

void tr_iteration(int textindex, NN &hopeless, float learning_rate){
    hopeless.neural_net_clear();
    float len = data_points[textindex].length();
    std::vector<float> inputs(70,0);
    std::vector<std::vector<float>> target(len);
    std::vector<std::vector<float>> output(len);
    std::vector<std::vector<float>> dl(len);
    std::vector<std::vector<float>> forwardpass_states(len);
    if (labels[textindex] == "Positive")
    {
        for (int i = 0; i < len; i++)
        {
            target[i].reserve(3);
            target[i].resize(3);
            target[i][0] = 1;
            target[i][1] = 0;
            target[i][2] = 0; 
        }
    }
    else if (labels[textindex] == "Negative")
    {
        for (int i = 0; i < len; i++)
        {
            target[i].reserve(3);
            target[i].resize(3);
            target[i][0] = 0;
            target[i][1] = 1;
            target[i][2] = 0; 
        }
    }
    else if (labels[textindex] == "Neutral")
    {
        for (int i = 0; i < len; i++)
        {
            target[i].reserve(3);
            target[i].resize(3);
            target[i][0] = 0;
            target[i][1] = 0;
            target[i][2] = 1; 
        }
    }
    else{
        std::cout<<"ERROR!!!!!!, label"<<std::endl;
        std::cout<<labels[textindex]<<std::endl;
        std::exit(EXIT_FAILURE);        
    }    
    for (int i = 0; i < len; i++)
    {
        output[i].reserve(3);
        output[i].resize(3,0);
    }
    for (int i = 0; i < len; i++)
    {
        dl[i].reserve(3);
        dl[i].resize(3,0);
    }
    for (int i = 0; i < len; i++)
    {
        forwardpass_states[i].reserve(hopeless.neural_net.size());
        forwardpass_states[i].resize(hopeless.neural_net.size(),0);
    }
    for (int i = 0; i < len; i+=10)
    {
        std::fill(inputs.begin(),inputs.end(),0);       //defaults to space if there is no character/ isn't ascii
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 10) < len) ? (i + 10):len); j++)
        {
            bool chara[7];
            std::string chra = "";
            chra += data_points[textindex][i];
            chara_convert7(chara,input_convert(chra));
            inputs[index + 0] += chara[0];
            inputs[index +1] += chara[1];
            inputs[index +2] += chara[2];
            inputs[index +3] += chara[3];
            inputs[index +4] += chara[4];
            inputs[index +5] += chara[5];
            inputs[index +6] += chara[6];
            index += 7;
        }
        hopeless.forward_pass(inputs,a);
        for (int j = 0; j < hopeless.neural_net.size(); j++)
        {
            forwardpass_states[i][j] = hopeless.neural_net[j].output;
        }

        for (int j = 0; j < hopeless.output_index.size(); j++)
        {
            output[i][j] = hopeless.neural_net[hopeless.output_index[j]].output;
        }        
    }

    soft_max(output[len -1]);

    dl_softmax(output[len -1], target[len -1], dl[len -1]);

    for (int j = 0; j < hopeless.output_index.size(); j++)
    {
        forwardpass_states[len -1][hopeless.output_index[j]] = output[len -1][j];
    }
    hopeless.bptt(forwardpass_states,dl,0,10);
    //hopeless.bptt_softsign_gradient(forwardpass_states,dl,learning_rate,0.9,0,10);
}

//average over 1000 iterations
float avg_loss_tr(NN &hopeless){
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    float loss = 0;
    for (int i = 0; i < 40000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(i,hopeless);
    }
    return loss/40000;
}

//average over 1000 iterations
float avg_loss_val(NN &hopeless){
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    float loss = 0;
    for (int i = 40000; i < 59999; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(i,hopeless);
    }
    return loss/20000;
}

//average over 300 iterations
float avg_acc_tr(NN &hopeless){
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    float acc = 0;
    for (int i = 0; i < 40000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        acc += acc_iteration(i,hopeless);
    }
    acc = acc/40000;
    return acc;
}

float avg_acc_val(NN &hopeless){
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    float acc = 0;
    for (int i = 40000; i < 59999; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        
        acc += acc_iteration(i,hopeless);
    }
    acc = acc / 20000;
    return acc;
}

void bias_reg(float param,NN &hopeless){
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        hopeless.neural_net[i].bias -= param * hopeless.neural_net[i].bias;
    }
    
}

void bias_noise(float sigma,NN &hopeless){
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    std::normal_distribution<float> ra(0,sigma);
    for (int i = 0; i < hopeless.neural_net.size(); i++)
    {
        hopeless.neural_net[i].bias += ra(mtwister);
    }
    
}



float He_initialisation(int n, float a){
    float w_variance = 2/ (n * (1 + a*a));  //I understand that this is absolutely not the correct use of this initialisation strategy but alas it might work
    return w_variance;
}



int main(){
    //omp_set_num_threads(8);
    //report.open("report.txt",std::fstream::trunc);
    load_data(); 
    std::vector<NN> hopeless(4);

    for (int i = 0; i < 1; i++){NN d("model.txt");hopeless[0] = d;}
    
    omp_set_num_threads(hopeless.size());
    #pragma omp parallel for
    for (int i = 1; i < hopeless.size(); i++)
    {
        hopeless[i] = hopeless[0];
    }
    int cycles;
    float l_r;
    std::cout<<"number of cycles"<<std::endl;
    std::cin>>cycles;
    std::cout<<"learning rate"<<std::endl;
    std::cin>>l_r; 
    /*
    for (int i = 0; i < 1; i++)
    {
        std::cout<<"Neurl net "<<i<<std::endl;
        report<<"Neurl net "<<i<<std::endl;
        parameter_check(hopeless[i]);
        float avg_1000_loss_tr_b = avg_loss_tr(hopeless[i]);
        std::cout<<"average loss on training dataset before training "<<avg_1000_loss_tr_b<<std::endl; 
        report<<"average loss on training dataset before training "<<avg_1000_loss_tr_b<<"\n";
        float avg_1000_loss_test_b = avg_loss_test(hopeless[i]);
        std::cout<<"average loss on test dataset before training "<<avg_1000_loss_test_b<<std::endl; 
        report<<"average loss on test dataset before training "<<avg_1000_loss_test_b<<"\n";
        float avg_300_acc_tr_b = avg_acc_tr(hopeless[i]);
        std::cout<<"average accuracy on training dataset before training "<<avg_300_acc_tr_b<<std::endl; 
        report<<"average accuracy on training dataset before training "<<avg_300_acc_tr_b<<"\n";
        float avg_300_acc_test_b = avg_acc_test(hopeless[i]);
        std::cout<<"average accuracy on test dataset before training "<<avg_300_acc_test_b<<std::endl; 
        report<<"average accuracy on test dataset before training "<<avg_300_acc_test_b<<"\n";
    }
    report.close(); */
    int progress = std::floor(cycles/10);
    for (int i = 0; i < cycles; i++)
    {   
        if (i % 100000 == 99999)
        {
            l_r = l_r * 0.75;
        }
        
        #pragma omp parallel
        {
            #pragma omp for
            for (int iiii = 0; iiii < hopeless.size(); iiii++)
            {
                std::random_device rd;                          
                std::mt19937 mtwister(rd());
                int randmsg = randtrain(mtwister);
                while (data_points[randmsg].size() == 0)
                {
                    randmsg = randtrain(mtwister);
                }
                tr_iteration(randmsg, hopeless[iiii],l_r); 
            }
            if (true)
            {
                
                if (i == 0)
                {
                    hopeless[0].momentum_clear();
                    for (int z = 0; z < hopeless.size(); z++)
                    {
                        for (int j = 0; j < hopeless[0].bias_g.size(); j++)
                        {
                            hopeless[0].momentumB[j] += hopeless[z].bias_g[j];
                        }
                        for (int ii = 0; ii < hopeless[z].weights_g.size(); ii++)
                        {
                            for (int j = 0; j < hopeless[z].weights_g[ii].size(); j++)
                            {
                                hopeless[0].momentumW[ii][j] += hopeless[z].weights_g[ii][j];
                            }                
                        }
                    }
                }
                else
                {  
                    for (int z = 1; z < hopeless.size(); z++)
                    {
                        for (int j = 0; j < hopeless[0].bias_g.size(); j++)
                        {
                            hopeless[0].bias_g[j] += hopeless[z].bias_g[j];
                        }
                        for (int ii = 0; ii < hopeless[z].weights_g.size(); ii++)
                        {
                            for (int j = 0; j < hopeless[z].weights_g[ii].size(); j++)
                            {
                                hopeless[0].weights_g[ii][j] += hopeless[z].weights_g[ii][j];
                            }                
                        }
                    }
                    for (int j = 0; j < hopeless[0].bias_g.size(); j++)
                    {
                        hopeless[0].bias_g[j] = hopeless[0].bias_g[j];
                    }
                    for (int ii = 0; ii < hopeless[0].weights_g.size(); ii++)
                    {
                        for (int j = 0; j < hopeless[0].weights_g[ii].size(); j++)
                        {
                            hopeless[0].weights_g[ii][j] = hopeless[0].bias_g[j];
                        }                
                    }
                    hopeless[0].update_momentum(0.9);
                    hopeless[0].update_parameters(l_r);
                    //hopeless[0].l2_reg(0.00001);
                    hopeless[0].gradient_clear();
                    for (int z = 1; z < hopeless.size(); z++)
                    {
                        hopeless[z] = hopeless[0];
                    }    
                }
            }        
        }
        if(i % progress == (progress - 1)){
        std::cout<<"#"<<std::flush;
        }
    }
    report.open("post_report.txt",std::fstream::trunc);
    std::cout<<"\n";  
    std::cout<<"training complete"<<std::endl;
    parameter_check(hopeless[0]);
    float avg_1000_loss_tr = avg_loss_tr(hopeless[0]);
    std::cout<<"average loss on training dataset  "<<avg_1000_loss_tr<<std::endl; 
    report<<"average loss on training dataset"<<avg_1000_loss_tr<<"\n";
    float avg_1000_loss_test = avg_loss_val(hopeless[0]);
    std::cout<<"average loss on validation dataset "<<avg_1000_loss_test<<std::endl;
    report<<"average loss on validation dataset "<<avg_1000_loss_test<<"\n";    
    float avg_300_acc_tr = avg_acc_tr(hopeless[0]);
    std::cout<<"average accuracy on training dataset  "<<avg_300_acc_tr<<std::endl; 
    report<<"average accuracy on training dataset "<<avg_300_acc_tr<<"\n"; 
    float avg_300_acc_test = avg_acc_val(hopeless[0]);
    std::cout<<"average accuracy on validation dataset"<<avg_300_acc_test<<std::endl;
    report<<"average accuracy on validation dataset "<<avg_300_acc_test<<"\n";
    report.close();
    std::string save_filename;
    while (true)
    {
        std::cout<<"save? (y/n) "<<std::endl;
        char yn;
        std::cin>>yn;
        if (yn == 'y')
        {
            std::cout<<"Enter name of file to save to"<<std::endl;
            std::cin>>save_filename;
            hopeless[0].save(save_filename);
            break;
        }
        else if (yn == 'n')
        {
            break;
        }
        else{
            std::cout<<"ERROR, enter y or n"<<std::endl;
            Sleep(150); //in case you hold onto a key and this message fills the terminal
        }
    }
    return 0;
}