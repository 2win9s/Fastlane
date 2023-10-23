#include"NN.hpp"
#include<iomanip>
#include<windows.h>
#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include<string>
#include<random>
#include<fstream>
#include<algorithm>
#include<numeric>
#include<omp.h>

NN stoopid("alter.txt");
static const int t_hreads = 16; //set no less than 4, else might segfault

float changing_gradient_limit = 10;

float a = 0;
float re_leak = 0;

const float max_val = 10000000000;
const float n_zero = 0.0001; //try and stop things from freaking out 

std::vector<std::string> data_points(60000);
std::vector<std::string> labels(60000);
std::vector<int> tr_data_index(40000);
std::vector<int> tr_data_sample(21000);     //training data is imbalanced

std::uniform_int_distribution<int> randtrain(0,39999);   //the first 40thousand data points will be training data
std::uniform_int_distribution<int> randval(40000,50000); //this is the validation data

thread_local std::random_device r_dev;                          
thread_local std::mt19937 mtwister(r_dev());

std::ofstream report;


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

//Fisher-Yates shuffle
void shuffle_tr_data(){
    std::uniform_int_distribution<int> rint(0,39998);
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for(int i = 0; i < 40000; i++){
        int swappos = rint(mtwister);
        swappos = (swappos >= i) ? swappos+1:swappos;
        std::swap(tr_data_index[i],tr_data_index[swappos]);
    }
    //we now sample from that shuffled training data to recify the imbalance
    int pcount = 0;
    int necount = 0;
    int neucount = 0;
    int index = 0;
    for(int i = 0; i < tr_data_index.size(); i++){
        if (( (labels[tr_data_index[i]] == "Neutral") || (data_points[tr_data_index[i]].length() < 3)) &&(neucount <= 7000))
        {
            tr_data_sample[index] = tr_data_index[i];
            index++;
            neucount++;
        }
        else if ((labels[tr_data_index[i]] == "Positive") && (pcount <= 7000))
        {
            tr_data_sample[index] = tr_data_index[i];
            index++;
            pcount++;
        }
        else if ((labels[tr_data_index[i]] == "Negative") && (necount <= 7000))
        {
            tr_data_sample[index] = tr_data_index[i];
            index++;
            necount++;
        }
        if ((pcount + necount + neucount) ==  21000)
        {
            break;
        } 
    }
    
    std::uniform_int_distribution<int> raint(0,tr_data_sample.size()-2);
    for (int i = 0; i < tr_data_sample.size(); i++)
    {
        int swappos = raint(mtwister);
        swappos = (swappos >= i) ? swappos+1:swappos;
        std::swap(tr_data_sample[i],tr_data_sample[swappos]);
    }
    
}

inline uint8_t input_convert(char s){
    /*
    static const std::array<char,95> characters = {' ','!','\"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
        '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K',
        'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g',
        'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','w','y','z','{','|','}','~'};           //the 95 printable ASCII charaters
    
    //I'm not about to write out a switch case statement with 95 cases, this'll do
    for (uint8_t i = 0; i < 95; i++)
    {
        if (s == characters[i])
        {
            return i;
        }    
    }
    */
    switch (s){
        case ' ':
            return 0;
        case '!':
            return 1;
        case '\"':
            return 2;
        case '#':
            return 3;
        case '$':
            return 4;
        case '%':
            return 5;
        case '&':
            return 6;
        case '\'':
            return 7;
        case '(':
            return 8;
        case ')':
            return 9;
        case '*':
            return 10;
        case '+':
            return 11;
        case ',':
            return 12;
        case '-':
            return 13;
        case '.':
            return 14;
        case '/':
            return 15;
        case '0':
            return 16;
        case '1':
            return 17;
        case '2':
            return 18;
        case '3':
            return 19;
        case '4':
            return 20;
        case '5':
            return 21;
        case '6':
            return 22;
        case '7':
            return 23;
        case '8':
            return 24;
        case '9':
            return 25;
        case ':':
            return 26;
        case ';':
            return 27;
        case '<':
            return 28;
        case '=':
            return 29;
        case '>':
            return 30;
        case '?':
            return 31;
        case '@':
            return 32;
        case 'A':
            return 33;
        case 'B':
            return 34;
        case 'C':
            return 35;
        case 'D':
            return 36;
        case 'E':
            return 37;
        case 'F':
            return 38;
        case 'G':
            return 39;
        case 'H':
            return 40;
        case 'I':
            return 41;
        case 'J':
            return 42;
        case 'K':
            return 43;
        case 'L':
            return 44;
        case 'M':
            return 45;
        case 'N':
            return 46;
        case 'O':
            return 47;
        case 'P':
            return 48;
        case 'Q':
            return 49;
        case 'R':
            return 50;
        case 'S':
            return 51;
        case 'T':
            return 52;
        case 'U':
            return 53;
        case 'V':
            return 54;
        case 'W':
            return 55;
        case 'X':
            return 56;
        case 'Y':
            return 57;
        case 'Z':
            return 58;
        case '[':
            return 59;
        case '\\':
            return 60;
        case ']':
            return 61;
        case '^':
            return 62;
        case '_':
            return 63;
        case '`':
            return 64;
        case 'a':
            return 65;
        case 'b':
            return 66;
        case 'c':
            return 67;
        case 'd':
            return 68;
        case 'e':
            return 69;
        case 'f':
            return 70;
        case 'g':
            return 71;
        case 'h':
            return 72;
        case 'i':
            return 73;
        case 'j':
            return 74;
        case 'k':
            return 75;
        case 'l':
            return 76;
        case 'm':
            return 77;
        case 'n':
            return 78;
        case 'o':
            return 79;
        case 'p':
            return 80;
        case 'q':
            return 81;
        case 'r':
            return 82;
        case 's':
            return 83;
        case 't':
            return 84;
        case 'u':
            return 85;
        case 'v':
            return 86;
        case 'w':
            return 87;
        case 'x':
            return 88;
        case 'y':
            return 89;
        case 'z':
            return 90;
        case '{':
            return 91;
        case '|':
            return 92;
        case '}':
            return 93;
        case '~':
            return 94;
        default:
            return 0;
    }
    return 0;   //default to whitespace if it isn't printable ascii    
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
    float avg_abs_alpha=0;
    for (int i = 0; i < stoopid.neural_net.size(); i++)
    {
        avg_abs_alpha+=std::abs(stoopid.neural_net[i].alpha);
    }
    avg_abs_alpha = avg_abs_alpha/stoopid.neural_net.size();
    std::cout<<"average absolute value of alpha "<<avg_abs_alpha<<"\n";
    std::cout<<"mean weights "<<w_mean;
    std::cout<<" variance of weights "<<w_variance<<"\n";
    std::cout<<"mean bias "<<b_mean;
    std::cout<<" variance of bias "<<b_variance<<"\n";
    std::cout<<"number of weights "<<w_count<<std::endl;

    report<<"average absolute value of alpha "<<avg_abs_alpha<<"\n";
    report<<"mean weights "<<w_mean;
    report<<" variance of weights "<<w_variance<<"\n";
    report<<"mean bias "<<b_mean;
    report<<" variance of bias "<<b_variance<<"\n";
    report<<"number of weights "<<w_count<<"\n";
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
    std::vector<float> expout(output.size());
    for (int i = 0; i < output.size(); i++)
    {
        expout[i] = std::exp(output[i]);
        if (broken_float(expout[i]))
        {
            expout[i] = max_val;
        }
        denominator += expout[i]; 
    }
    denominator = 1 / denominator; 
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = expout[i] * denominator;
    }
}

void dl_softmax(std::vector<float> &output, std::vector<float> &target, std::vector<float> &dl){
    dl[0] = output[0] - target[0];
    dl[1] = output[1] - target[1];
    dl[2] = output[2] - target[2]; 
}

int wow(std::vector<float> &output, std::vector<float> &target){
    int guess;
    int answer;
    guess = std::distance(output.begin(),std::max_element(output.begin(),output.end()));
    answer = std::distance(target.begin(),std::max_element(target.begin(),target.end()));
    if (guess == answer){return 1;}
    else{return 0;}
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

float loss_iteration(int textindex, NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    hopeless.neural_net_clear();
    int len = data_points[textindex].length();
    float loss;
    if (len < 3)
    {
        target[0] = 0;
        target[1] = 0;
        target[2] = 1;
    }
    else if (labels[textindex] == "Positive")
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
    for (int i = 0; i < data_points[textindex].length(); i+=4)
    {
        std::fill(inputs.begin(),inputs.end(),0);       
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 4) < len) ? (i + 4):len); j++)
        {
            inputs[input_convert(data_points[textindex][j]) + index] = 1;
            index += 95;
        }
        hopeless.forward_pass(stoopid,inputs,a);
    }
    for(int i = 0; i < 20; i++){
        std::fill(inputs.begin(),inputs.end(),0); 
        hopeless.forward_pass(stoopid,inputs,a); 
    }
    for (int i = 0; i < stoopid.output_index.size(); i++)
    {
        output[i] = hopeless.neuron_states[stoopid.output_index[i]];
    }
    soft_max(output);
    loss = soft_max_loss(target,output); 
    return loss;   
}

int acc_iteration(int textindex, NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    hopeless.neural_net_clear();
    int len = data_points[textindex].length();
    if (len < 3)
    {
        target[0] = 0;
        target[1] = 0;
        target[2] = 1;
    }
    else if (labels[textindex] == "Positive")
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
    for (int i = 0; i < data_points[textindex].length(); i+=4)
    {
        std::fill(inputs.begin(),inputs.end(),0);       
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 4) < len) ? (i + 4):len); j++)
        {
            inputs[input_convert(data_points[textindex][j]) + index] = 1;
            index += 95;
        }
        hopeless.forward_pass(stoopid,inputs,a);
    }

    //thinking 
    for( int i = 0 ; i <20;i++){
        std::fill(inputs.begin(),inputs.end(),0); 
        hopeless.forward_pass(stoopid,inputs,a);
    }

    for (int i = 0; i < stoopid.output_index.size(); i++)
    {
        output[i] = hopeless.neuron_states[stoopid.output_index[i]];
    }
    soft_max(output);
    int correct = wow(target,output);
    return correct;   
}

//2 basic typos that wouldn't render a message unreadable, hopefully pads out the training data
std::string typo_noise(std::string words){
    if (words.length()<5)
    {
        return words;
    }
    std::uniform_real_distribution<float> ran(0,10);
    if(ran(mtwister) <= 3){
        float ccc = ran(mtwister);
        if (ccc < 3)            //swap 2 random characters
        {
            std::uniform_int_distribution<int> randchar(1,words.length() - 2);
            int c = randchar(mtwister);
            char ph = words[c];
            if (rand() % 2)
            {
                words[c] = words[c+1];
                words[c+1] = ph;
            }
            else{
                words[c] = words[c-1];
                words[c-1] = ph;
            }
        }
        else if (ccc < 6)       //insert space
        {
            std::uniform_int_distribution<int> randchar(1,words.length() - 2);
            int c = randchar(mtwister);
            words.insert(c," ");
        }
        else                    //remove character
        {
            std::uniform_int_distribution<int> randchar(1,words.length() - 2);
            int c = randchar(mtwister);
            words.erase(c,1);
        }
    }
    return words;
}

void tr_iteration(int textindex, NNclone &hopeless, float learning_rate,std::vector<float> &inputs,std::vector<std::vector<float>> &target,std::vector<std::vector<float>> &output,std::vector<std::vector<float>> &dl,std::vector<std::vector<float>> &forwardpass_states,std::vector<std::vector<float>> &forwardpass_pa, std::vector<std::vector<float>> &forwardpass_fx_tr, float noise = 0.001){
    hopeless.neural_net_clear();
    hopeless.act_func_derivatives_clear();
    int len;
    
    std::string tweet = typo_noise(data_points[textindex]);
    
    if ((tweet.length()  % 4) == 0)
    {
        len = std::floor(tweet.length() / 4);
    }
    else{
        len = std::floor(tweet.length() / 4) + 1;
    }
    len = (len>0) ? len:1;
    len += 20;                      //after reading the string 16 timesteps to "think" about the output
    int c_size = target.size();
    c_size = (len>c_size) ? len:c_size;
    target.resize(c_size);
    output.resize(c_size);
    dl.resize(c_size);
    forwardpass_states.resize(c_size);
    forwardpass_pa.resize(c_size);
    forwardpass_fx_tr.resize(c_size);
    if (tweet.length() < 3)
    {
        for (int i = 0; i < len; i++)
        {
            target[i].resize(3);
            target[i][0] = 0;
            target[i][1] = 0;
            target[i][2] = 1; 
        }
    }
    else if (labels[textindex] == "Positive")
    {
        for (int i = 0; i < len; i++)
        {
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
        output[i].resize(3);
        output[i][0] = 0;
        output[i][1] = 0;
        output[i][2] = 0;
    }
    for (int i = 0; i < len; i++)
    {
        dl[i].resize(3);
        dl[i][0] = 0;
        dl[i][1] = 0;
        dl[i][2] = 0;
    }
    for (int i = 0; i < len; i++)
    {
        forwardpass_fx_tr[i].resize(stoopid.neural_net.size());
        forwardpass_states[i].resize(stoopid.neural_net.size());
        forwardpass_pa[i].resize(stoopid.neural_net.size());
        std::fill(forwardpass_states[i].begin(),forwardpass_states[i].end(),0);
        std::fill(forwardpass_pa[i].begin(),forwardpass_pa[i].end(),0);
        std::fill(forwardpass_fx_tr[i].begin(),forwardpass_fx_tr[i].end(),0);
    }
    int fpasscount = 0;
    std::normal_distribution<float> jitter(0,0.001);
    for (int i = 0; i < tweet.length(); i+=4)
    {
        for (int j = 0; j < inputs.size(); j++)
        {
            inputs[j] = jitter(mtwister);
        }
        //defaults to space if there is no character/ isn't ascii
        //just tiling the loop, we read 10 characters at a time
        int index = 0;
        for (int j = i; j < (((i + 4) < tweet.length()) ? (i + 4):tweet.length()); j++)
        {
            inputs[input_convert(tweet[j]) + index] += 1;
            index += 95;
        }
        hopeless.forward_pass_s_pa(stoopid, inputs,a);
        for (int j = 0; j < stoopid.neural_net.size(); j++)
        {
            forwardpass_states[fpasscount][j] = hopeless.neuron_states[j];
            forwardpass_pa[fpasscount][j] = hopeless.pre_activations[j];
            forwardpass_fx_tr[fpasscount][j] = hopeless.f_x[j];
        }
        for (int j = 0; j < stoopid.output_index.size(); j++)
        {
            output[fpasscount][j] = hopeless.neuron_states[stoopid.output_index[j]];
        }     
        fpasscount++;   
    }
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < inputs.size(); j++)
        {
            inputs[j] = jitter(mtwister);
        }
        hopeless.forward_pass_s_pa(stoopid, inputs,a);

        for (int j = 0; j < stoopid.neural_net.size(); j++)
        {
            forwardpass_states[fpasscount][j] = hopeless.neuron_states[j];
            forwardpass_pa[fpasscount][j] = hopeless.pre_activations[j];
            forwardpass_fx_tr[fpasscount][j] = hopeless.f_x[j];
        }
        for (int j = 0; j < stoopid.output_index.size(); j++)
        {
            output[fpasscount][j] = hopeless.neuron_states[stoopid.output_index[j]];
        }     
        fpasscount++;   
    }
    
    soft_max(output[len -1]);   //only last prediction matters

    dl_softmax(output[len -1], target[len -1], dl[len -1]); 
    for (int j = 0; j < stoopid.output_index.size(); j++)
    {
        forwardpass_states[len -1][stoopid.output_index[j]] = output[len -1][j];
    }
    hopeless.bptt(stoopid,len,forwardpass_states,forwardpass_pa,forwardpass_fx_tr,dl,re_leak,changing_gradient_limit);
}

float avg_loss_tr(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float loss = 0;
    for (int i = 0; i < 40000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(i,hopeless,inputs,target,output);
    }
    return loss/40000;
}

float avg_loss_tr2000(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float loss = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 0; i < 2000; i++)
    {
        int ind = randtrain(mtwister);
        if (data_points[ind].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(ind,hopeless,inputs,target,output);
    }
    return loss/2000;
}

float avg_loss_val_s(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float loss = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 0; i < 2000; i++)
    {
        int ind = randval(mtwister);
        if (data_points[ind].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(ind,hopeless,inputs,target,output);
    }
    return loss/2000;
}

float avg_loss_val(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float loss = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 40000; i < 50000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        loss += loss_iteration(i,hopeless,inputs,target,output);
    }
    return loss/10000;
}

float avg_acc_tr(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float acc = 0;
    for (int i = 0; i < 40000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        acc += acc_iteration(i,hopeless,inputs,target,output);
    }
    acc = acc/40000;
    return acc;
}

float avg_acc_val_s(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float acc = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 0; i < 2000; i++)
    {
        int ind = randval(mtwister);
        if (data_points[ind].size() == 0)
        {
            continue;
        }
        acc += acc_iteration(ind,hopeless,inputs,target,output);
    }
    acc = acc / 2000;
    return acc;
}

float avg_acc_val(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float acc = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 40000; i < 50000; i++)
    {
        if (data_points[i].size() == 0)
        {
            continue;
        }
        acc += acc_iteration(i,hopeless,inputs,target,output);
    }
    acc = acc / 10000;
    return acc;
}

float avg_acc_val_100s(NNclone &hopeless,std::vector<float> &inputs,std::vector<float> &target,std::vector<float> &output){
    float acc = 0;
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for (int i = 0; i < 100; i++)
    {
        int ind = randval(mtwister);
        if (data_points[ind].size() == 0)
        {
            continue;
        }
        acc += acc_iteration(ind,hopeless,inputs,target,output);
    }
    acc = acc / 100;
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

struct s{
    std::vector<float> inputs;
    std::vector<float> target;
    std::vector<float> output;
    std::vector<std::vector<float>> target_tr;
    std::vector<std::vector<float>> output_tr;
    std::vector<std::vector<float>> dl_tr;
    std::vector<std::vector<float>> forwardpass_states_tr;
    std::vector<std::vector<float>> forwardpass_pa_tr;
    std::vector<std::vector<float>> forwardpass_fx_tr;
    s();
};

s::s()
:inputs(380,0)
,target(3,0)
,output(3,0)
{

}

/*
void increase_alpha(){
    for (int i = 0; i < stoopid.neural_net.size(); i++)
    {
        stoopid.neural_net[i].alpha+=0.0001;
    }
}*/

int main(){
    std::fill(tr_data_sample.begin(),tr_data_sample.end(),0);
    std::iota(tr_data_index.begin(),tr_data_index.end(),0);
    load_data(); 
    std::ifstream config ("config.txt");
    std::string config_line;
    float l_r = 0.01;
    int epochs = 10;
    float momentum0 = 0.8;
    int minibatchsize = 8;
    float l_r_decay = 0.1;
    float momentum_decay = 0.1;
    std::vector<NNclone> hopeless(t_hreads,NNclone(stoopid));    //mini batch size of 8   
    std::vector<s> shared_s(t_hreads,s());
    report.open("report.txt",std::fstream::trunc);
    report.close();
    while(true){
        std::getline(config,config_line);
        config_line.erase(config_line.find_last_not_of(" \n\r\f\v\t")+1);
        config_line.erase(0,config_line.find_first_not_of(" \n\r\f\v\t"));
        if (config_line == "learning rate")
        {
            std::getline(config,config_line);
            l_r = std::stof(config_line);
        }
        else if (config_line == "epochs")
        {
            std::getline(config,config_line);
            epochs = std::stoi(config_line);
        }
        else if (config_line == "initial momentum")
        {
            std::getline(config,config_line);
            momentum0 = std::stof(config_line);
        }
        else if (config_line == "mini-batch size")
        {
            std::getline(config,config_line);
            minibatchsize = std::stoi(config_line);
            if (minibatchsize % (hopeless.size()))
            {
                std::cout<<"warning, minibatch size is not a multiple of "<<hopeless.size()<<" will be changed to multiple of 16 less than set size"<<std::endl;
            }
            
        }
        else if(config_line == "learning rate decay parameter"){
            std::getline(config,config_line);
            l_r_decay = std::stof(config_line);
        }
        else if (config_line == "momentum decay parameter")
        {
            std::getline(config,config_line);
            momentum_decay = std::stof(config_line);
        }
        else{
            break;
        }
    }
    float mt = momentum0;
    minibatchsize = minibatchsize - (minibatchsize % hopeless.size());
    int minibatch_m = minibatchsize / hopeless.size();
    float lr0 = l_r;
    float avg_loss_tr_b;
    float avg_loss_val_b;
    float avg_acc_tr_b;
    float avg_acc_val_b;
    float previousL = 6;
    float previouslL = 0.333;
    std::cout<<"Generating Summary"<<std::endl;
    omp_set_num_threads(t_hreads);
    /*
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            //avg_loss_tr_b = avg_loss_tr(hopeless[0],shared_s[0].inputs,shared_s[0].target,shared_s[0].output);
        }
        #pragma omp section
        {
            avg_loss_val_b = avg_loss_val(hopeless[1],shared_s[1].inputs,shared_s[1].target,shared_s[1].output);
        }
        #pragma omp section
        {
            //avg_acc_tr_b = avg_acc_tr(hopeless[2],shared_s[2].inputs,shared_s[2].target,shared_s[2].output);
        }
        #pragma omp section
        {
            avg_acc_val_b = avg_acc_val(hopeless[3],shared_s[3].inputs,shared_s[3].target,shared_s[3].output);
        }
        #pragma omp section
        {
            parameter_check(stoopid);
        }
    }
    //std::cout<<"average loss on training dataset before training "<<avg_loss_tr_b<<std::endl;
    std::cout<<"estimated loss on validation dataset before training "<<avg_loss_val_b<<std::endl; 
    //std::cout<<"average accuracy on training dataset before training "<<avg_acc_tr_b<<std::endl;
    std::cout<<"estimated accuracy on validation dataset before training "<<avg_acc_val_b<<std::endl;
    */
    changing_gradient_limit = 10;
    for (int epc = 0; epc < epochs; epc++)
    {
        shuffle_tr_data();
        std::cout<<std::endl;
        std::cout<<"epoch "<< epc + 1 <<" / "<< epochs;
        std::cout<<": Progress for this epoch...    "<<std::flush;

        for (int t_set_itr = 0; t_set_itr < tr_data_sample.size(); t_set_itr += hopeless.size() * minibatch_m) 
        {
            #pragma omp parallel for schedule(static)
            for (int iiii = 0; iiii < hopeless.size(); iiii++)
            {
                for (int minib = 0; minib < minibatch_m; minib++)
                {
                    int msg = t_set_itr + iiii * minibatch_m + minib;
                    if (msg >= tr_data_sample.size())
                    {
                        break;
                    }
                    msg = tr_data_sample[msg];  
                    if ((data_points[msg].length()) == 0)
                    {
                        continue;
                    }
                    tr_iteration(msg, hopeless[iiii],l_r,shared_s[iiii].inputs,shared_s[iiii].target_tr,shared_s[iiii].output_tr,shared_s[iiii].dl_tr,shared_s[iiii].forwardpass_states_tr,shared_s[iiii].forwardpass_pa_tr,shared_s[iiii].forwardpass_fx_tr); 
                }
            }   
            for (int zz = 0; zz < hopeless.size(); zz++)
            {
                #pragma omp parallel for schedule(dynamic)
                for (int ii = 0; ii < hopeless[zz].weights_g.size(); ii++)
                {
                    #pragma omp simd
                    for (int j = 0; j < hopeless[zz].weights_g[ii].size(); j++)
                    {
                        stoopid.weights_g[ii][j] += hopeless[zz].weights_g[ii][j];
                    }                
                }
                #pragma omp simd
                for (int j = 0; j < hopeless[0].bias_g.size(); j++)
                {
                    stoopid.bias_g[j] += hopeless[zz].bias_g[j];
                }
                #pragma omp simd
                for (int j = 0; j < stoopid.alpha_g.size();j++){
                    stoopid.alpha_g[j] += hopeless[zz].alpha_g[j];
                }
            }

            stoopid.update_momentum(mt);

            stoopid.update_parameters(l_r);

            stoopid.gradient_clear();

            stoopid.l2_reg(l_r * 0.1);

            #pragma omp parallel for
            for (int z = 0; z < hopeless.size(); z++)
            {
                hopeless[z].gradient_clear();
            }


            //keeping the output layer linear, as softmax will be applied
            for (int i = 0; i < stoopid.output_index.size(); i++)
            {
                stoopid.neural_net[stoopid.output_index[i]].alpha = 0;
            }
            //keeping the output layer linear, as softmax will be applied
            for (int i = 0; i < stoopid.input_index.size(); i++)
            {
                stoopid.neural_net[stoopid.input_index[i]].alpha = 0;
            }
            std::cout<<"\b\b\b\b";
            std::cout<<std::setfill(' ')<<std::setw(3)<<std::floor(100 * t_set_itr / tr_data_sample.size() )<< "%";
            std::cout<<std::flush;
            float lL = avg_acc_val_100s(hopeless[0],shared_s[0].inputs,shared_s[0].target,shared_s[0].output);
            if ((lL + 0.1) < previouslL)
            {
                lr0 *= 0.5;
                stoopid.momentum_clear();
            }
            if (lL>previouslL)
            {
                previouslL = lL;
            }
        }

        std::cout<<"\r                                                                                                      "<<std::flush;              //ensure that line is cleared out
        std::cout<<"\r"<<std::flush;
        l_r = lr0 /(1 + l_r_decay*epc);   //decay
        mt = momentum0/(1 + momentum_decay*epc);
        stoopid.save("temp.txt");
        if ((epc % 5 == 4 ) || (epc == 0))
        {
            report.open("report.txt",std::fstream::app);
            std::cout<<"Summary of progress"<<std::endl;
            report<<"Summary of progress\n";
            report<<"---------------------------------------------------------\n";
            parameter_check(stoopid);
            float LT = 0;
            float L = 0;
            float A = 0;
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    L = avg_loss_val_s(hopeless[0],shared_s[0].inputs,shared_s[0].target,shared_s[0].output);
                }
                #pragma omp section
                {
                    A = avg_acc_val_s(hopeless[2],shared_s[2].inputs,shared_s[2].target,shared_s[2].output);
                }
                #pragma omp section
                {
                    LT = avg_loss_tr2000(hopeless[1],shared_s[1].inputs,shared_s[1].target,shared_s[1].output);
                }
            }
            std::cout<<"validation_loss"<<L<<std::endl;
            std::cout<<"estimated validation accuracy "<<A<<std::endl;
            std::cout<<"saved to epoch "<<epc + 1<<" loss "<< L<<std::endl;
            std::cout<<"sample training loss "<<LT<<std::endl;
            if ((L - 0.05) > previousL)
            {
                lr0 *= 0.5;
                stoopid.momentum_clear();
            }
            if (L<previousL)
            {
                previousL = L;
            }
            report<<" validation_loss"<<L<<std::endl;
            report<<"estimated validation accuracy "<<A<<std::endl;
            report<<"saved to epoch "<<epc + 1<<" loss "<< L<<std::endl;
            report<<"sample training loss "<<LT<<std::endl;

            std::string f = "epoch " + std::to_string(epc + 1) + " loss " + std::to_string(L) + ".txt";
            
            report<<"---------------------------------------------------------\n"<<"\n";
            stoopid.save(f);
            report.close();
            //stoopid.l1_reg(l_r * 0.02);
        }
    }
    std::cout<<"\n";
    std::cout<<"training complete"<<"\n";
    std::cout<<"Generating Summary"<<std::endl;  
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            avg_loss_tr_b = avg_loss_tr(hopeless[0],shared_s[0].inputs,shared_s[0].target,shared_s[0].output);
        }
        #pragma omp section
        {
            avg_loss_val_b = avg_loss_val(hopeless[1],shared_s[1].inputs,shared_s[1].target,shared_s[1].output);
        }
        #pragma omp section
        {
            avg_acc_tr_b = avg_acc_tr(hopeless[2],shared_s[2].inputs,shared_s[2].target,shared_s[2].output);
        }
        #pragma omp section
        {
            avg_acc_val_b = avg_acc_val(hopeless[3],shared_s[3].inputs,shared_s[3].target,shared_s[3].output);
        }
        #pragma omp section
        {
            parameter_check(stoopid);
        }
    }
    std::cout<<"average loss on training dataset "<<avg_loss_tr_b<<std::endl;
    std::cout<<"average loss on validation dataset "<<avg_loss_val_b<<std::endl; 
    std::cout<<"average accuracy on training dataset "<<avg_acc_tr_b<<std::endl;
    std::cout<<"average accuracy on validation dataset "<<avg_acc_val_b<<std::endl;   
    std::cout<<"\n";  
    std::cout<<std::flush;
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
            stoopid.save(save_filename);
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