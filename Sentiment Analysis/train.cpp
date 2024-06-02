// setup 380(10 characters at a time) input units and 5(3 t decide sentiment, 2 to decide to stop) output units
#include<vector>
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
#include<omp.h>
#include<array>
#include"version3.hpp"
#include<sstream>
#include<deque>
#include<cstring>
#include <map>
///#include<windows.h>


NN hopeless("model.txt");
std::vector<std::string> characters = {" ","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","z","{","|","}","~"};
std::vector<std::string> inputdata(60000);
std::vector<std::string> labels(60000);
std::vector<int> datapoints(60000);


std::uniform_int_distribution<int> randtrain(0,39999);   //the first 40thousand data points will be training data
std::uniform_int_distribution<int> randval(40000,50000); //this is the validation data

std::vector<std::vector<std::array<float,383>>> encoded_input(60000);
std::vector<std::vector<float>> encoded_target(60000,{0,0,0});
std::array<float,383> blank_input = {0};
std::array<float,383> typo_holder = {0};
const int thinktime=5;

float avg_entrophy = 0;

int threads = 12;
int epochs = 100;
int batch_size = 64;
float mal_r;
float mil_r;
int period;


// typos to create synthetic data for training
float mistake_rate = 0.99;
int min_length = 7;             /*the minimum length for typos to be added*/

float rand_string_min_length = 1; 
float rand_string_max_length = 5;

float remove_character = 0.5;
float extra_space = 0.5;
float swap_adjacent = 0.5;
float replace_letter = 0.5;
std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
std::string symbols = " !\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";
std:: string numbers = "0123456789";
std::string capital_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
float capitalise_rate = 0.5;

void template_config_create(){
    std::ofstream config("config.txt",std::fstream::trunc);
    config << "no_threads: 16\n"<<
    "epochs: 1000\n"<<
    "batch_size: 64\n"<<
    "max_lr: 0.01\n"<<
    "min_lr: 0.000001\n"<<
    "period: 42\n"<<
    "\n"<<
    "\n"<<
    "\n"<<
    "-------------------------------------\n"<<
    "NOTE LEAVE SPACE BETWEEN COLON \":\" AND THE VALUE OF THE HYPERPARAMETER!!!!";
    config.close();
}

void read_config(){
    std::string hparam;
    std::ifstream config("config.txt");
    if (config.good()){;}else{template_config_create();std::cout<<"ERROR "<<"config.txt"<<" does not exist, templace created"<<std::endl; std::exit(EXIT_FAILURE);}
    config >> hparam;
    config >> hparam;
    threads = std::stoi(hparam);
    config >> hparam;
    config >> hparam;
    epochs = std::stoi(hparam);
    config >> hparam;
    config >> hparam;
    batch_size = std::stoi(hparam);   
    config >> hparam;
    config >> hparam;
    mal_r = std::stof(hparam);   
    config >> hparam;
    config >> hparam;
    mil_r = std::stof(hparam); 
    config >> hparam;
    config >> hparam;
    period = std::stoi(hparam); 
    config.close();
}

void load_data(){
    std::ifstream data("text.txt");
    std::ifstream label("label.txt");
    for (int i = 0; i < 60000; i++)
    {
        std::getline(data,inputdata[i]);
        inputdata[i].erase(inputdata[i].find_last_not_of("\r\n") + 1);
    }
    data.close();
    for (int i = 0; i < 60000; i++)
    {
        std::getline(label,labels[i]);
        labels[i].erase(labels[i].find_last_not_of("\r\n") + 1);
    }
    label.close();
}

uint8_t input_convert(std::string s){
    for (uint8_t i = 0; i < 95; i++)
    {
        if (s == characters[i])
        {
            return i;
        }    
    }
    return 0;   //default to whitespace if it isn't printable ascii    
}

void encode(){
    std::string pos = "Positive";
    std::string neg = "Negative";
    std::string neu = "Neutral";
    for (int i = 0; i < 60000; i++)
    {
        if (labels[i].compare(pos)==0)
        {
            encoded_target[i][0] = 1;
            encoded_target[i][1] = 0;
            encoded_target[i][2] = 0;
        }
        else if (labels[i].compare(neg)==0)
        {
            encoded_target[i][0] = 0;
            encoded_target[i][1] = 1;
            encoded_target[i][2] = 0;
        }
        else if (labels[i].compare(neu)==0)
        {
            encoded_target[i][0] = 0;
            encoded_target[i][1] = 0;
            encoded_target[i][2] = 1;
        }
        else{
            std::cout<<labels[i]<<std::flush;
            std::cout<<"ERROR!!!!!!, label"<<std::endl;
            std::cout<<labels[i].compare(pos)<<" "<<labels[i].length()<<std::endl;
            std::exit(EXIT_FAILURE);        
        }    
    }
    for (int i = 0; i < 60000; i++)
    {
        int len = 0;
        if ((inputdata[i].length()  % 4) == 0)
        {
            len = std::floor(inputdata[i].length() / 4);
        }
        else{
            len = std::floor(inputdata[i].length() / 4) + 1;
        }
        encoded_input[i].resize(len);
        #pragma omp parallel for
        for (int j = 0; j < len; j++)
        {
            #pragma omp simd
            for (int k = 0; k < 383; k++)
            {
                encoded_input[i][j][k] = 0;
            }
        }
        int msgind = 0;
        #pragma omp parallel for
        for (int j = 0; j < inputdata[i].length(); j+=4)
        {
            //just tiling the loop, we read 4 characters at a time
            for (int k = j; k < (((j + 4) < inputdata[i].length()) ? (j + 4):inputdata[i].length()); k++)
            {
                std::string chra = "";
                chra += inputdata[i][k];
                encoded_input[i][msgind][input_convert(chra) + ((k%4)*95)] = 1;
            }
            msgind ++;
        }
    }
}


std::string rand_string_generator(std::mt19937 & wwwwww){
    std::string randstring("");
    std::uniform_int_distribution<int> a(rand_string_min_length,rand_string_max_length);
    std::uniform_real_distribution<float> b(0,1);
    int length = a(wwwwww);
    float rand_string_type = b(wwwwww);
    if(rand_string_type > 0.5){            // random letter string
        std::uniform_int_distribution<int> rand_letter(0,51);
        for(int i = 0; i < length; i++){
            int letter = rand_letter(wwwwww);
            if(letter >= 26){
                randstring += capital_alphabet[letter - 26];
            }
            else{
                randstring += alphabet[letter];
            }
        }
    }
    else if (rand_string_type > 0.2){      // random symbol string
        std::uniform_int_distribution<int> rand_sym(0,symbols.length()-1);
        for(int i = 0 ; i < length ; i++){
            int rsymbol = rand_sym(wwwwww);
            randstring+=symbols[rsymbol];
        }
    }
    else{                   // random number string
        std::uniform_int_distribution<int> randno(0,numbers.length()-1);
        for(int i = 0; i <length; i++){
            int rno = randno(wwwwww);
            randstring+= numbers[rno];
        }
    }
    return randstring;
}

void add_typos(){
    #pragma omp parallel
    {
        std::random_device wedewojdoiewdjiw;                          
        std::mt19937 wwwwww(wedewojdoiewdjiw());
        #pragma omp for
        for(int i = 0 ; i < 40000; i++){
            std::uniform_real_distribution<float> rng(0,1);
            float randno = rng(wwwwww);
            std::string message = inputdata[i];
            if((randno < mistake_rate)){
                if(inputdata[i].length() > min_length){
                    randno = rng(wwwwww);
                    if(randno < remove_character){
                        std::uniform_int_distribution<int> rand_char(0,message.length()-1);
                        int pos = rand_char(wwwwww);
                        message.erase(message.begin()+pos);
                    }
                    randno = rng(wwwwww);
                    if(randno < extra_space){
                        std::uniform_int_distribution<int> rand_char(0,message.length()-1);
                        unsigned int pos = rand_char(wwwwww);
                        //std::string space = " ";
                        message.insert(pos, " ");
                    }
                    randno = rng(wwwwww);
                    if(randno < swap_adjacent){
                        std::uniform_int_distribution<int> rand_char(0,message.length()-2);
                        int swappos = rand_char(wwwwww);
                        std::swap(message[swappos],message[swappos+1]);
                    }
                    randno = rng(wwwwww);
                    if(randno < replace_letter){
                        std::uniform_int_distribution<int> rand_char(0,message.length()-1);
                        int pos = rand_char(wwwwww);
                        std::uniform_int_distribution<int> rand_letter(0,alphabet.length()-1);
                        int letter_ind = rand_letter(wwwwww);
                        message[pos] = alphabet[letter_ind];
                    }
                    randno = rng(wwwwww);
                    if(randno < capitalise_rate){
                        std::uniform_int_distribution<int> rand_char(0,message.length()-1);
                        int pos = rand_char(wwwwww);
                        for(int k = 0; k < alphabet.length();k++){
                            if(message[k] == alphabet[k]){
                                message[k] = capital_alphabet[k];
                                break;
                            }
                        }
                    }
                }
                std::uniform_int_distribution<int> rand_char(0,message.length()-1);
                int pos = rand_char(wwwwww);
                message.insert(pos, rand_string_generator(wwwwww));
                int len = 0;
                if ((message.length()  % 4) == 0)
                {
                    len = std::floor(message.length() / 4);
                }
                else{
                    len = std::floor(message.length() / 4) + 1;
                }
                encoded_input[i].resize(len);
                for (int j = 0; j < len; j++)
                {
                    #pragma omp simd
                    for (int k = 0; k < 383; k++)
                    {
                        encoded_input[i][j][k] = 0;
                    }
                }
                int msgind = 0;
                for (int j = 0; j < message.length(); j+=4)
                {
                    //just tiling the loop, we read 4 characters at a time
                    for (int k = j; k < (((j + 4) < message.length()) ? (j + 4):message.length()); k++)
                    {
                        std::string chra = "";
                        chra += message[k];
                        encoded_input[i][msgind][input_convert(chra) + ((k%4)*95)] = 1;
                    }
                    msgind ++;
                }
            }
            else{
                int len = 0;
                if ((inputdata[i].length()  % 4) == 0)
                {
                    len = std::floor(inputdata[i].length() / 4);
                }
                else{
                    len = std::floor(inputdata[i].length() / 4) + 1;
                }
                encoded_input[i].resize(len);
                for (int j = 0; j < len; j++)
                {
                    #pragma omp simd
                    for (int k = 0; k < 383; k++)
                    {
                        encoded_input[i][j][k] = 0;
                    }
                }
                int msgind = 0;
                for (int j = 0; j < inputdata[i].length(); j+=4)
                {
                    //just tiling the loop, we read 4 characters at a time
                    for (int k = j; k < (((j + 4) < inputdata[i].length()) ? (j + 4):inputdata[i].length()); k++)
                    {
                        std::string chra = "";
                        chra += inputdata[i][k];
                        encoded_input[i][msgind][input_convert(chra) + ((k%4)*95)] = 1;
                    }
                    msgind ++;
                }
            }
        }
    }
}

void shuffle_tr_data(){
    std::uniform_int_distribution<int> rint(0,39998);
    std::random_device rd;                          
    std::mt19937 mtwister(rd());
    for(int i = 0; i < 40000; i++){
        int swappos = rint(mtwister);
        swappos = (swappos >= i) ? swappos+1:swappos;
        std::swap(datapoints[i],datapoints[swappos]);
    }
}

int_float acc_iteration(int textindex, NN::training_essentials &helper){
    int_float acc_loss=int_float(0,0.0f);
    helper.resize(encoded_input[textindex].size()+thinktime);   
    int fpasscount = 0;
    std::vector<float> output(3,0);
    for(int j = 0; j < 3; j++){
        output[j] = 0;
    }
    soft_max(output);
    for (int i = 0; i < encoded_input[textindex].size(); i++)
    {
        for(int j = 0; j < 3; j++){
            encoded_input[textindex][i][380+j]=output[j];
        }
        hopeless.forward_pass(encoded_input[textindex][i],helper.post[fpasscount],helper.states,fpasscount);
        for(int j = 0; j < 3; j++){
            output[j] = hopeless.neural_net[hopeless.output_index[j]].units[15];
        }
        soft_max(output);
        fpasscount++;   
    }
    while(true)
    {
        for(int j = 0; j < 3; j++){
            blank_input[380+j]=output[j];
        }
        hopeless.forward_pass(blank_input,helper.post[fpasscount],helper.states,fpasscount);
        for(int j = 0; j < 3; j++){
            output[j] = hopeless.neural_net[hopeless.output_index[j]].units[15];
        }
        soft_max(output);
        if ((fpasscount==(helper.dloss.vec_size-1)))
        {
            if (encoded_target[textindex][prediction(output)]==1)
            {
                acc_loss.x=1;
            }   
            acc_loss.y = cross_entrophy_loss(encoded_target[textindex],output);
            break;
        }
        fpasscount++;
    }
    return acc_loss;
}

void sample_acc(NN::training_essentials &helper){
    float acc = 0;
    float lo = 0;
    for (int i = 40000; i < 50000; i++)
    {
        if (inputdata[i].size() == 0)
        {
            continue;
        }
        int_float ret = acc_iteration(i,helper);
        acc += ret.x;
        lo += ret.y;
    }
    acc *= 0.0001;
    lo *= 0.0001;
    std::cout<<"accuracy on validation set is:"<<acc<<", average loss is:"<<lo<<std::endl;
}

float tr_iteration(int textindex, NN::training_essentials &helper){
    int len;
    float loss; 
    helper.resize(encoded_input[textindex].size()+thinktime+1);   
    int fpasscount = 0;
    std::vector<float> output(3,0);
    for(int j = 0; j < 3; j++){
        output[j] = 0;
    }
    soft_max(output);
        for (int i = 0; i < encoded_input[textindex].size(); i++)
        {
            for(int j = 0; j < 3; j++){
                encoded_input[textindex][i][380+j]=output[j];
            }
            hopeless.forward_pass(encoded_input[textindex][i],helper.post[fpasscount],helper.states,fpasscount);
            for(int j = 0; j < 3; j++){
                output[j] = hopeless.neural_net[hopeless.output_index[j]].units[15];
            }
            soft_max(output);
            fpasscount++;   
        }
        while(true)
        {
            std::vector<float> losso(3);
            for(int j = 0; j < 3; j++){
                blank_input[380+j]=output[j];
            }
            hopeless.forward_pass(blank_input,helper.post[fpasscount],helper.states,fpasscount);
            for(int j = 0; j < 3; j++){
                output[j] = hopeless.neural_net[hopeless.output_index[j]].units[15];
            }
            soft_max(output);
            dsoft_max(output,encoded_target[textindex],losso);
            for (int j = 0; j < 3; j++)
            {
                helper.dloss(fpasscount,j)=(std::abs(hopeless.neural_net[hopeless.output_index[j]].units[15])<40) ? losso[j]:sign_of(hopeless.neural_net[hopeless.output_index[j]].units[15]) * (std::abs(hopeless.neural_net[hopeless.output_index[j]].units[15]) - 40);   
            }
            if (fpasscount==(helper.dloss.vec_size-1))
            {
                    /*
                    std::cout<<std::endl;
                    for(int j = 0; j < 3; j++){
                        std::cout<<encoded_target[textindex][j]<<",";
                    }
                    std::cout<<std::endl;
                    for(int j = 0; j < 3; j++){
                        std::cout<<hopeless.neural_net[hopeless.output_index[j]].units[15]<<",";
                    }
                    std::cout<<std::endl;
                    for(int j = 0; j < 3; j++){
                        std::cout<<output[j]<<",";
                    }
                    std::cout<<std::endl;*/
                loss = cross_entrophy_loss(encoded_target[textindex],output);
                avg_entrophy += pmf_entrophy(output);
                helper.resize(fpasscount+1);
                hopeless.bptt(helper.dloss,helper.post,helper.f,helper.states,helper.gradients);
                break;
            }
            fpasscount++;
        }
        helper.f.global_norm_clip(6969.0f);
        return loss;
}

float cosine_anneal_lr(int epoch, int period, float minlr, float maxlr){
    /*
    if((epoch%period) <= period * 1.0f/3.0f)
    {
        return minlr + (maxlr-minlr)*(epoch%period)/(period * 1.0f/3.0f);
    }
    epoch -= period * 1.0f/3.0f;
    period = period * 2.0f/3.0f;*/
    return (minlr+(0.5*(maxlr - minlr)*(1+std::cos(3.141592653589793f * epoch/period))));
}

int main(){
    for (int i = 0; i < 60000; i++)
    {
        datapoints[i] = i;
    }
    load_data(); 
    encode();
    read_config();
    NN::training_essentials gradientsandmore(hopeless);
    NN::network_gradient past_grad(hopeless);
    NN::network_gradient current_grad(hopeless);
    omp_set_num_threads(threads);
    /*
    std::cout<<"number of parameters in model:"<<hopeless.parameter_count()<<std::endl;
    std::cout<<"number of epochs"<<std::endl;
    std::cin>>epochs;
    std::cout<<"maximum learning rate"<<std::endl;
    std::cin>>mal_r;
    std::cout<<"minimum learning rate"<<std::endl;
    std::cin>>mil_r;
    std::cout<<"cosine annealing period"<<std::endl;
    std::cin>>period;*/
    //sample_acc(gradientsandmore);
    int epc = 0;        
    int t_set_itr = 0;  
    long long count = 0;
    float epochloss = 0;
    hopeless.weight_index_sort();
    std::cout<<"number of parameters in model:"<<hopeless.parameter_count()<<std::endl;
    //std::cout<<"no of neuron_units"<<hopeless.neural_net.size()<<std::endl;
    //hopeless.parameter_count();
    while(epc < epochs)
    {
        epc++;
        float l_r = cosine_anneal_lr(epc,period,mil_r,mal_r);
        shuffle_tr_data();
        //add_typos();
        std::cout<<"Progress for this epoch..."<<std::flush;
        t_set_itr = 0;
        epochloss = 0;
        while(t_set_itr < 40000)
        {     
            for (int i = 0; i < batch_size; i++)
            {
                int msg = t_set_itr + i;
                if (inputdata[datapoints[msg]].size() == 0)
                {
                    continue;
                }
                epochloss+=tr_iteration(datapoints[msg],gradientsandmore);
                current_grad.condense_clear(gradientsandmore.f);
            }
            //current_grad.global_norm_clip(0.25f); ///?????
            float inv_batchsize = 1.0f/batch_size;
            #pragma omp parallel 
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < current_grad.net_grads.size(); i++)
                {
                    #pragma omp simd
                    for (int j = 0; j < 16; j++)
                    {
                        current_grad.net_grads[i].bias[j]*=inv_batchsize;
                    }
                    current_grad.net_grads[i].padding_or_param1*=inv_batchsize;
                    #pragma omp simd collapse(2)
                    for (int j = 0; j < 9; j++)
                    {
                        for (int k = 0; k < 7; k++)
                        {
                            current_grad.net_grads[i].weights[j][k]*=inv_batchsize;
                        }   
                    }
                }
                #pragma omp for schedule(dynamic,16)
                for (int i = 0; i < current_grad.weight_gradients.size(); i++)
                {
                    #pragma omp simd
                    for (int j = 0; j < current_grad.weight_gradients[i].size(); j++)
                    {
                        current_grad.weight_gradients[i][j]*=inv_batchsize;
                    }
                }     
            }
            past_grad.sgd_with_nesterov(hopeless,l_r,0.95,current_grad);
            if((t_set_itr % (batch_size*10))==0){
                std::cout<<"\r                                                                                                          ";
                std::cout<<"\rEpoch "<<epc<<" Progress[";
                float progresspercent =100 * t_set_itr/40000;  
                std::cout<<progresspercent<<"%]"<<", lr:"<<l_r<<", avg loss:"<<epochloss/(t_set_itr+batch_size)<<", avg entrophy:"<<avg_entrophy/(t_set_itr+batch_size);
                std::cout<<std::flush;
            }
            t_set_itr += batch_size;
        }
        avg_entrophy = 0;
        std::cout<<"\r                                                                                                                    "<<std::endl;
        std::cout<<std::endl;
        std::cout<<"epoch "<< epc <<" out of "<< epochs << " complete"<<std::endl;
        std::cout<<std::flush;
        std::cout<<"total cross entrophy loss "<<epochloss<<", learning rate:"<<l_r<<std::endl;
        if ((epc % 10 == 0)&&(epc!=0))
        {
            std::string s = std::to_string(epc);
            hopeless.save_to_txt("checkpoint"+s+".txt");
            //sample_acc(gradientsandmore);
        }
        if ((epc % 5 == 0)&&(epc!=0))
        {
            sample_acc(gradientsandmore);
        }
        std::cout<<std::endl;
        hopeless.save_to_txt("dump.txt");
        //past_grad.valclear();
    }    
    sample_acc(gradientsandmore);
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
            hopeless.save_to_txt(save_filename);
            break;
        }
        else if (yn == 'n')
        {
            break;
        }
        else{
            std::cout<<"ERROR, enter y or n"<<std::endl;
            //Sleep(150); //in case you hold onto a key and this message fills the terminal
        }
    }
    return 0;
}


