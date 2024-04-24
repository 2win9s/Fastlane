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

#include<windows.h>


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
const int thinktime=5;

void load_data(){
    std::ifstream data("text.txt");
    std::ifstream label("label.txt");
    for (int i = 0; i < 60000; i++)
    {
        std::getline(data,inputdata[i]);
    }
    data.close();
    for (int i = 0; i < 60000; i++)
    {
        std::getline(label,labels[i]);
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
    for (int i = 0; i < 60000; i++)
    {
        if (labels[i] == "Positive")
        {
            encoded_target[i][0] = 1;
            encoded_target[i][1] = 0;
            encoded_target[i][2] = 0;
        }
        else if (labels[i] == "Negative")
        {
            encoded_target[i][0] = 0;
            encoded_target[i][1] = 1;
            encoded_target[i][2] = 0;
        }
        else if (labels[i] == "Neutral")
        {
            encoded_target[i][0] = 0;
            encoded_target[i][1] = 0;
            encoded_target[i][2] = 1;
        }
        else{
            std::cout<<"ERROR!!!!!!, label"<<std::endl;
            std::cout<<labels[i]<<std::endl;
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

int acc_iteration(int textindex, NN::training_essentials &helper){
    int acc=0; 
    helper.resize(encoded_input[textindex].size()+thinktime);   
    int fpasscount = 0;
    std::vector<float> output(3,0);
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
        if ((*std::max_element(output.begin(),output.end())>0.95f)||(fpasscount==(helper.dloss.vec_size-1)))
        {
            if (encoded_target[textindex][prediction(output)]==1)
            {
                acc=1;
            }   
            break;
        }
        fpasscount++;
    }
    return acc;
}

void sample_acc(NN::training_essentials &helper){
    float acc = 0;
    for (int i = 40000; i < 50000; i++)
    {
        if (inputdata[i].size() == 0)
        {
            continue;
        }
        acc+=acc_iteration(i,helper);
    }
    acc *= 0.0001;
    std::cout<<"accuracy on validation set is:"<<acc<<std::endl;
}

float tr_iteration(int textindex, NN::training_essentials &helper){
    int len;
    float loss; 
    helper.resize(encoded_input[textindex].size()+thinktime);   
    int fpasscount = 0;
    std::vector<float> output(3,0);
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
        if ((*std::max_element(output.begin(),output.end())>0.95f)||(fpasscount==(helper.dloss.vec_size-1)))
        {
            loss = cross_entrophy_loss(encoded_target[textindex],output);
            dsoft_max(output,encoded_target[textindex],losso);
            for (int j = 0; j < 3; j++)
            {
                helper.dloss(fpasscount,j)=(std::abs(hopeless.neural_net[hopeless.output_index[j]].units[15])<50) ? losso[j]:sign_of(hopeless.neural_net[hopeless.output_index[j]].units[15]) /** (std::abs(hopeless.neural_net[hopeless.output_index[j]].units[15]) - 50)*/;   
            }
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
            helper.resize(fpasscount+1);
            hopeless.bptt(helper.dloss,helper.post,helper.f,helper.states,helper.gradients);
            break;
        }
        fpasscount++;
    }
    helper.f.global_norm_clip(10.0f);
    return loss;
}

float cosine_anneal_lr(int epoch, int period, float minlr, float maxlr){
    if((epoch%period) <= period * 1.0f/3.0f)
    {
        return minlr + (maxlr-minlr)*(epoch%period)/(period * 1.0f/3.0f);
    }
    epoch -= period * 1.0f/3.0f;
    period = period * 2.0f/3.0f;
    return (minlr+(0.5*(maxlr - minlr)*(1+std::cos(3.141592653589793f * epoch/period))));
}

int main(){
    for (int i = 0; i < 60000; i++)
    {
        datapoints[i] = i;
    }
    load_data(); 
    encode();
    u_int batch_size = 64;
    u_int threads = 16;
    NN::training_essentials gradientsandmore(hopeless);
    NN::network_gradient past_grad(hopeless);
    NN::network_gradient current_grad(hopeless);
    omp_set_num_threads(threads);
    int epochs;
    float mal_r;
    float mil_r;
    int period;
    std::cout<<"number of parameters in model:"<<hopeless.parameter_count()<<std::endl;
    std::cout<<"number of epochs"<<std::endl;
    std::cin>>epochs;
    std::cout<<"maximum learning rate"<<std::endl;
    std::cin>>mal_r;
    std::cout<<"minimum learning rate"<<std::endl;
    std::cin>>mil_r;
    std::cout<<"cosine annealing period"<<std::endl;
    std::cin>>period;
    int epc = 0;        
    int t_set_itr = 0;  
    long long count = 0;
    float epochloss = 0;
    hopeless.weight_index_sort();
    while(epc < epochs)
    {
        float l_r = cosine_anneal_lr(epc,period,mil_r,mal_r);
        shuffle_tr_data();
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
            count++;
            past_grad.sgd_with_momentum(hopeless,l_r /* 1.0f/(std::sqrt(batch_size))*/,0.95,current_grad,count);
            if((t_set_itr % (batch_size*10))==0){
                std::cout<<"\r                                                                                            ";
                std::cout<<"\rProgress for this epoch[";
                float progresspercent =100 * t_set_itr/40000;  
                std::cout<<progresspercent<<"%]"<<", learning rate:"<<l_r<<", average loss:"<<epochloss/(t_set_itr+batch_size);
            }
            t_set_itr += batch_size;
        }
        std::cout<<"\r                                                                                                                    "<<std::endl;
        std::cout<<std::endl;
        std::cout<<"epoch "<< epc + 1 <<" out of "<< epochs << " complete"<<std::endl;
        std::cout<<std::flush;
        std::cout<<"total cross entrophy loss "<<epochloss<<", learning rate:"<<l_r<<std::endl;
        if (epc % 10 == 9)
        {
            std::string s = std::to_string(epc+1);
            hopeless.save_to_txt("checkpoint"+s+".txt");
            sample_acc(gradientsandmore);
        }
        epc++;
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
            Sleep(150); //in case you hold onto a key and this message fills the terminal
        }
    }
    return 0;
}


