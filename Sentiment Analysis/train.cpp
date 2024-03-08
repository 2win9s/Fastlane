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


NN model("model.txt");
std::vector<std::string> characters = {" ","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","z","{","|","}","~"};

std::vector<std::string> inputdata(60000);
std::vector<std::string> labels(60000);
std::vector<int> datapoints(60000);


std::uniform_int_distribution<int> randtrain(0,39999);   //the first 40thousand data points will be training data
std::uniform_int_distribution<int> randval(40000,50000); //this is the validation data

std::vector<std::vector<std::array<float,380>>> encoded_input(60000);
std::vector<std::vector<float>> encoded_target(60000,{0,0,0});
const std::array<float,380> blank_input = {0};
const int thinktime=10;

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
    for (uint8_t i = 0; i < characters.size(); i++)
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
        #pragma omp simd collapse(2)
        for (int j = 0; j < len; j++)
        {
            for (int k = 0; k < 380; k++)
            {
                encoded_input[i][j][k] = 0;
            }
        }
        int msgind = 0;
        for (int j = 0; j < inputdata[i].length(); j+=4)
        {
            //just tiling the loop, we read 4 characters at a time
            #pragma omp simd
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

int acc_iteration(int textindex, NN::npNN &hopeless, NN::training_essentials &helper){
    int acc=0; 
    helper.resize(encoded_input[textindex].size()+thinktime);   
    int fpasscount = 0;
    for (int i = 0; i < encoded_input[textindex].size(); i++)
    {
        hopeless.forward_pass(model,encoded_input[textindex][i],helper.states,fpasscount,helper.pre[fpasscount]);
        fpasscount++;   
    }
    while(true)
    {
        std::vector<float> output(3);
        std::vector<float> losso(3);
        std::vector<float> lossb(2);
        std::vector<float> buzzer(2);
        hopeless.forward_pass(model,blank_input,helper.states,fpasscount,helper.pre[fpasscount]);
        for(int j = 0; j < output.size(); j++){
            output[j] = hopeless.neural_net[model.output_index[j]].units[15];
        }
        for (int j = output.size(); j < 5; j++)
        {
            buzzer[j-3] = hopeless.neural_net[model.output_index[j]].units[15];
        }
        soft_max(output);
        soft_max(buzzer);
        if ((buzzer[0]>0.95f)||(fpasscount==(helper.dloss.vec_size-1)))
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

void sample_acc(NN::npNN &hopeless, NN::training_essentials &helper){
    float acc = 0;
    shuffle_tr_data();
    for (int i = 40000; i < 50000; i++)
    {
        if (inputdata[i].size() == 0)
        {
            continue;
        }
        acc+=acc_iteration(i,hopeless,helper);
    }
    acc *= 0.0001;
    std::cout<<"accuracy on validation set is:"<<acc<<std::endl;
}

float tr_iteration(int textindex, NN::npNN &hopeless, NN::training_essentials &helper){
    int len;
    float loss; 
    helper.resize(encoded_input[textindex].size()+thinktime);   
    int fpasscount = 0;
    for (int i = 0; i < encoded_input[textindex].size(); i++)
    {
        hopeless.forward_pass(model,encoded_input[textindex][i],helper.pre[fpasscount],helper.post[fpasscount],helper.states,fpasscount);
        fpasscount++;   
    }
    #pragma omp simd
    for (int i = 0; i < helper.dloss.vec_size; i++)
    {
        helper.dloss(i,0) = 0;
    }
    while(true)
    {
        std::vector<float> output(3);
        std::vector<float> losso(3);
        std::vector<float> lossb(2);
        std::vector<float> buzzer(2);
        hopeless.forward_pass(model,blank_input,helper.pre[fpasscount],helper.post[fpasscount],helper.states,fpasscount);
        for(int j = 0; j < output.size(); j++){
            output[j] = hopeless.neural_net[model.output_index[j]].units[15];
        }
        for (int j = output.size(); j < 5; j++)
        {
            buzzer[j-3] = hopeless.neural_net[model.output_index[j]].units[15];
        }
        soft_max(output);
        soft_max(buzzer);
        if (*std::max_element(output.begin(),output.end())>0.95f)
        {
            std::vector<float> ring(2,0);
            ring[0] = 1;
            dsoft_max(buzzer,ring,lossb);
            for (int j = 0; j < 2; j++)
            {
                helper.dloss(fpasscount,3+j)=lossb[j] * 0.1;
            }
            dsoft_max(output,encoded_target[textindex],losso);
            for (int j = 0; j < 3; j++)
            {
                helper.dloss(fpasscount,j)=losso[j] * 0.1;
            }
        }
        if ((buzzer[0]>0.95f)||(fpasscount==(helper.dloss.vec_size-1)))
        {
            loss = cross_entrophy_loss(encoded_target[textindex],output);
            dsoft_max(output,encoded_target[textindex],losso);
            for (int j = 0; j < 3; j++)
            {
                helper.dloss(fpasscount,j)=losso[j];
            }
            helper.resize(fpasscount+1);
            hopeless.bptt(model,helper.dloss,helper.pre,helper.post,helper.f,helper.states,helper.gradients);
            break;
        }
        fpasscount++;
    }
    return loss;
}

float cosine_anneal_lr(int epoch, int period, float minlr, float maxlr){
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
    NN::npNN ns(model);
    std::vector<NN::npNN> hopeless(threads,ns);
    NN::training_essentials eh(model);
    std::vector<NN::training_essentials> gradientsandmore(threads,eh);
    NN::network_gradient past_grad(model);
    NN::network_gradient current_grad(model);
    omp_set_num_threads(threads);
    int epochs;
    float mal_r;
    float mil_r;
    int period;
    std::cout<<"number of parameters in model:"<<model.parameter_count()<<std::endl;
    std::cout<<"number of epochs"<<std::endl;
    std::cin>>epochs;
    std::cout<<"maximum learning rate"<<std::endl;
    std::cin>>mal_r;
    std::cout<<"minimum learning rate"<<std::endl;
    std::cin>>mil_r;
    std::cout<<"cosine annealing period"<<std::endl;
    std::cin>>period;
    int epc = 0;        //to replace the for loops with while loops using a shared variable
    int t_set_itr = 0;  //to replace the for loops with while loops using a shared variable
    float epochloss;
    while(epc < epochs)
    {
        float l_r = cosine_anneal_lr(epc,period,mil_r,mal_r);
        shuffle_tr_data();
        std::cout<<"Progress for this epoch..."<<std::flush;
        t_set_itr = 0;
        epochloss = 0;
        while(t_set_itr < 40000)
        {       
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < batch_size; i++)
            {
                int t_num = omp_get_thread_num();
                int msg = t_set_itr + i;
                if (inputdata[datapoints[msg]].size() == 0)
                {
                    continue;
                }
                epochloss+=tr_iteration(datapoints[msg], hopeless[t_num],gradientsandmore[t_num]);
            }
            current_grad.valclear();
            for (int i = 0; i < hopeless.size(); i++)
            {
                current_grad.condense(gradientsandmore[i].f);
            }
            current_grad.norm_clip(0.5);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < hopeless.size(); i++)
            {
                gradientsandmore[i].f.valclear();
            }
            past_grad.sgd_with_momentum(model,l_r,0.9,current_grad);
            if((t_set_itr % (batch_size*10))==0){
                std::cout<<"\r                                                           ";
                std::cout<<"\rProgress for this epoch...";
                float progresspercent =100 * t_set_itr/40000;  
                std::cout<<progresspercent<<"%";
            }
            t_set_itr += batch_size;
        }
        std::cout<<"\r                                                           "<<std::endl;
        std::cout<<std::endl;
        std::cout<<"epoch "<< epc + 1 <<" out of "<< epochs << " complete"<<std::endl;
        std::cout<<std::flush;
        std::cout<<"total cross entrophy loss "<<epochloss<<std::endl;
        if (epc % 10 == 9)
        {
            std::string s = std::to_string(epc+1);
            model.save_to_txt("checkpoint"+s+".txt");
            sample_acc(hopeless[0],gradientsandmore[0]);
        }
        epc++;
        std::cout<<std::endl;
        model.save_to_txt("dump.txt");
        //past_grad.valclear();
    }    
    sample_acc(hopeless[0],gradientsandmore[0]);
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
            model.save_to_txt(save_filename);
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


