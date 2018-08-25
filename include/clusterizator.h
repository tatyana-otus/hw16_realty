#include "const.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <dlib/svm_threaded.h>

struct clusterizator 
{

    clusterizator(std::string fn_base_, long clusters_num_):
    fn_base(fn_base_), clusters_num(clusters_num_){}


    void get_data()
    {
        for(std::string line; std::getline(std::cin, line);){ 
            
            std::vector<std::string> data;
            sample_type m;
            boost::split(data, line, boost::is_any_of(";"));
            if(data.size() != 8) throw std::invalid_argument("Invalid input: " + line);
           
            try {
                auto empty =  std::count_if(data.begin(), data.end(), [](const std::string& i){ return i == ""; });
                if(!empty) {
                    for(int i = 0; i < N ; ++i) { 
                        raw_coord[i].push_back(std::stod(data[i]));  
                    }
                    auto val = std::stod(data[N]);
                    if(raw_coord[N - 1].back() == 1 || raw_coord[N - 1].back() == val) 
                        raw_coord[N - 1].back() = 0;
                    else
                        raw_coord[N - 1].back() = 1;    
                }  

            }
            catch(const std::exception &e) {
                throw std::invalid_argument("Invalid input: " + line);
            }        
        }
    }


    void clustering()
    {
        raw_to_norm();
        norm_to_sample();

        get_labels();

        save_raw_coord();
        save_norm_coef();
        save_df();
    }


private:

    void raw_to_norm()
    {
        assert(N >= 2);

        std::array<double, 2> dists;
        for(int i = 0; i < 2; ++i){
            norm_coord[i].reserve(raw_coord[i].size());

            auto min_max = std::minmax_element(raw_coord[i].begin(), raw_coord[i].end());

            dists[i] = *min_max.second - *min_max.first;
            norm_coef[i][0] = *min_max.first;
        }

        auto dist = std::max(dists[0], dists[1]);
        for(int i = 0; i < 2; ++i){
            std::transform(raw_coord[i].begin(), raw_coord[i].end(), back_inserter(norm_coord[i]),
                       [dist](double val) { return val / dist; });
            norm_coef[i][1] = dist;
        }


        for(int i = 2; i < N; ++i){
            norm_coord[i].reserve(raw_coord[i].size());

            auto min_max = std::minmax_element(raw_coord[i].begin(), raw_coord[i].end());

            auto min = *min_max.first;
            auto max = *min_max.second;

            std::transform(raw_coord[i].begin(), raw_coord[i].end(), back_inserter(norm_coord[i]),
                        [max](double val) { return val / max; });
            norm_coef[i][0] = min;
            norm_coef[i][1] = max;
        }
    }


    void norm_to_sample()
    {

        for(size_t i = 0; i < norm_coord[0].size(); ++i){
            sample_type m;
            for(int j = 0; j < N; ++j){
                m(j) = norm_coord[j][i];
            }
            samples.push_back(std::move(m));
        }
    }


    void save_norm_coord() const
    {
        std::vector<std::ofstream> fs;
        for(int i = 0; i < clusters_num; ++i){
            std::string fn = fn_base + "_norm" + "." + std::to_string(i);
            std::ofstream of(fn);
            fs.push_back(std::move(of));
        }

        for(size_t i = 0; i < samples.size(); ++i){
            auto f_idx = labels[i];
            fs[f_idx] << samples[i](0);
            for(auto j = 1; j < N; ++j)
                fs[f_idx] << ";" << samples[i](j);
            fs[f_idx] << "\n";
        }  
    }


    void save_raw_coord() const
    {
        std::vector<std::ofstream> fs;
        for(int i = 0; i < clusters_num; ++i){
            std::string fn = fn_base + "." + std::to_string(i);
            std::ofstream of(fn);
            fs.push_back(std::move(of));
        }


        for(size_t i = 0; i < samples.size(); ++i){
            auto f_idx = labels[i];
            fs[f_idx] << std::fixed << std::setprecision(precision[0]);
            fs[f_idx] << raw_coord[0][i] << ";" << raw_coord[1][i];
            for(auto j = 2; j < N; ++j){
                fs[f_idx] << std::fixed << std::setprecision(precision[j]);
                fs[f_idx] << ";" << raw_coord[j][i];
            }    
            fs[f_idx] << "\n";
        }  
    }


    void save_norm_coef() const
    {
        std::string fn = fn_base + ".coef";
        std::ofstream of(fn);

        of << N << "\n";
        for(int i = 0; i < N; ++i){
            of << norm_coef[i][0] << " " << norm_coef[i][1] << "\n";
        }
    }


    void get_labels()
    {
        typedef radial_basis_kernel<sample_type> kernel_type;

        kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

        kkmeans<kernel_type> test(kc);
        
        std::vector<sample_type> initial_centers;

        test.set_number_of_centers(clusters_num);
        pick_initial_centers(clusters_num, initial_centers, samples, test.get_kernel());
      
        test.train(samples, initial_centers);

        for(auto const& s: samples){
            labels.push_back(test(s));
        }
    }


    void save_df() 
    {
        ovo_trainer trainer;

        krr_trainer<rbf_kernel> rbf_trainer;
        svm_nu_trainer<poly_kernel> poly_trainer;

        poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
        rbf_trainer.set_kernel(rbf_kernel(0.1));

        trainer.set_trainer(rbf_trainer);
        trainer.set_trainer(poly_trainer, 1, 2);

        //----------
        //randomize_samples(samples, labels);
        //std::cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << std::endl;       
        //----------

        one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);
       
        one_vs_one_decision_function<ovo_trainer, 
        decision_function<poly_kernel>,  // This is the output of the poly_trainer
        decision_function<rbf_kernel>    // This is the output of the rbf_trainer
        > df2;

        df2 = df;
        serialize(fn_base) << df2;
    }

    std::array<std::vector<double>, N> raw_coord;
    std::array<std::vector<double>, N> norm_coord;

    std::array<std::array<double, 2>, N> norm_coef;

    std::vector<sample_type> samples;
    std::vector<double> labels;	

    std::string fn_base;
    long clusters_num;   
};