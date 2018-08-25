#include "const.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <dlib/svm_threaded.h>

struct classificator
{
	classificator(std::string fn_base_):fn_base(fn_base_)
	{
        read_norm_coef();
        deserialize(fn_base) >> df;
	}


    double make_design(const sample_type & coord) const
    {   
        auto test_coord = coord;
        for(int i = 0; i < N; ++i){
            test_coord(i) = test_coord(i) / norm_coef[i][1];
        }

        return df(test_coord); 
    }


    void get_cluster_data(size_t num, const sample_type& coord) 
    {
        std::vector<std::array<double, N>> raw_data;

        std::string fn = fn_base + "." + std::to_string(num);
        std::ifstream ifs(fn);

        if (!ifs.is_open()) {
            throw std::invalid_argument("File could not be opened");
        }

        for(std::string line; std::getline(ifs, line);){ 
                
            std::vector<std::string> data;
            boost::split(data, line, boost::is_any_of(";"));
            if(data.size() != N) throw std::invalid_argument("Invalid input: " + line);
            
            std::array<double, N> coord;
            try {
                std::generate_n(coord.begin(), N, [i = 0, &data] () mutable 
                                                  { return std::stod(data[i++]); });
            }
            catch(const std::exception &e) {
                throw std::invalid_argument("Invalid input: " + line);
            }
            raw_data.push_back(std::move(coord));
        }  

    //----------------------------
        sort_cluster_data(raw_data, coord);
        out_cluster_data(raw_data); 
    }

private:
    void out_cluster_data(const std::vector<std::array<double, N>>& raw_data) const
    {
        for(size_t i = 0; i < raw_data.size(); ++i){
            std::cout << std::fixed << std::setprecision(precision[0]);
            std::cout << raw_data[i][0] << ";" << raw_data[i][1];
            for(auto j = 2; j < N; ++j){
                std::cout << std::fixed << std::setprecision(precision[j]);
                std::cout << ";" << raw_data[i][j];
            }    
            std::cout << "\n";
        } 
    }


    void sort_cluster_data(std::vector<std::array<double, N>>& raw_data, const sample_type& coord) const
    {
        auto x = coord(0);
        auto y = coord(1);

        std::sort(raw_data.begin(), raw_data.end(), [x, y](const auto& a, const auto& b){
        return (std::sqrt(std::pow(x - a[0], 2) + std::pow(y - a[1], 2)) < std::sqrt(std::pow(x - b[0], 2) + std::pow(y - b[1], 2)));
        });
    }


    void read_norm_coef()
    {
        std::string fn = fn_base + ".coef";
        std::ifstream ifs(fn);

        if (!ifs.is_open()) {
            throw std::invalid_argument("File could not be opened");
        }

        int n;
        ifs >> n;
        for(int i = 0; i < N; ++i){
           ifs >> norm_coef[i][0];
           ifs >> norm_coef[i][1];
        }
    }

    std::string fn_base;
    std::array<std::array<double, 2>, N> norm_coef;

    one_vs_one_decision_function<ovo_trainer, 
            decision_function<poly_kernel>,  // This is the output of the poly_trainer
            decision_function<rbf_kernel>    // This is the output of the rbf_trainer
            >  df;

};