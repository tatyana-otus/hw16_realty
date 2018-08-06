#include "classificator.h"


int main(int argc, char** argv)
{

    try {    
        if (argc !=  2){
            std::cerr << "Usage: rclss modelfname" << std::endl;
            return 1;
        }

        classificator clss(argv[1]);

        for(std::string line; std::getline(std::cin, line);){ 
            
            std::vector<std::string> data;
            boost::split(data, line, boost::is_any_of(";"));
                       
            sample_type coord;
            try {
                if(data.size() != N) throw std::invalid_argument("");

                for(int i = 0; i < N ; ++i) {
                    if(data[i] != "")
                        coord(i) = std::stod(data[i]);
                    else
                        coord(i) = 0;
                }
            }
            catch(const std::exception &e) {
                std::cerr << "Invalid input" << std::endl;
                continue;
            }

            auto num = clss.make_design(coord);
            clss.get_cluster_data(num, coord);

            std::cout << std::endl;
            std::cout << "predicted label: " << num << std::endl;
            std::cout << "------------------------------------------";
            std::cout << std::endl << std::endl;      
        }
    }
    catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0; 
}