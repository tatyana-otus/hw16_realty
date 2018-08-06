#include "clusterizator.h"


int main(int argc, char** argv)
{
    long clusters_num;

    try {    
        if (argc !=  3){
            std::cerr << "Usage: rclst <n> modelfname" << std::endl;
            return 1;
        }
        std::string str_clusters_num = argv[1];
        if(!std::all_of(str_clusters_num.begin(), str_clusters_num.end(), ::isdigit))
            throw std::invalid_argument("Invalid <n>");
        clusters_num = std::stoull(str_clusters_num);
        if(clusters_num == 0)
            throw std::invalid_argument("Invalid <n>");

        clusterizator clst(argv[2], clusters_num);

        clst.get_data();
        clst.clustering();
    }
    catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}