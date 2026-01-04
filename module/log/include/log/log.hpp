#ifndef LOG_HPP
#define LOG_HPP
#include"core/std.hpp"
#include <string>
namespace sm {
    SM_EXPORTS class Write_Log {
        Write_Log();
    public:
        SM_EXPORTS static Write_Log* get_init();
        SM_EXPORTS void  Log(std::string infor);
    };
    /*SM_EXPORTS std::string get_gui();

    SM_EXPORTS int run_gui(int argc, char* argv[]);

    SM_EXPORTS int run_gui2(int argc, char* argv[]);*/

}

#endif//SMARTSM_GUI_HPP