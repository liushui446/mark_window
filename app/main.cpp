#include "qtgui/gui.hpp"
#include "core/core.hpp"
int main(int argc, char *argv[])
{
    //sublib1 hi;
    //hi.print();
    sm::Core::get_init()->timeout = 100;
    double x =  sm::Core::get_init()->timeout;
    Point_f kk = { 199,200 };
    sm::Core::get_init()->temp.push_back(kk);
    sm::run_gui(argc, argv);
    return 0;
}
