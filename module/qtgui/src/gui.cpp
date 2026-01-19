#include <iostream>
#include"qtgui/gui.hpp"
#include "mainwindow.h"
#include <QApplication>
#include "core/core.hpp"
namespace sm {


    std::string get_gui()
    {
        return std::string("gui");
    }

    int run_gui(int argc, char* argv[])
    {
        std::cout << "sm run_gui !" << std::endl;
        double y = sm::Core::get_init()->timeout;
        double xx = sm::Core::get_init()->temp.at(0).x;
        double yy = sm::Core::get_init()->temp.at(0).y;
        QApplication a(argc, argv);
        MainWindow w;
        w.show();
        return a.exec();
       
    }
}