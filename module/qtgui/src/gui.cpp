#include <iostream>
#include"qtgui/gui.hpp"
#include "mainwindow.h"
#include <QApplication>

namespace sm {


    std::string get_gui()
    {
        return std::string("gui");
    }

    int run_gui(int argc, char* argv[])
    {
        std::cout << "sm run_gui !" << std::endl;
        QApplication a(argc, argv);
        MainWindow w;
        w.show();
        return a.exec();
       
    }
}