#include <iostream>
#include <QString>
#include <QDateTime>
#include"log/log.hpp"
#include <fstream>
std::ofstream output;
sm::Write_Log* m_log;
namespace sm {
    
    Write_Log::Write_Log()
    {
        output.open("MessageDeal.txt", std::ios_base::app);
        if (!output.is_open())
        {
            std::cout << "XXXXXXXXXXXX" << std::endl << "日志文件打开失败" << std::endl;
        }
    }
    Write_Log* Write_Log::get_init()
    {
        if (m_log == NULL)
        {
            m_log = new Write_Log();
        }
        return m_log;
    }
    void Write_Log::Log(std::string infor)
    {
        QDateTime dateTime = QDateTime::currentDateTime();
        // 字符串格式化
        QString timestamp = dateTime.toString("yyyy-MM-dd hh:mm:ss.zzz  ");
        std::cout << timestamp.toStdString().data() << infor <<std::endl;
        output << timestamp.toStdString().data() << infor << std::endl;
    }

    
}