#include "qtgui/ImageTransform.hpp"

/*
QImage::Format枚举参考：https://doc.qt.io/qt-5/qimage.html#Format-enum
Qt的枚举体现了通道顺序和一个像素点的位宽
如：Format_ARGB32一个32位整形存储ARGB值
Format_RGB888相当于三个8位整形存储RGB值
Qt5.14增加了Format_BGR888，和opencv转换更方便了

cv的枚举体现位宽和通道数，而通道顺序是固定的
如：CV_8UC1
U表示存储类型
U无符号整形、S有符号整形、F单精度浮点型
8U就是每个通道由8位的无符号整形表示，unsigned char
C表示通道，多通道时排列顺序为BGR
C1-灰度图片,单通道
C3-BGR图像,3通道
C4-BGRA图像,4通道
BGR转换成RGB调用cv::cvtColor(cv::Mat src, cv::Mat dst, cv::COLOR_BGR2RGB)
*/

QImage cvMat2QImage(const cv::Mat &mat)
{
    QImage image;
    switch (mat.type())
    {
    case CV_8UC1:
        // QImage构造：数据，宽度，高度，每行多少字节，存储结构
        image = QImage((const unsigned char *)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        break;
    case CV_8UC3:
        image = QImage((const unsigned char *)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        image = image.rgbSwapped(); // BRG转为RGB
        // Qt5.14增加了Format_BGR888
        // image = QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.cols * 3, QImage::Format_BGR888);
        break;
    case CV_8UC4:
        image = QImage((const unsigned char *)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        break;
    case CV_16UC4:
        image = QImage((const unsigned char *)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGBA64);
        image = image.rgbSwapped(); // BRG转为RGB
        break;
    }
    return image;
}

cv::Mat QImage2cvMat(const QImage& image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_Grayscale8: // 灰度图，每个像素点1个字节（8位）
        // Mat构造：行数，列数，存储结构，数据，step每行多少字节
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_ARGB32: // uint32存储0xAARRGGBB，pc一般小端存储低位在前，所以字节顺序就成了BGRA
    case QImage::Format_RGB32: // Alpha为FF
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888: // RR,GG,BB字节顺序存储
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGR的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_RGBA64: // uint64存储，顺序和Format_ARGB32相反，RGBA
        mat = cv::Mat(image.height(), image.width(), CV_16UC4, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGRA的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    return mat;
}