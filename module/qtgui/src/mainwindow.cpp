#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "log/log.hpp"
#include <QTextCodec> 
#include <QFileDialog>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>       // 鼠标事件
#include <QWheelEvent>       // 滚轮事件（用于 zoom）
#include <QDebug>            // 输出日志
#include <QPixmap>
#include <cmath>
#include <QFile>
#include <QXmlStreamReader>
#include <QVector>
#include <QPushButton>
#include <fstream>
#include <filesystem>  // C++17的文件系统支持
#include <opencv2/opencv.hpp>
#include <QTime>
#include <algorithm/markInterface.h>
#include <QMessageBox>
#include <AutoFoucs/Foucs.h>
#include <QStringListModel> 
#include "selectablegraphicsview.h"
#include <algorithm/NccMatchDll.h>
#include "core/core.hpp"

#include <stdio.h>
#include <Windows.h>
#include <conio.h>
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>

// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>
#include "camera/CameraManager.hpp"
#include "process/CVisionInterface.hpp"
#include <qtgui/ImageTransform.hpp>



//#include "camera/camera.hpp"
//#include <Windows.h>  // 包含 SetThreadDescription 所需的声明

//#include "camera.hpp"
//#include <algorithm/src/markInterface.h>

namespace fs = std::filesystem;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace sm;

using namespace Pylon;
using namespace GenApi;
using namespace cv;


//MainWindow::MainWindow(QWidget* parent)
//    : QMainWindow(parent)
//    , ui(new Ui::MainWindow)
//{
//    ui->setupUi(this);
//
//    //setWindowState(Qt::WindowMaximized);设置窗口最大化
//    //QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
//   // QPushButton* loadFeatureButton = new QPushButton(codec->toUnicode("加载特征轨迹"), this);
//    //loadFeatureButton->setGeometry(760, 100, 150, 40);
//    connect(ui->pushButton_6, &QPushButton::clicked, this, &MainWindow::on_loadFeatureButton_clicked);
//    // 初始化图像显示相关
//    // 添加 graphicsView 到 verticalLayout 中
//    //if (ui->verticalLayout && ui->graphicsView->parent() == nullptr) {
//    //    ui->verticalLayout->addWidget(ui->graphicsView);
//    //}
//    scene = new QGraphicsScene(this);
//    ui->graphicsView->setScene(scene);
//    
//    //ui->graphicsView->setRenderHint(QPainter::Antialiasing);
//    ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
//    ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
//    //connect(ui->pushButton_3, &QPushButton::clicked, this, &MainWindow::on_pushButton_3_clicked);
//    scene2 = new QGraphicsScene(this);  // 记得在 MainWindow 类中定义 scene2 成员
//    ui->graphicsView_2->setScene(scene2);
//    ui->graphicsView_2->setDragMode(QGraphicsView::ScrollHandDrag);
//    ui->graphicsView_2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
//
//    logModel = new QStringListModel(this);
//    ui->listView_log->setModel(logModel);  // 绑定模型到 listView
//    //// 日志输出
//    //sm::Write_Log::get_init()->Log("Hello");
//    //sm::Write_Log::get_init()->Log("你好结局么");
//    //sm::Write_Log::get_init()->Log("Hello");
//     // 创建并显示图像窗口（不传this，确保独立）
//    //imageWindow = new ImageWindow();  // 不设置 parent
//    //imageWindow->show();              // 独立窗口，可自由移动
//    
//}
// 在MainWindow构造函数中替换原有graphicsView初始化代码
// 在MainWindow构造函数中初始化自定义视图
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    // 初始化第一个视图的ROI状态
    , hasSelectedRoi(false)
    , selectedRoi(QRectF())
    // 初始化第二个视图的ROI状态
    , hasSelectedRoi2(false)
    , selectedRoi2(QRectF())
    ,modelInitialized(false)  // 模板初始化
    ,m_userScaled(false)
{
    ui->setupUi(this);

    /*std::thread outputExeclThread(&sm::CameraManager::UseBaslerCam, &sm::CameraManager::GetInstance());
    SetThreadDescription(outputExeclThread.native_handle(), L"outputExeclThread");
    outputExeclThread.detach();*/

    //CameraManager::GetInstance().UseBaslerCam();
    // 连接按钮信号槽
    connect(ui->pushButton_6, &QPushButton::clicked, this, &MainWindow::on_loadFeatureButton_clicked);

    connect(ui->pushButton_7, &QPushButton::clicked, this, &MainWindow::on_Camera_test);

    connect(ui->pushButton_tool1, &QPushButton::clicked, this, &MainWindow::on_OpenCamera_test);

    connect(ui->pushButton_tool2, &QPushButton::clicked, this, &MainWindow::on_CloseCamera_test);

    // 为graphicsView的视口安装事件过滤器
    ui->graphicsView->viewport()->installEventFilter(this);

    // 初始化最后一次变换矩阵
    m_lastTransform = ui->graphicsView->transform();
    // --------------------------
    // 替换第一个视图（保留布局）
    // --------------------------
    scene = new QGraphicsScene(this);

    // 保存原有视图的布局相关属性
    QGraphicsView* oldView1 = ui->graphicsView;
    QWidget* parentWidget1 = oldView1->parentWidget();
    QLayout* parentLayout1 = parentWidget1 ? parentWidget1->layout() : nullptr;
    QSizePolicy sizePolicy1 = oldView1->sizePolicy();
    QMargins margins1 = oldView1->contentsMargins();
    int minimumWidth1 = oldView1->minimumWidth();
    int minimumHeight1 = oldView1->minimumHeight();
    int maximumWidth1 = oldView1->maximumWidth();
    int maximumHeight1 = oldView1->maximumHeight();

    // 创建自定义视图并继承原有属性
    SelectableGraphicsView* newView1 = new SelectableGraphicsView(parentWidget1);
    newView1->setScene(scene);
    newView1->setRenderHint(QPainter::Antialiasing);
    newView1->setDragMode(QGraphicsView::ScrollHandDrag);
    newView1->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    newView1->setMouseTracking(true);
    // 关键：继承原有布局属性
    newView1->setSizePolicy(sizePolicy1);
    newView1->setContentsMargins(margins1);
    newView1->setMinimumSize(minimumWidth1, minimumHeight1);
    newView1->setMaximumSize(maximumWidth1, maximumHeight1);

    // 如果存在布局，用布局管理器替换控件（保持布局关系）
    if (parentLayout1) {
        // 找到原控件在布局中的位置
        QLayoutItem* item = parentLayout1->replaceWidget(oldView1, newView1);
        if (item) {
            // 保留原有的拉伸因子
            if (auto boxLayout = qobject_cast<QBoxLayout*>(parentLayout1)) {
                int index = parentLayout1->indexOf(newView1);
                boxLayout->setStretch(index, boxLayout->stretch(index));
            }
            delete item; // 释放原有布局项
        }
    }
    else {
        // 没有布局时直接继承位置和大小
        newView1->setGeometry(oldView1->geometry());
    }

    // 替换ui指针指向新视图
    delete oldView1;
    ui->graphicsView = newView1;

    // 连接信号槽
    connect(ui->graphicsView, &SelectableGraphicsView::rectSelected,
        this, &MainWindow::onFirstViewRectSelected);

    // --------------------------
    // 替换第二个视图（保留布局）
    // --------------------------
    scene2 = new QGraphicsScene(this);

    // 保存原有视图的布局相关属性
    QGraphicsView* oldView2 = ui->graphicsView_2;
    QWidget* parentWidget2 = oldView2->parentWidget();
    QLayout* parentLayout2 = parentWidget2 ? parentWidget2->layout() : nullptr;
    QSizePolicy sizePolicy2 = oldView2->sizePolicy();
    QMargins margins2 = oldView2->contentsMargins();
    int minimumWidth2 = oldView2->minimumWidth();
    int minimumHeight2 = oldView2->minimumHeight();
    int maximumWidth2 = oldView2->maximumWidth();
    int maximumHeight2 = oldView2->maximumHeight();

    // 创建自定义视图并继承原有属性
    SelectableGraphicsView* newView2 = new SelectableGraphicsView(parentWidget2);
    newView2->setScene(scene2);
    newView2->setRenderHint(QPainter::Antialiasing);
    newView2->setDragMode(QGraphicsView::ScrollHandDrag);
    newView2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    newView2->setMouseTracking(true);
    // 关键：继承原有布局属性
    newView2->setSizePolicy(sizePolicy2);
    newView2->setContentsMargins(margins2);
    newView2->setMinimumSize(minimumWidth2, minimumHeight2);
    newView2->setMaximumSize(maximumWidth2, maximumHeight2);

    // 如果存在布局，用布局管理器替换控件（保持布局关系）
    if (parentLayout2) {
        QLayoutItem* item = parentLayout2->replaceWidget(oldView2, newView2);
        if (item) {
            if (auto boxLayout = qobject_cast<QBoxLayout*>(parentLayout2)) {
                int index = parentLayout2->indexOf(newView2);
                boxLayout->setStretch(index, boxLayout->stretch(index));
            }
            delete item;
        }
    }
    else {
        newView2->setGeometry(oldView2->geometry());
    }

    // 替换ui指针指向新视图
    delete oldView2;
    ui->graphicsView_2 = newView2;

    // 连接信号槽
    connect(ui->graphicsView_2, &SelectableGraphicsView::rectSelected,
        this, &MainWindow::onSecondViewRectSelected);
    // 连接第一个视图的菜单动作（用于第一个视图的ROI）
    connect(ui->graphicsView, &SelectableGraphicsView::menuActionTriggered,
        this, &MainWindow::onFirstViewMenuAction);

    // 新增：连接第二个视图的菜单动作（用于第二个视图的ROI）
    connect(ui->graphicsView_2, &SelectableGraphicsView::menuActionTriggered,
        this, &MainWindow::onSecondViewMenuAction);
    // 初始化日志模型
    logModel = new QStringListModel(this);
    ui->listView_log->setModel(logModel);
    this->showFullScreen();

    // 移除最小化按钮
    setWindowFlags(windowFlags()  & ~Qt::WindowMaximizeButtonHint);
    /*connect(ui->pushButton_detailsDontCare, &QPushButton::clicked,
        this, &MainWindow::on_pushButton_detailsDontCare_clicked);*/

    sm::CVisionInterface::Ins().Init();   // 只初始化一次
}
// 添加框选事件处理函数
void MainWindow::onFirstViewRectSelected(const QRectF& rect)
{
    // 在日志中显示选中的区域
    QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
    QString log = QString(codec->toUnicode("第一个视图选中区域: x=%1, y=%2, 宽=%3, 高=%4"))
        .arg(rect.x()).arg(rect.y())
        .arg(rect.width()).arg(rect.height());

    // 添加到日志模型
    QStringList logs = logModel->stringList();
    logs.append(log);
    logModel->setStringList(logs);

    // 滚动到最后一行
    ui->listView_log->scrollToBottom();

    // 在这里可以添加对选中区域的处理逻辑
}

void MainWindow::onSecondViewRectSelected(const QRectF& rect)
{
    // 在日志中显示选中的区域
    QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
    QString log = QString(codec->toUnicode("第二个视图选中区域: x=%1, y=%2, 宽=%3, 高=%4"))
        .arg(rect.x()).arg(rect.y())
        .arg(rect.width()).arg(rect.height());

    // 添加到日志模型
    QStringList logs = logModel->stringList();
    logs.append(log);
    logModel->setStringList(logs);

    // 滚动到最后一行
    ui->listView_log->scrollToBottom();

    // 在这里可以添加对选中区域的处理逻辑
}
// 第一个视图的菜单动作处理（原功能保留）
void MainWindow::onFirstViewMenuAction(const QString& action)
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    if (action == codec->toUnicode("save_region")) {
        hasSelectedRoi = true;
        selectedRoi = ui->graphicsView->getSelectedRect();  // 第一个视图的区域
        appendLog(codec->toUnicode("第一个视图：已保存选中区域作为ROI"));
    }
    else if (action == codec->toUnicode("clear_region")) {
        hasSelectedRoi = false;
        selectedRoi = QRectF();
        appendLog(codec->toUnicode("第一个视图：已清除选中区域"));
    }
}

// 新增：第二个视图的菜单动作处理
void MainWindow::onSecondViewMenuAction(const QString& action)
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    if (action == codec->toUnicode("save_region")) {
        hasSelectedRoi2 = true;
        selectedRoi2 = ui->graphicsView_2->getSelectedRect();  // 第二个视图的区域
        appendLog(codec->toUnicode("第二个视图：已保存选中区域作为ROI"));
    }
    else if (action == codec->toUnicode("clear_region")) {
        hasSelectedRoi2 = false;
        selectedRoi2 = QRectF();
        appendLog(codec->toUnicode("第二个视图：已清除选中区域"));
    }
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::appendLog(const QString& msg)
{
    QString time = QTime::currentTime().toString("HH:mm:ss");
    QString logEntry = QString("[%1] %2").arg(time, msg);

    logList.append(logEntry);

    // 保持最大条数（例如 100）
    const int maxLogCount = 100;
    if (logList.size() > maxLogCount) {
        logList.removeFirst();
    }

    logModel->setStringList(logList);

    // 滚动到底部
    QModelIndex lastIndex = logModel->index(logList.size() - 1);
    ui->listView_log->scrollTo(lastIndex);
}


void MainWindow::on_actionw_triggered()
{
    // 打开文件操作
    //QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
    QString filePath = QFileDialog::getOpenFileName(this, "", "", "Images (*.png *.jpg *.bmp)");

    if (!filePath.isEmpty()) {
        loadImage(filePath);
    }
}
void MainWindow::on_pushButton_3_clicked()
{
   filePath_orgin = QFileDialog::getOpenFileName(
        this,
        "",  
        "",
        "Images (*.png *.jpg *.bmp)"
    );

    if (!filePath_orgin.isEmpty()) {
        // 显示在 lineEdit 上
        ui->lineEdit->setText(filePath_orgin);

        // 加载显示图片
        loadImage(filePath_orgin);
    }
}
void MainWindow::on_pushButton_8_clicked()
{
    // 选择图像文件
    filePath = QFileDialog::getOpenFileName(
        this,
        "",
        "",
        "Images (*.png *.jpg *.bmp)"
    );

    if (!filePath.isEmpty()) {
        // 显示路径到 lineEdit_10
        ui->lineEdit_10->setText(filePath);

        // 加载图像到第二视图
        loadImageToSecondView(filePath);
    }
}

//void MainWindow::loadImage(const QString& path)
//{
//    QPixmap pixmap(path);
//    if (pixmap.isNull()) {
//        qDebug() << "图片加载失败：" << path;
//        return;
//    }
//
//    // 初始化 scene（只初始化一次）
//    if (!scene) {
//        scene = new QGraphicsScene(this);
//        ui->graphicsView->setScene(scene);
//        //ui->graphicsView->setRenderHint(QPainter::Antialiasing);
//        ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
//        ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
//    }
//    else {
//        scene->clear();  // 清除旧图像
//    }
//
//    // 添加图片并设置场景大小
//    pixmapItem = scene->addPixmap(pixmap);
//    scene->setSceneRect(pixmap.rect());
//
//    // 重置视图
//    scaleFactor = 1.0;
//    ui->graphicsView->resetTransform();
//    ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);
//
//    qDebug() << "图片加载成功：" << path;
//}
void MainWindow::loadImage(const QString& path)
{
    QPixmap pixmap(path);
    if (pixmap.isNull()) {
        qDebug() << "图片加载失败：" << path;
        return;
    }

    originalPixmap = pixmap;
    binaryPixmap = generateBinaryPixmap(pixmap);  // 自动生成二值图
    showingBinary = false;

    if (!scene) {
        scene = new QGraphicsScene(this);
        ui->graphicsView->setScene(scene);
        ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
        ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    }
    else {
        scene->clear();
    }

    pixmapItem = scene->addPixmap(originalPixmap);
    scene->setSceneRect(originalPixmap.rect());

    scaleFactor1 = 1.0;
    ui->graphicsView->resetTransform();
    ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);

    qDebug() << "图片加载成功：" << path;
}
void MainWindow::loadImageToSecondView(const QString& path)
{
    QPixmap pixmap(path);
    if (pixmap.isNull()) {
        qDebug() << "图片加载失败：" << path;
        return;
    }

    originalPixmap2 = pixmap;
    binaryPixmap2 = generateBinaryPixmap(pixmap);
    showingBinary2 = false;

    if (!scene2) {
        scene2 = new QGraphicsScene(this);
        ui->graphicsView_2->setScene(scene2);
        ui->graphicsView_2->setDragMode(QGraphicsView::ScrollHandDrag);
        ui->graphicsView_2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    }
    else {
        scene2->clear();
    }

    pixmapItem2 = scene2->addPixmap(originalPixmap2);
    scene2->setSceneRect(originalPixmap2.rect());

    ui->graphicsView_2->resetTransform();
    ui->graphicsView_2->fitInView(pixmapItem2, Qt::KeepAspectRatio);

    qDebug() << "图片加载成功（第二视图）：" << path;
}


QPixmap MainWindow::generateBinaryPixmap(const QPixmap& pixmap)
{
    QImage image = pixmap.toImage().convertToFormat(QImage::Format_Grayscale8);

    // 转为 OpenCV Mat
    cv::Mat mat(image.height(), image.width(), CV_8UC1, (void*)image.bits(), image.bytesPerLine());

    // 读取 lineEdit_4 的值，计算 offset
    bool ok = false;
    double v = ui->lineEdit_4->text().toDouble(&ok);
    if (!ok) v = 0.0;  // 如果无效就当0处理
    const double scale = 0.017;
    int offset = static_cast<int>(v / scale);

    // 计算 ROI 以中心为点，宽高减去 offset
    int centerX = mat.cols / 2;
    int centerY = mat.rows / 2;
    int halfWidth = max((mat.cols - offset) / 2, 0);
    int halfHeight = max((mat.rows - offset) / 2, 0);
    // 选择较小的半边长作为正方形 ROI 的半边长
    int halfSize = min(halfWidth, halfHeight);
    int x = max(centerX - halfSize, 0);
    int y = max(centerY - halfSize, 0);
    // 调整半边长以确保 ROI 不超出图像边界
    halfSize = min(halfSize, mat.cols - x, mat.rows - y);

    if (halfSize <= 0 ) {
        // ROI无效，直接整体二值化或返回原图
        // 这里选择返回原图
        return pixmap;
    }
    // 正方形 ROI 的边长为半边长的两倍
    int size = halfSize * 2;
    cv::Rect roi(x, y, size, size);


    // 提取 ROI 区域
    cv::Mat roiMat = mat(roi);
    cv::Mat binaryRoiMat;

    int threshold_value = 180;  // 默认阈值

    if (ui->checkBox_2->isChecked()) {
        // 自动阈值：Otsu
        cv::threshold(roiMat, binaryRoiMat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    else {
        // 手动读取 lineEdit_3 中的阈值
        bool ok2;
        int value = ui->lineEdit_3->text().toInt(&ok2);
        if (ok2 && value >= 0 && value <= 255) {
            threshold_value = value;
        }
        else {
            qDebug() << "手动阈值无效，使用默认值 128";
            threshold_value = 128;
        }

        cv::threshold(roiMat, binaryRoiMat, threshold_value, 255, cv::THRESH_BINARY);
    }

    // 把二值化结果写回 mat 中对应 ROI 区域
    binaryRoiMat.copyTo(mat(roi));

    // 转回 QImage → QPixmap
    QImage binImg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    return QPixmap::fromImage(binImg.copy());
}

MarkRect calculateROI(const cv::Mat& mat, bool hasSelectedRoi, const QRectF& selectedRoi) {
    MarkRect roi;

    if (hasSelectedRoi && !selectedRoi.isEmpty()) {
        // 使用选中区域作为ROI
        int x = static_cast<int>(qRound(selectedRoi.x()));
        int y = static_cast<int>(qRound(selectedRoi.y()));
        int width = static_cast<int>(qRound(selectedRoi.width()));
        int height = static_cast<int>(qRound(selectedRoi.height()));

        // 边界检查
        x = qMax(0, x);
        y = qMax(0, y);
        width = qMin(mat.cols - x, width);
        height = qMin(mat.rows - y, height);
        width = qMax(10, width);
        height = qMax(10, height);

        roi.x = x;
        roi.y = y;
        roi.width = width;
        roi.height = height;

        qDebug() << "使用选中ROI: 位置(" << x << "," << y << ") 大小(" << width << "×" << height << ")";
    }
    else {
        // 无选中区域，使用整个图像
        roi.x = 0;
        roi.y = 0;
        roi.width = mat.cols;
        roi.height = mat.rows;

        qDebug() << "使用全图ROI: 大小(" << mat.cols << "×" << mat.rows << ")";
    }

    return roi;
}

int MainWindow::convertMarkTypeToInt(const QString& markTypeStr)
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    QString decodedStr = codec->toUnicode(markTypeStr.toUtf8());

    // 根据之前讨论的晶圆标记类型进行映射
    if (markTypeStr == "Cross Mark") {
        return 0; // Cross Mark
    }
    else if (markTypeStr == "Bar-in-Bar") {
        return 1; // Bar-in-Bar
    }
    else if (markTypeStr == "Box-in-Box") {
        return 2; // Box-in-Box
    }
    else if (markTypeStr == "Frame-in-Frame") {
        return 3; // Frame-in-Frame
    }
    else if (markTypeStr == "AIM") {
        return 4; // AIM (Advanced Imaging Mark)
    }
    else {
        // 默认返回0
        return 0;
    }
}

void MainWindow::on_pushButton_clicked()
{
    if (!pixmapItem) return;

    showingBinary = !showingBinary;

    // 每次点击都重新生成 binaryPixmap，确保阈值变化能立即反映
    if (showingBinary) {
        binaryPixmap = generateBinaryPixmap(originalPixmap);
    }

    // 更新 scene 中的图像
    scene->removeItem(pixmapItem);
    pixmapItem = scene->addPixmap(showingBinary ? binaryPixmap : originalPixmap);

    // 更新按钮文字
    QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
    //ui->pushButton->setText(showingBinary ? codec->toUnicode("真实显示") : codec->toUnicode("二值化"));

    // 保持图像适应视图
    ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);
}
//生成模板
void MainWindow::on_pushButton_5_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");  // 编码格式
    bool result = 0;
    QString tempInputPath;
    MarkRect roi;
    // 记录开始时间
    qint64 startTime = QDateTime::currentMSecsSinceEpoch();
    //离线测试
    if (onlinetest == 0) {
        
        if (!pixmapItem) {
            QMessageBox::warning(this, codec->toUnicode("提示"), codec->toUnicode("请先加载图像！"));
            return;
        }

        // 原图转换成 BGR 图像（cv::Mat）
        //QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
        QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
        cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
        if (mat.empty()) {
            QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("图像转换失败！"));
            return;
        }

        // 确保临时目录存在并保存临时图像
         tempInputPath = QCoreApplication::applicationDirPath() + "/temp_input.jpg";
        if (!cv::imwrite(tempInputPath.toStdString(), mat)) {
            QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("临时图像保存失败！"));
            return;
        }

        // ✅ 读取二值化阈值（范围检查优化）
        bool ok_thresh = false;
        double threshold = ui->lineEdit_3->text().toDouble(&ok_thresh);
        if (!ok_thresh || threshold < 0 || threshold > 255) {
            threshold = 128.0;  // 默认值
            appendLog(codec->toUnicode("二值化阈值无效，使用默认值 128"));
        }
        else {
            appendLog(QString(codec->toUnicode("使用二值化阈值: %1")).arg(threshold));
        }

        // 计算ROI偏移量
        roi = calculateROI(mat, hasSelectedRoi, selectedRoi);
        appendLog(QString(codec->toUnicode("使用ROI: 位置(%1,%2) 大小(%3×%4)"))
            .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));
        // 1. 生成模板
        //const char* templatePath = "template.jpg";
        const char* xmlPath = "edge_points.xml";

        // 可选：调整参数 NCC模板生成，使用时取消注释
        //NCC_SetCannyParams(150, 200);
        //NCC_SetContourAreaThreshold(10.0);
        NccRect roi1;
        roi1.x = roi.x;
        roi1.y = roi.y;
        roi1.width = roi.width;
        roi1.height = roi.height;
        result = NCC_CreateTemplate(filePath_orgin.toStdString().c_str(), xmlPath, &roi1, 1, -5, 5);
    }
    else {
        //在线测试
        //1获取图像
        cv::Mat frame;
        //APIErrCode err = sm::CVisionInterface::Ins().CameraCapture(frame);
        //if (err != APIErrCode::SUCCESS) {
        //    QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("相机采集失败！"));
        //    return;
        //}

        //// 2. 停止实时显示和相机抓图
        //if (m_timer && m_timer->isActive()) {
        //    m_timer->stop();
        //    ui->pushButton_7->setText(codec->toUnicode("开始实时显示"));
        //}
        //sm::CVisionInterface::Ins().StopCapture();
        /////////////////
        QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
        cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
        frame = mat.clone();
        ///////////////
        // 3. 检查图像有效性
        if (frame.empty()) {
            QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("采集的图像为空！"));
            return;
        }

        // 4. 保存临时图像文件（与离线模式保持一致）
        tempInputPath = QCoreApplication::applicationDirPath() + "/temp_input.jpg";
        if (!cv::imwrite(tempInputPath.toStdString(), frame)) {
            QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("临时图像保存失败！"));
            return;
        }

        // 5. 读取二值化阈值（如果需要，可复用离线模式的阈值逻辑）
        bool ok_thresh = false;
        double threshold = ui->lineEdit_3->text().toDouble(&ok_thresh);
        if (!ok_thresh || threshold < 0 || threshold > 255) {
            threshold = 128.0;
            appendLog(codec->toUnicode("二值化阈值无效，使用默认值 128"));
        }
        else {
            appendLog(QString(codec->toUnicode("使用二值化阈值: %1")).arg(threshold));
        }

        // 6. 计算 ROI（基于采集的 frame）
        roi = calculateROI(frame, hasSelectedRoi, selectedRoi);
        appendLog(QString(codec->toUnicode("使用ROI: 位置(%1,%2) 大小(%3×%4)"))
            .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));

        // 7. 调用 NCC 创建模板（传入临时文件路径）
        const char* xmlPath = "edge_points.xml";
        NccRect roi1;
        roi1.x = roi.x;
        roi1.y = roi.y;
        roi1.width = roi.width;
        roi1.height = roi.height;

        // 注意：filePath_orgin 可能仅用于离线模式，在线测试时直接使用临时文件路径
        result = NCC_CreateTemplate(tempInputPath.toStdString().c_str(), xmlPath, &roi1, 1, -5, 5);
    }
    

   
    

    //// ✅ 模型保存路径（确保目录存在）
    //QString saveDir = "./demo";
    //QDir().mkpath(saveDir);  // 确保保存目录存在
    //QString modelName = "demo";

    //// ✅ 调用算法接口（传递自定义ROI结构体）
    //bool result = GenerateMarkTemplate(
    //    tempInputPath.toStdString().c_str(),
    //    saveDir.toStdString().c_str(),
    //    modelName.toStdString().c_str(),
    //    static_cast<int>(threshold),  // 二值化阈值转int
    //    &roi  // 传递自定义ROI（传nullptr表示全图）
    //);

    // 处理结果
    if (result) {
        /*QMessageBox::information(this, codec->toUnicode("成功"),
            codec->toUnicode("模板已保存到 ./demo 下，名称为 demo！"));*/
        modelInitialized = false;
        // 读取并绘制特征点（路径优化）
        QVector<QPointF> features;
        //QString filePath = "generate_features.txt";  // 与算法输出路径一致
        //QFile file(filePath);
        //if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        //    appendLog("无法打开特征点文件: " + filePath);
        //}
        //else {
        //    QTextStream in(&file);
        //    while (!in.atEnd()) {
        //        double x = 0, y = 0;
        //        in >> x >> y;
        //        if (in.status() == QTextStream::Ok) {
        //            features.append(QPointF(x, y));
        //        }
        //        else {
        //            break;
        //        }
        //    }
        //    file.close();
        //   
        //    appendLog(QString(codec->toUnicode("读取到 %1 个特征点")).arg(features.size()));
        //}
        // 清空features（如果需要）
        features.clear();
        sm::Core* core = sm::Core::get_init();
        // 预留空间以提高性能
        features.reserve(core->temp_features.size());

        // 遍历并转换每个点
        for (const auto& pt : core->temp_features) {
            features.append(QPointF(pt.x, pt.y));
        }
        appendLog(QString(codec->toUnicode("读取到 %1 个特征点")).arg(features.size()));
        scene->clear();
        QImage originalImage(tempInputPath);
        if (!originalImage.isNull()) {
            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(originalImage));
            scene->addItem(item);
            ui->graphicsView->setScene(scene);
            ui->graphicsView->fitInView(item, Qt::KeepAspectRatio);
        }
        drawFeatureTrajectory1(features, roi.x, roi.y, 0, Qt::red);
    }
    else {
 /*       QMessageBox::warning(this, codec->toUnicode("失败"),
            codec->toUnicode("模板生成失败，请检查图像或参数！"));*/
    }

    // 计算并输出总耗时
    qint64 totalTime = QDateTime::currentMSecsSinceEpoch() - startTime;
    double totalTimeSec = totalTime / 1000.0;
    appendLog(QString(codec->toUnicode("模板生成%1，总耗时：%2 秒（%3 毫秒）"))
        .arg(result ? codec->toUnicode("成功") : codec->toUnicode("失败"))
        .arg(totalTimeSec, 0, 'f', 2)
        .arg(totalTime));
}

//模板匹配
void MainWindow::on_pushButton_2_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

  //// 模型只初始化一次 - 现在是类成员变量  第一种方法 暂时屏蔽
  //  if (!modelInitialized) {
  //      if (!InitMarkMatcher("./demo", "demo")) {
  //          QMessageBox::critical(this, "模型加载失败", "模型加载失败，请检查模型路径和内容！");
  //          return;
  //      }
  //      modelInitialized = true;
  //  }

    // ✅ 从 UI 获取阈值和 ROI 设置
    bool ok_thresh = false;
    int threshold = ui->lineEdit_3->text().toDouble(&ok_thresh);
    if (!ok_thresh || threshold < 0 || threshold > 255) {
        threshold = 128;  // 确保为int类型（算法接口要求）
    }
    QString algorithm = ui->comboBox_algorithm->currentText();
    bool success = false;
    if (onlinetest == 0)
    {
        // ✅ 连续测试
        if (ui->checkBox->isChecked()) {
            if (testImageFiles.isEmpty()) {
                QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("请先选择连续测试文件夹！"));
                return;
            }

            QString resultFile = testFolderPath + "/Test_result.csv";
            std::ofstream resultOut(resultFile.toStdString());
            resultOut << "文件路径及文件名,测试结果,offset_X,offset_Y,offset_R,测试时间(ms),测试分数,错误代码\n";

            int numstest = 0;

            for (const QString& filePath : testImageFiles) {
                QImage image(filePath);
                numstest++;
                if (numstest > 10)
                {
                    break;
                }
                if (image.isNull()) continue;

                QString tempPath = QCoreApplication::applicationDirPath() + "/temp_input.png";
                if (!image.save(tempPath)) {
                    appendLog(codec->toUnicode("临时图像保存失败: %1").arg(filePath));
                    continue;
                }
                QString outputPath = QCoreApplication::applicationDirPath() + "/temp_result.jpg";

                // 读取图像尺寸，计算 ROI（核心修改：使用自定义MarkRect）
                //cv::Mat mat = cv::imread(tempPath.toStdString());
                //if (mat.empty()) {
                //    appendLog(codec->toUnicode("无法读取图像: %1").arg(filePath));
                //    continue;
                //}
                QImage image1 = originalPixmap2.toImage().convertToFormat(QImage::Format_RGB888);
                cv::Mat mat(image1.height(), image1.width(), CV_8UC3, (void*)image1.bits(), image1.bytesPerLine());
                // 计算ROI偏移量
                MarkRect roi = calculateROI(mat, hasSelectedRoi2, selectedRoi2);

                // 运行匹配（传入自定义ROI）
                double offset_x = 0.0, offset_y = 0.0, offset_r = 0.0;
                double time_ms = 0.0;
                float similarity = 0.0f;

                //bool success = RunMarkMatchingSingle(
                //    filePath.toStdString().c_str(),
                //    outputPath.toStdString().c_str(),
                //    &offset_x, &offset_y, &offset_r,
                //    &similarity, &time_ms,
                //    &threshold,
                //    &roi  // 传入自定义MarkRect的地址（关键修改）
                //);
                // 
                /*NccRect roi0;
                roi0.x = roi.x;
                roi0.y = roi.y;
                roi0.width = roi.width;
                roi0.height = roi.height;
                NccMatchResult result1;
                const char* xmlPath = "edge_points.xml";
                bool success = NCC_PerformMatching(filePath.toStdString().c_str(), xmlPath, &roi0, &result1);
                offset_x = result1.x;
                offset_y = result1.y;*/
                // 
                ///////////
                NccRect roi0;
                roi0.x = roi.x;
                roi0.y = roi.y;
                roi0.width = roi.width;
                roi0.height = roi.height;
                NccMatchResult result0;
                const char* xmlPath = "edge_points.xml";
                if (algorithm == "icp")
                {
                    success = NCC_PerformMatching(filePath.toStdString().c_str(), xmlPath, &roi0, &result0);
                }
                else
                {
                    QString markTypeStr = ui->comboBox_symmetry->currentText();
                    int markType = convertMarkTypeToInt(markTypeStr); // 需要实现这个转换函数
                    success = Region_PerformMatching(filePath.toStdString().c_str(), outputPath.toStdString().c_str(), &roi0, &result0, markType);
                }

                offset_x = result0.x;
                offset_y = result0.y;
                offset_r = result0.angle;
                if (success) {
                    std::string utf8FilePath = filePath.toUtf8().toStdString();
                    resultOut << utf8FilePath << ",测试成功,"
                        << offset_x << "," << offset_y << "," << offset_r << ","
                        << time_ms << "," << similarity << ",0\n";
                    appendLog(codec->toUnicode("成功 - %1 | X: %2 Y: %3 R: %4° 分数: %5 耗时: %6 ms")
                        .arg(filePath)
                        .arg(offset_x, 0, 'f', 3)
                        .arg(offset_y, 0, 'f', 3)
                        .arg(offset_r, 0, 'f', 3)
                        .arg(similarity, 0, 'f', 3)
                        .arg(time_ms, 0, 'f', 3));

                    // 特征点绘制（路径优化为相对路径）
                    //QVector<QPointF> features;
                    //QString featureFilePath = "template_features.txt";  // 与算法输出路径一致
                    //QFile featureFile(featureFilePath);
                    //if (!featureFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    //    appendLog("无法打开特征点文件: " + featureFilePath);
                    //}
                    //else {
                    //    QTextStream in(&featureFile);
                    //    while (!in.atEnd()) {
                    //        double x = 0, y = 0;
                    //        in >> x >> y;
                    //        if (in.status() == QTextStream::Ok) {
                    //            features.append(QPointF(x, y));
                    //        }
                    //        else {
                    //            break;
                    //        }
                    //    }
                    //    featureFile.close();
                    //}

                    // 更新第二视图
                    offset_r += threshold - 6;
                    scene2->clear();
                    QImage originalImage(filePath);
                    if (!originalImage.isNull()) {
                        QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(originalImage));
                        scene2->addItem(item);
                        ui->graphicsView_2->setScene(scene2);
                        ui->graphicsView_2->fitInView(item, Qt::KeepAspectRatio);
                    }
                    //drawFeatureTrajectory(features, offset_x, offset_y, offset_r, Qt::red);

                }
                else {
                    resultOut << filePath.toStdString() << ",测试失败,0,0,0,0,0,9999\n";
                    appendLog(codec->toUnicode("失败 - %1").arg(filePath));
                }

                QCoreApplication::processEvents();  // 保持界面响应
            }

            QMessageBox::information(this, codec->toUnicode("完成"),
                codec->toUnicode("连续测试完成，结果保存在:\n") + resultFile);
        }

        // ✅ 单张测试
        else {
            if (originalPixmap2.isNull()) {
                QMessageBox::warning(this, codec->toUnicode("提示"), codec->toUnicode("请先加载图像！"));
                return;
            }

            // 转换图像为OpenCV格式
            QImage image = originalPixmap2.toImage().convertToFormat(QImage::Format_RGB888);
            cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
            if (mat.empty()) {
                QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("图像转换失败！"));
                return;
            }

            QString tempInputPath = QCoreApplication::applicationDirPath() + "/temp_input.png";
            cv::imwrite(tempInputPath.toStdString(), mat);
            QString outputPath = QCoreApplication::applicationDirPath() + "/match_result.jpg";

            // 计算ROI偏移量
            MarkRect roi = calculateROI(mat, hasSelectedRoi2, selectedRoi2);
            // 2. 执行匹配

            /// NCC模板匹配过程，使用时取消注释
            NccRect roi1;
            roi1.x = roi.x;
            roi1.y = roi.y;
            roi1.width = roi.width;
            roi1.height = roi.height;
            NccMatchResult result1;
            const char* xmlPath = "edge_points.xml";
            if (algorithm == "icp")
            {
                success = NCC_PerformMatching(filePath.toStdString().c_str(), xmlPath, &roi1, &result1);
            }
            else
            {
                QString markTypeStr = ui->comboBox_symmetry->currentText();
                int markType = convertMarkTypeToInt(markTypeStr); // 需要实现这个转换函数
                success = Region_PerformMatching(filePath.toStdString().c_str(), outputPath.toStdString().c_str(), &roi1, &result1, markType);
            }
            //bool success = NCC_PerformMatching(filePath.toStdString().c_str(), xmlPath,&roi1, &result1);

             //////////////////////////////////////////
             //bool success = Region_PerformMatching(filePath.toStdString().c_str(), xmlPath, &roi1, &result1);
             //
             // 运行匹配（传入自定义ROI）
            double offset_x = 0.0, offset_y = 0.0, offset_r = 0.0;
            double time_ms = 0.0;
            float similarity = 0.0f;
            offset_x = result1.x;
            offset_y = result1.y;
            offset_r = result1.angle;
            similarity = result1.similarity;
            time_ms = result1.time_ms;
            ////第二种方法
            //bool success = RunMarkMatchingSingle(
            //    tempInputPath.toStdString().c_str(),
            //    outputPath.toStdString().c_str(),
            //    &offset_x, &offset_y, &offset_r,
            //    &similarity, &time_ms,
            //    &threshold,
            //    &roi  // 传入自定义MarkRect的地址（关键修改）
            //);

            if (success) {
                //offset_r -= threshold - 6;
                // 更新UI显示
      /*          ui->lineEdit_5->setText(QString::number(offset_x, 'f', 3));
                ui->lineEdit_6->setText(QString::number(offset_y, 'f', 3));
                ui->lineEdit_7->setText(QString::number(offset_r, 'f', 3));
                ui->lineEdit_8->setText(QString::number(time_ms, 'f', 3));
                ui->lineEdit_9->setText(QString::number(similarity, 'f', 3))*/;
                /*  QMessageBox::information(this, codec->toUnicode("匹配成功"),
                      codec->toUnicode("匹配结果已保存并显示。"));*/
                loadImageToSecondView(tempInputPath);
                appendLog(codec->toUnicode("匹配成功 | X: %1 Y: %2 R: %3° 分数: %4 耗时: %5 ms")
                    .arg(offset_x, 0, 'f', 3)
                    .arg(offset_y, 0, 'f', 3)
                    .arg(offset_r, 0, 'f', 3)
                    .arg(similarity, 0, 'f', 3)
                    .arg(time_ms, 0, 'f', 3));

                // 将结果添加到表格中
                int row = ui->tableWidget_results->rowCount();
                ui->tableWidget_results->insertRow(row);
                ui->tableWidget_results->setItem(row, 0, new QTableWidgetItem(codec->toUnicode("匹配结果")));
                ui->tableWidget_results->setItem(row, 1, new QTableWidgetItem(QString::number(offset_x, 'f', 3)));
                ui->tableWidget_results->setItem(row, 2, new QTableWidgetItem(QString::number(offset_y, 'f', 3)));
                ui->tableWidget_results->setItem(row, 3, new QTableWidgetItem(QString::number(offset_r, 'f', 3)));
                ui->tableWidget_results->setItem(row, 4, new QTableWidgetItem(QString::number(similarity * 100.0, 'f', 2)));
                ui->tableWidget_results->setItem(row, 5, new QTableWidgetItem(QString::number(time_ms, 'f', 3)));

                // 绘制特征点（使用相对路径）
                /*QVector<QPointF> features;
                QString filePath = "template_features.txt";
                QFile file(filePath);
                if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    appendLog("无法打开特征点文件: " + filePath);
                }
                else {
                    QTextStream in(&file);
                    while (!in.atEnd()) {
                        double x = 0, y = 0;
                        in >> x >> y;
                        if (in.status() == QTextStream::Ok) {
                            features.append(QPointF(x, y));
                        }
                        else {
                            break;
                        }
                    }
                    file.close();
                }*/
                //offset_r -= threshold - 6;
                //drawFeatureTrajectory(features, offset_x, offset_y, -offset_r, Qt::red);

                //drawFeatureTrajectory(offset_x, offset_y, -offset_r, Qt::red);

            }
            else {
                /* QMessageBox::warning(this, codec->toUnicode("匹配失败"),
                     codec->toUnicode("未匹配到结果，请检查图像或模型。"));*/
                appendLog(codec->toUnicode("匹配失败！"));
            }
        }
    }
    else
    {
    //在线测试
      // 1. 从相机采集一帧图像
      cv::Mat frame;
      //APIErrCode err = sm::CVisionInterface::Ins().CameraCapture(frame);
      //if (err != APIErrCode::SUCCESS) {
      //    QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("相机采集失败！"));
      //    return;
      //}

      //// 2. 停止实时显示和相机流（避免占用资源）
      //if (m_timer && m_timer->isActive()) {
      //    m_timer->stop();
      //    ui->pushButton_7->setText(codec->toUnicode("开始实时显示"));
      //}
      //sm::CVisionInterface::Ins().StopCapture();
      ///////////
      QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
      cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
      frame = mat.clone();
      ///////////////
      // 3. 检查图像有效性
      if (frame.empty()) {
          QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("采集的图像为空！"));
          return;
      }

      // 4. 保存临时图像文件（与单张测试保持一致格式）
      QString tempInputPath = QCoreApplication::applicationDirPath() + "/temp_input_online.png";
      if (!cv::imwrite(tempInputPath.toStdString(), frame)) {
          QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("临时图像保存失败！"));
          return;
      }
      QString outputPath = QCoreApplication::applicationDirPath() + "/match_result_online.jpg";

      // 5. 计算 ROI（基于采集的图像）
      //    注意：这里需要将 cv::Mat 转为 QImage 或者直接使用 frame 计算 ROI
      //    假设 calculateROI 支持 cv::Mat，直接传入 frame
      MarkRect roi = calculateROI(frame, hasSelectedRoi2, selectedRoi2);
      appendLog(QString(codec->toUnicode("在线测试使用ROI: 位置(%1,%2) 大小(%3×%4)"))
          .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));

      // 6. 执行匹配（与单张测试完全相同的逻辑）
      NccRect roi1;
      roi1.x = roi.x;
      roi1.y = roi.y;
      roi1.width = roi.width;
      roi1.height = roi.height;
      NccMatchResult result1;
      const char* xmlPath = "edge_points.xml";
      bool success = false;

      if (algorithm == "icp") {
          success = NCC_PerformMatching(tempInputPath.toStdString().c_str(), xmlPath, &roi1, &result1);
      }
      else {
          QString markTypeStr = ui->comboBox_symmetry->currentText();
          int markType = convertMarkTypeToInt(markTypeStr);
          success = Region_PerformMatching(tempInputPath.toStdString().c_str(),
              outputPath.toStdString().c_str(),
              &roi1, &result1, markType);
      }

      double offset_x = result1.x;
      double offset_y = result1.y;
      double offset_r = result1.angle;
      float similarity = result1.similarity;
      double time_ms = result1.time_ms;

      if (success) {
          // 更新第二个视图显示采集的图像
          QImage displayImage;
          if (displayImage.load(tempInputPath)) {
              scene2->clear();
              QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(displayImage));
              scene2->addItem(item);
              ui->graphicsView_2->setScene(scene2);
              ui->graphicsView_2->fitInView(item, Qt::KeepAspectRatio);
          }

          // 将结果显示在表格中
          int row = ui->tableWidget_results->rowCount();
          ui->tableWidget_results->insertRow(row);
          ui->tableWidget_results->setItem(row, 0, new QTableWidgetItem(codec->toUnicode("在线测试")));
          ui->tableWidget_results->setItem(row, 1, new QTableWidgetItem(QString::number(offset_x, 'f', 3)));
          ui->tableWidget_results->setItem(row, 2, new QTableWidgetItem(QString::number(offset_y, 'f', 3)));
          ui->tableWidget_results->setItem(row, 3, new QTableWidgetItem(QString::number(offset_r, 'f', 3)));
          ui->tableWidget_results->setItem(row, 4, new QTableWidgetItem(QString::number(similarity * 100.0, 'f', 2)));
          ui->tableWidget_results->setItem(row, 5, new QTableWidgetItem(QString::number(time_ms, 'f', 3)));

          // 添加日志
          appendLog(codec->toUnicode("在线测试匹配成功 | X: %1 Y: %2 R: %3° 分数: %4 耗时: %5 ms")
              .arg(offset_x, 0, 'f', 3)
              .arg(offset_y, 0, 'f', 3)
              .arg(offset_r, 0, 'f', 3)
              .arg(similarity, 0, 'f', 3)
              .arg(time_ms, 0, 'f', 3));

          // 可选：绘制特征轨迹（如果需要，取消注释）
          // drawFeatureTrajectory(features, offset_x, offset_y, offset_r, Qt::red);
      }
      else {
          appendLog(codec->toUnicode("在线测试匹配失败！"));
          QMessageBox::warning(this, codec->toUnicode("匹配失败"),
              codec->toUnicode("未匹配到结果，请检查图像或模型。"));
      }
    }
    
}

//选择文件夹
void MainWindow::on_pushButton_4_clicked()
{
    QString folderPath = QFileDialog::getExistingDirectory(this, "");
    if (folderPath.isEmpty()) return;

    testFolderPath = folderPath;
    ui->lineEdit_2->setText(folderPath);

    QDir dir(folderPath);
    QStringList filters = { "*.jpg", "*.png", "*.bmp" };
    testImageFiles = dir.entryList(filters, QDir::Files);

    if (testImageFiles.isEmpty()) {
        QMessageBox::warning(this, "提示", "该文件夹内没有图像文件！");
        return;
    }

    for (auto& file : testImageFiles) {
        file = dir.absoluteFilePath(file); // 转换为完整路径
    }
}
//对焦
void MainWindow::on_pushButton_9_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    if (ui->checkBox_3->isChecked()) {
        // 连续测试模式
        if (testImageFiles.isEmpty()) {
            QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("请先选择连续测试文件夹！"));
            return;
        }

        QString resultFile = testFolderPath + "/Test_result.csv";
        std::ofstream resultOut(resultFile.toStdString());
        resultOut << "文件路径及文件名,清晰度评分\n";

        for (const QString& filePath : testImageFiles) {
            QImage image(filePath);
            if (image.isNull()) {
                appendLog(codec->toUnicode("无法加载图像: ") + filePath);
                resultOut << filePath.toStdString() << ",加载失败\n";
                continue;
            }

            // 转换为 Mat
            QImage formatted = image.convertToFormat(QImage::Format_RGB888);
            cv::Mat mat(formatted.height(), formatted.width(), CV_8UC3, (void*)formatted.bits(), formatted.bytesPerLine());

            double score = ComputerTenengrad(mat);

            resultOut << filePath.toStdString() << "," << score << "\n";
            appendLog(QString(codec->toUnicode("图像：%1 清晰度得分为：%2"))
                .arg(filePath)
                .arg(score, 0, 'f', 2));

            QCoreApplication::processEvents(); // 保持界面响应
        }

        QMessageBox::information(this, codec->toUnicode("完成"),
            codec->toUnicode("连续清晰度测试完成，结果保存在:\n") + resultFile);
    }
    else {
        // 单张测试模式
        if (originalPixmap.isNull()) {
            QMessageBox::warning(this, codec->toUnicode("提示"), codec->toUnicode("请先加载图像！"));
            return;
        }

        QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
        cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());

        double score = ComputerTenengrad(mat);

        QString msg = QString(codec->toUnicode("图像清晰度（Tenengrad）得分为：%1")).arg(score);
        QMessageBox::information(this, codec->toUnicode("清晰度评分"), msg);
        appendLog(QString(codec->toUnicode("图像清晰度得分为：%1")).arg(score));
    }
}

//对焦稳定性测试
void MainWindow::on_pushButton_10_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    // 1. 获取检测次数（从lineEdit_11读取）
    bool isNumber;
    int detectCount = ui->lineEdit_11->text().toInt(&isNumber);
    // 检查输入有效性
    if (!isNumber || detectCount <= 0) {
        QMessageBox::warning(this, codec->toUnicode("输入错误"),
            codec->toUnicode("请在输入框中填写有效的正整数作为检测次数！"));
        return;
    }

    // 2. 检查图像是否已加载
    if (originalPixmap.isNull()) {
        QMessageBox::warning(this, codec->toUnicode("提示"),
            codec->toUnicode("请先加载图像！"));
        return;
    }

    // 3. 图像格式转换（仅转换一次，提升效率）
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(image.height(), image.width(), CV_8UC3,
        (void*)image.bits(), image.bytesPerLine());

    // 4. 初始化结果存储容器
    QVector<double> results;
    results.reserve(detectCount); // 预分配内存，提升性能

    // 5. 连续检测指定次数
    for (int i = 0; i < detectCount; ++i) {
        // 调用清晰度计算函数（每次检测都重新计算，确保实时性）
        double score = ComputerTenengrad(mat);

        // 存储当前结果
        results.push_back(score);

        // 记录日志（包含检测次数）
        QString logMsg = QString(codec->toUnicode("第%1次检测，图像清晰度得分为：%2"))
            .arg(i + 1)  // 次数从1开始计数
            .arg(score, 0, 'f', 6); // 保留6位小数，提升精度

        appendLog(logMsg);

        // 立即刷新日志显示
        QApplication::processEvents();
    }

    // 6. 计算平稳性指标（3σ分析）
    if (results.isEmpty()) {
        QMessageBox::warning(this, codec->toUnicode("错误"),
            codec->toUnicode("检测结果为空，无法计算平稳性！"));
        return;
    }

    // 6.1 计算平均值（mean）
    double mean = 0.0;
    for (double s : results) {
        mean += s;
    }
    mean /= results.size();

    // 6.2 计算标准差（σ）
    double sigma = 0.0;
    for (double s : results) {
        sigma += pow(s - mean, 2);
    }
    sigma = sqrt(sigma / results.size()); // 总体标准差（适用于全部检测数据）

    // 6.3 计算3σ值
    double threeSigma = 3 * sigma;

    // 6.4 分析数据是否在[mean-3σ, mean+3σ]范围内（3σ原则：99.7%数据应在此区间）
    bool allInRange = true;
    for (double s : results) {
        if (s < (mean - threeSigma) || s >(mean + threeSigma)) {
            allInRange = false;
            break;
        }
    }

    // 7. 输出统计结果（日志+弹窗）
    // 7.1 记录统计日志
    appendLog(codec->toUnicode("===== 检测统计结果 ====="));
    appendLog(QString(codec->toUnicode("检测总次数：%1次")).arg(detectCount));
    appendLog(QString(codec->toUnicode("平均值（mean）：%1")).arg(mean, 0, 'f', 6));
    appendLog(QString(codec->toUnicode("标准差（σ）：%1")).arg(sigma, 0, 'f', 6));
    appendLog(QString(codec->toUnicode("3σ值：%1")).arg(threeSigma, 0, 'f', 6));
    appendLog(QString(codec->toUnicode("所有数据是否在mean±3σ范围内：%1"))
        .arg(allInRange ? codec->toUnicode("是") : codec->toUnicode("否")));
    appendLog(codec->toUnicode("======================="));

    // 7.2 弹窗显示统计结果
    QString statsMsg = codec->toUnicode("连续检测完成！\n\n") +
        codec->toUnicode("检测次数：%1次\n").arg(detectCount) +
        codec->toUnicode("平均值（mean）：%1\n").arg(mean, 0, 'f', 6) +
        codec->toUnicode("标准差（σ）：%1\n").arg(sigma, 0, 'f', 6) +
        codec->toUnicode("3σ值：%1\n\n").arg(threeSigma, 0, 'f', 6) +
        codec->toUnicode("数据平稳性分析：\n") +
        codec->toUnicode("所有检测结果%1在均值±3σ范围内\n")
        .arg(allInRange ? codec->toUnicode("均") : codec->toUnicode("未均")) +
        codec->toUnicode("（3σ原则：99.7%数据应在此区间内）");
    QMessageBox::information(this, codec->toUnicode("连续检测统计结果"), statsMsg);
}
//晶圆标记稳定性测试
void MainWindow::on_pushButton_11_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    // ✅ 模型只初始化一次
    static bool modelInitialized = false;
    if (!modelInitialized) {
        if (!InitMarkMatcher("./demo", "demo")) {
            QMessageBox::critical(this, "模型加载失败", "模型加载失败，请检查模型路径和内容！");
            return;
        }
        modelInitialized = true;
    }

    // ✅ 从 UI 获取阈值和 ROI 设置
    bool ok_thresh = false;
    int threshold = ui->lineEdit_3->text().toDouble(&ok_thresh);
    if (!ok_thresh || threshold < 0 || threshold > 255) {
        threshold = 128;  // 修正为int类型，避免浮点警告
        qDebug() << "二值化阈值无效，使用默认 128";
    }

    bool ok_roi = false;
    double mm = ui->lineEdit_4->text().toDouble(&ok_roi);
    double scale = 0.017;
    int offset = ok_roi ? static_cast<int>(mm / scale) : 0;

    // 1. 获取测试次数（从lineEdit_12读取）改成lineEdit_8
    bool isNumber;
    int testCount = ui->lineEdit_8->text().toInt(&isNumber);
    if (!isNumber || testCount <= 0) {
        QMessageBox::warning(this, codec->toUnicode("输入错误"),
            codec->toUnicode("请在输入框中填写有效的正整数作为测试次数！"));
        return;
    }

    // 2. 检查图像是否已加载
    if (originalPixmap2.isNull()) {
        QMessageBox::warning(this, codec->toUnicode("提示"),
            codec->toUnicode("请先加载图像！"));
        return;
    }

    // 3. 图像格式转换并保存临时文件
    QImage image = originalPixmap2.toImage().convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(image.height(), image.width(), CV_8UC3,
        (void*)image.bits(), image.bytesPerLine());
    if (mat.empty()) {
        QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("图像转换失败！"));
        return;
    }

    QString tempInputPath = QCoreApplication::applicationDirPath() + "/temp_input.jpg";
    if (!cv::imwrite(tempInputPath.toStdString(), mat)) {
        QMessageBox::warning(this, codec->toUnicode("错误"), codec->toUnicode("临时图像保存失败！"));
        return;
    }

    // 4. 计算ROI区域（核心修改：支持选中区域或默认计算）
    MarkRect roi;  // 使用自定义ROI结构体
    bool useSelectedRoi = false;

    // 优先使用界面选中的ROI（假设使用第二个视图的选中区域）
    if (hasSelectedRoi2 && !selectedRoi2.isEmpty()) {
        // 转换选中区域为图像坐标（取整避免精度问题）
        int x = static_cast<int>(qRound(selectedRoi2.x()));
        int y = static_cast<int>(qRound(selectedRoi2.y()));
        int width = static_cast<int>(qRound(selectedRoi2.width()));
        int height = static_cast<int>(qRound(selectedRoi2.height()));

        // 严格边界检查（防止超出图像范围）
        x = qMax(0, x);
        y = qMax(0, y);
        width = qMin(mat.cols - x, width);  // 宽度不超过图像右边界
        height = qMin(mat.rows - y, height);  // 高度不超过图像下边界
        width = qMax(10, width);  // 最小宽度限制（避免无效ROI）
        height = qMax(10, height);  // 最小高度限制

        // 赋值给自定义ROI结构体
        roi.x = x;
        roi.y = y;
        roi.width = width;
        roi.height = height;
        useSelectedRoi = true;
        appendLog(codec->toUnicode("使用选中区域作为ROI进行测试"));
    }
    else {
        // 无选中区域时使用默认正方形ROI计算
        int centerX = mat.cols / 2;
        int centerY = mat.rows / 2;
        int halfWidth = max((mat.cols - offset) / 2, 0);
        int halfHeight = max((mat.rows - offset) / 2, 0);
        int halfSize = min(halfWidth, halfHeight);
        halfSize = max(halfSize, 50);  // 最小半边长限制（确保有效区域）

        int x = max(centerX - halfSize, 0);
        int y = max(centerY - halfSize, 0);
        int size = min(2 * halfSize, min(mat.cols - x, mat.rows - y));

        // 赋值给自定义ROI结构体
        roi.x = x;
        roi.y = y;
        roi.width = size;
        roi.height = size;
        appendLog(codec->toUnicode("使用默认计算ROI进行测试"));
    }

    // 输出ROI详细信息
    appendLog(QString(codec->toUnicode("ROI参数：位置(%1,%2) 大小(%3×%4)"))
        .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));

    // 5. 初始化结果存储容器
    QVector<double> offsetXResults;
    QVector<double> offsetYResults;
    QVector<double> offsetRResults;
    QVector<double> similarityResults;
    QVector<double> timeResults;
    int successCount = 0;

    // 显示初始进度
    appendLog(QString(codec->toUnicode("开始重复测试，总次数：%1")).arg(testCount));

    // 6. 重复测试指定次数
    for (int i = 0; i < testCount; ++i) {
        QString outputPath = QCoreApplication::applicationDirPath() +
            QString("/match_result_%1.jpg").arg(i + 1);

        double offset_x = 0.0, offset_y = 0.0, offset_r = 0.0;
        double time_ms = 0.0;
        float similarity = 0.0f;

        // 调用匹配接口（传入自定义ROI结构体）
        bool success = RunMarkMatchingSingle(
            tempInputPath.toStdString().c_str(),
            outputPath.toStdString().c_str(),
            &offset_x, &offset_y, &offset_r,
            &similarity, &time_ms,
            &threshold,
            &roi  // 传入自定义ROI的地址（关键修改）
        );

        if (success) {
            successCount++;
            offset_r -= threshold - 6;
            // 存储成功结果
            offsetXResults.append(offset_x);
            offsetYResults.append(offset_y);
            offsetRResults.append(offset_r);
            similarityResults.append(similarity);
            timeResults.append(time_ms);

            // 记录日志
            QString logMsg = QString(codec->toUnicode("第%1次测试 | 匹配成功 | X: %2 Y: %3 R: %4° 分数: %5 耗时: %6 ms"))
                .arg(i + 1)
                .arg(offset_x, 0, 'f', 2)
                .arg(offset_y, 0, 'f', 2)
                .arg(offset_r, 0, 'f', 2)
                .arg(similarity, 0, 'f', 3)
                .arg(time_ms, 0, 'f', 2);
            appendLog(logMsg);
        }
        else {
            // 记录失败日志
            QString logMsg = QString(codec->toUnicode("第%1次测试 | 匹配失败！")).arg(i + 1);
            appendLog(logMsg);
        }

        // 立即刷新日志显示
        QApplication::processEvents();
    }

    // 7. 计算平稳性指标（仅对成功结果）
    if (successCount == 0) {
        QMessageBox::warning(this, codec->toUnicode("错误"),
            codec->toUnicode("所有测试均失败，无法计算平稳性！"));
        return;
    }

    // 定义计算3σ值的辅助函数
    auto calculateThreeSigma = [](const QVector<double>& data) -> QVector<double> {
        if (data.isEmpty()) return { 0, 0, 0 };

        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();

        double variance = 0.0;
        for (double value : data) {
            variance += std::pow(value - mean, 2);
        }
        variance /= data.size();

        double sigma = std::sqrt(variance);
        double threeSigma = 3 * sigma;

        return { mean, sigma, threeSigma };
    };

    // 计算各指标的3σ值
    QVector<double> xStats = calculateThreeSigma(offsetXResults);
    QVector<double> yStats = calculateThreeSigma(offsetYResults);
    QVector<double> rStats = calculateThreeSigma(offsetRResults);
    QVector<double> simStats = calculateThreeSigma(similarityResults);
    QVector<double> timeStats = calculateThreeSigma(timeResults);

    // 8. 输出统计结果
    appendLog(codec->toUnicode("===== 重复测试统计结果 ====="));
    appendLog(QString(codec->toUnicode("总测试次数：%1次，成功次数：%2次")).arg(testCount).arg(successCount));
    appendLog(QString(codec->toUnicode("X偏移量：均值=%1，σ=%2，3σ=%3")).arg(xStats[0], 0, 'f', 3).arg(xStats[1], 0, 'f', 3).arg(xStats[2], 0, 'f', 3));
    appendLog(QString(codec->toUnicode("Y偏移量：均值=%1，σ=%2，3σ=%3")).arg(yStats[0], 0, 'f', 3).arg(yStats[1], 0, 'f', 3).arg(yStats[2], 0, 'f', 3));
    appendLog(QString(codec->toUnicode("角度偏移：均值=%1，σ=%2，3σ=%3")).arg(rStats[0], 0, 'f', 3).arg(rStats[1], 0, 'f', 3).arg(rStats[2], 0, 'f', 3));
    appendLog(QString(codec->toUnicode("相似度分数：均值=%1，σ=%2，3σ=%3")).arg(simStats[0], 0, 'f', 3).arg(simStats[1], 0, 'f', 3).arg(simStats[2], 0, 'f', 3));
    appendLog(codec->toUnicode("==========================="));

    // 9. 弹窗显示关键统计结果
    QString statsMsg = codec->toUnicode("重复测试完成！\n\n") +
        codec->toUnicode("总测试次数：%1次\n").arg(testCount) +
        codec->toUnicode("成功次数：%1次\n\n").arg(successCount) +
        codec->toUnicode("平稳性分析结果：\n") +
        codec->toUnicode("• X偏移量：3σ = %1\n").arg(xStats[2], 0, 'f', 3) +
        codec->toUnicode("• Y偏移量：3σ = %1\n").arg(yStats[2], 0, 'f', 3) +
        codec->toUnicode("• 角度偏移：3σ = %1°\n").arg(rStats[2], 0, 'f', 3) +
        codec->toUnicode("• 相似度分数：3σ = %1\n").arg(simStats[2], 0, 'f', 3) +
        codec->toUnicode("（3σ值越小，表示系统稳定性越高）");

    QMessageBox::information(this, codec->toUnicode("重复测试统计结果"), statsMsg);
}
void MainWindow::on_loadFeatureButton_clicked()
{
    // 假设你要加载的 XML 文件路径为 "path/to/your/xml/file.xml"
    QString xmlFilePath = QString::fromLocal8Bit("D:/angletemplates/angletemplatespath/matches.xml");

    // 设置颜色为红色（你可以根据需要修改）
    QColor color = Qt::red;

    // 调用 loadAndDrawFeatureTrajectory 函数
    loadAndDrawFeatureTrajectory(xmlFilePath, color);
}
void MainWindow::drawFeatureTrajectory(const QVector<QPointF>& features, double match_x, double match_y, double angle_deg, const QColor& color)
{
    if (!scene) return;

    double anglerad = angle_deg * M_PI / 180.0; // 注意 M_PI 要 include <cmath>
    double cos_r = cos(anglerad);
    double sin_r = sin(anglerad);

    QVector<QPointF> points;
    for (const auto& feature : features) {
        double x = feature.x() * cos_r - feature.y() * sin_r + match_x+0.5;
        double y = feature.x() * sin_r + feature.y() * cos_r + match_y+0.5;
        points.append(QPointF(x, y));
    }

    QPen pen(color, 1);

    //// 连线绘制
    //for (int i = 0; i < points.size() - 1; ++i) {
    //    scene->addLine(points[i].x(), points[i].y(), points[i + 1].x(), points[i + 1].y(), pen);
    //}
    if (points.size() < 3) return; // 少于3个点无法形成轮廓

    double radius = 0.5;
    // 画点
    for (const auto& pt : points) {
        scene2->addEllipse(pt.x() - radius, pt.y() - radius,
            radius * 2, radius * 2,
            Qt::NoPen, QBrush(color));
    }
}
void MainWindow::drawFeatureTrajectory1(const QVector<QPointF>& features, double match_x, double match_y, double angle_deg, const QColor& color)
{
    if (!scene) return;

    double anglerad = angle_deg * M_PI / 180.0;
    double cos_r = cos(anglerad);
    double sin_r = sin(anglerad);

    QVector<QPointF> points;
    for (const auto& feature : features) {
        double x = feature.x() * cos_r - feature.y() * sin_r + match_x;
        double y = feature.x() * sin_r + feature.y() * cos_r + match_y;
        points.append(QPointF(x, y));
    }

    QPen pen(color, 1);

    // 连线绘制（可选）
    // for (int i = 0; i < points.size() - 1; ++i) {
    //     QGraphicsLineItem* line = scene->addLine(
    //         points[i].x(), points[i].y(), 
    //         points[i + 1].x(), points[i + 1].y(), 
    //         pen
    //     );
    //     trajectoryItems.append(line);
    // }

    if (points.size() < 3) return;

    double radius = 1;
    // 画点
    for (const auto& pt : points) {
        QGraphicsEllipseItem* ellipse = scene->addEllipse(
            pt.x() - radius, pt.y() - radius,
            radius * 2, radius * 2,
            Qt::NoPen, QBrush(color)
        );
        trajectoryItems.append(ellipse);
    }
}
void MainWindow::drawFeatureTrajectory( double match_x, double match_y, double angle_deg, const QColor& color)
{
    if (!scene) return;

    double anglerad = angle_deg * M_PI / 180.0; // 注意 M_PI 要 include <cmath>
    double cos_r = cos(anglerad);
    double sin_r = sin(anglerad);

    QVector<QPointF> points;
    sm::Core* core = sm::Core::get_init();
    for (const auto& feature : core->temp_features) {
        double x = feature.x * cos_r - feature.y * sin_r + match_x + 0.5;
        double y = feature.x * sin_r + feature.y * cos_r + match_y + 0.5;
        points.append(QPointF(x, y));
    }

    QPen pen(color, 1);

    //// 连线绘制
    //for (int i = 0; i < points.size() - 1; ++i) {
    //    scene->addLine(points[i].x(), points[i].y(), points[i + 1].x(), points[i + 1].y(), pen);
    //}
    if (points.size() < 3) return; // 少于3个点无法形成轮廓

    double radius = 0.5;
    // 画点
    for (const auto& pt : points) {
        scene2->addEllipse(pt.x() - radius, pt.y() - radius,
            radius * 2, radius * 2,
            Qt::NoPen, QBrush(color));
    }
}
// 重写事件过滤器
bool MainWindow::eventFilter(QObject* watched, QEvent* event)
{
    // 检查是否是我们关注的视口和事件类型
    if (watched == ui->graphicsView->viewport()) {
        // 检测鼠标滚轮事件（缩放操作）
        if (event->type() == QEvent::Wheel) {
            m_userScaled = true;
            m_lastTransform = ui->graphicsView->transform();
            return false; // 不拦截事件，让它继续传递
        }

        // 检测鼠标释放事件（可能是拖动或缩放后的释放）
        else if (event->type() == QEvent::MouseButtonRelease) {
            if (ui->graphicsView->transform() != m_lastTransform) {
                m_userScaled = true;
                m_lastTransform = ui->graphicsView->transform();
            }
            return false; // 不拦截事件
        }
    }

    // 其他事件交给默认处理
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);

    // 处理第一个graphicsView
    if (ui->graphicsView && pixmapItem) {
        QGraphicsScene* scene = ui->graphicsView->scene();
        if (scene && scene->items().contains(pixmapItem)) {
            QRectF itemRect = pixmapItem->boundingRect();
            if (!itemRect.isEmpty() && event->size().width() > 0 && event->size().height() > 0) {

                // 关键优化：仅在用户未手动缩放时，才自动适应窗口
                if (!m_userScaled) {
                    // 设置场景范围为项目实际大小（避免场景无限大导致计算异常）
                    scene->setSceneRect(itemRect);

                    // 调整变换锚点为项目中心，避免缩放时偏移
                    ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorViewCenter);

                    // 执行适应视图（使用item的矩形，而非item本身，更稳定）
                    ui->graphicsView->fitInView(itemRect, Qt::KeepAspectRatio);

                    // 保存初始变换（用于判断用户是否手动缩放）
                    m_lastTransform = ui->graphicsView->transform();
                }
            }
        }
    }

    // 处理第二个graphicsView（同理）
    if (ui->graphicsView_2 && pixmapItem2) {
        QGraphicsScene* scene2 = ui->graphicsView_2->scene();
        if (scene2 && scene2->items().contains(pixmapItem2)) {
            QRectF itemRect2 = pixmapItem2->boundingRect();
            if (!itemRect2.isEmpty() && event->size().width() > 0 && event->size().height() > 0) {
                scene2->setSceneRect(itemRect2);
                ui->graphicsView_2->setTransformationAnchor(QGraphicsView::AnchorViewCenter);
                ui->graphicsView_2->fitInView(itemRect2, Qt::KeepAspectRatio);
            }
        }
    }
}
// 额外：添加重置缩放的接口（比如在图像重新加载时调用）
void MainWindow::resetViewZoom()
{
    m_userScaled = false;  // 重置用户缩放状态
    if (ui->graphicsView && pixmapItem && ui->graphicsView->scene()) {
        QRectF itemRect = pixmapItem->boundingRect();
        ui->graphicsView->fitInView(itemRect, Qt::KeepAspectRatio);
        m_lastTransform = ui->graphicsView->transform();
    }
}
MainWindow::MatchData MainWindow::loadMatchesFromXml(const std::string& filePath)
{
    MatchData matchData;  // 单个 MatchData
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return matchData;
    }

    std::string line;
    bool insideMatch = false;
    bool insideFeatures = false;

    while (std::getline(file, line)) {
        // 去掉前后空格
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line == "<match>") {
            insideMatch = true;
        }
        else if (line == "</match>") {
            // 第一个 <match> 读完了，直接退出
            break;
        }
        else if (insideMatch) {
            if (line.find("<match_x>") != std::string::npos) {
                auto start = line.find(">") + 1;
                auto end = line.find("</");
                matchData.match_x = std::stod(line.substr(start, end - start));
            }
            else if (line.find("<match_y>") != std::string::npos) {
                auto start = line.find(">") + 1;
                auto end = line.find("</");
                matchData.match_y = std::stod(line.substr(start, end - start));
            }
            else if (line.find("<angle>") != std::string::npos) {
                auto start = line.find(">") + 1;
                auto end = line.find("</");
                matchData.angle_deg = std::stod(line.substr(start, end - start));
            }
            else if (line.find("<w>") != std::string::npos) {
                auto start = line.find(">") + 1;
                auto end = line.find("</");
                matchData.w = std::stoi(line.substr(start, end - start));
            }
            else if (line.find("<h>") != std::string::npos) {
                auto start = line.find(">") + 1;
                auto end = line.find("</");
                matchData.h = std::stoi(line.substr(start, end - start));
            }
            else if (line == "<features>") {
                insideFeatures = true;
            }
            else if (line == "</features>") {
                insideFeatures = false;
            }
            else if (insideFeatures && line.find("<feature") != std::string::npos) {
                // 解析 feature 的属性
                double x = 0.0, y = 0.0;
                auto xPos = line.find("x=\"");
                if (xPos != std::string::npos) {
                    xPos += 3;
                    auto xEnd = line.find("\"", xPos);
                    x = std::stod(line.substr(xPos, xEnd - xPos));
                }
                auto yPos = line.find("y=\"");
                if (yPos != std::string::npos) {
                    yPos += 3;
                    auto yEnd = line.find("\"", yPos);
                    y = std::stod(line.substr(yPos, yEnd - yPos));
                }
                matchData.features.push_back({ x, y });
            }
        }
    }

    file.close();
    return matchData; // 返回单个
}

void MainWindow::loadAndDrawFeatureTrajectory(const QString& xmlFilePath, const QColor& color)
{
    std::string path = "D:/mark_window/module/qtgui/src/matches.xml";
    MatchData matches = loadMatchesFromXml(path);

    // 调用原有的绘制方法
    //drawFeatureTrajectory(matches.features, matches.match_x, matches.match_y, matches.angle_deg, color);
}
void MainWindow::wheelEvent(QWheelEvent* event)
{
    const double zoomStep = 1.15;

    QPoint globalPos = QPoint(event->globalPos().x(), event->globalPos().y());  // 全局位置（Qt6 用 globalPosition；Qt5 用 globalPos）

    // 将全局坐标映射到 graphicsView 和 graphicsView_2 的局部坐标
    QPoint posInView1 = ui->graphicsView->mapFromGlobal(globalPos);
    QPoint posInView2 = ui->graphicsView_2->mapFromGlobal(globalPos);

    if (ui->graphicsView->rect().contains(posInView1)) {
        if (event->angleDelta().y() > 0)
            updateZoom(ui->graphicsView, scaleFactor1, zoomStep);
        else
            updateZoom(ui->graphicsView, scaleFactor1, 1.0 / zoomStep);
    }
    else if (ui->graphicsView_2->rect().contains(posInView2)) {
        if (event->angleDelta().y() > 0)
            updateZoom(ui->graphicsView_2, scaleFactor2, zoomStep);
        else
            updateZoom(ui->graphicsView_2, scaleFactor2, 1.0 / zoomStep);
    }
}



void MainWindow::updateZoom(QGraphicsView* view, double& factor, double scale)
{
    factor *= scale;
    view->scale(scale, scale);
}


void MainWindow::mousePressEvent(QMouseEvent* event)
{
    QPointF scenePos = ui->graphicsView->mapToScene(event->pos());
    //qDebug() << "点击位置：" << scenePos;
    QMainWindow::mousePressEvent(event);
}

void MainWindow::on_pushButton_detailsDontCare_clicked()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    // 检查是否已加载图像
    if (originalPixmap.isNull()) {
        QMessageBox::warning(this, codec->toUnicode("提示"),
            codec->toUnicode("请先加载模板图像！"));
        return;
    }

    // 创建对话框
    DetailsDialog dialog(this);

    // 设置当前图像
    dialog.setPixmap(originalPixmap);

    // 如果已有忽略区域，传递给对话框
    if (!detailsIgnoreRegions.isEmpty()) {
        dialog.setSelectedRegions(detailsIgnoreRegions);
    }

    // 显示对话框
    if (dialog.exec() == QDialog::Accepted) {
        // 用户点击了确定，保存忽略区域
        detailsIgnoreRegions = dialog.getSelectedRegions();

        appendLog(QString(codec->toUnicode("已设置 %1 个细节忽略区域"))
            .arg(detailsIgnoreRegions.size()));

        // 可选：在主视图上也显示这些区域
        //drawIgnoreRegionsOnMainView();
    }
    else {
        // 用户点击了取消
        appendLog(codec->toUnicode("取消设置细节忽略区域"));
    }
}

void MainWindow::on_Camera_test()
{
    //sm::CameraManager::GetInstance().UseCameraDemo();
    /*sm::CVisionInterface::Ins().Init();
    cv::Mat img;
    sm::CVisionInterface::Ins().CameraCapture(img);

    QPixmap qpimage = QPixmap::fromImage(cvMat2QImage(img));

    if (!scene2) {
        scene2 = new QGraphicsScene(this);
        ui->graphicsView_2->setScene(scene2);
        ui->graphicsView_2->setDragMode(QGraphicsView::ScrollHandDrag);
        ui->graphicsView_2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    }
    else {
        scene2->clear();
    }

    pixmapItem2 = scene2->addPixmap(qpimage);
    scene2->setSceneRect(qpimage.rect());

    ui->graphicsView_2->resetTransform();
    ui->graphicsView_2->fitInView(pixmapItem2, Qt::KeepAspectRatio);*/
    QTextCodec* codec = QTextCodec::codecForName("GBK");//添加编码格式
    if (!m_timer) {
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &MainWindow::updateFrame);
    }

    //if (!m_timer->isActive()) {
    //    // 确保相机只初始化一次（移到构造函数或其他地方调用一次即可）
    //    // sm::CVisionInterface::Ins().Init();   // 建议放到 MainWindow 构造函数中调用一次
    //    m_timer->start(33);   // 约30帧/秒 (1000/33 ≈ 30)
    //    ui->pushButton_7->setText(codec->toUnicode("停止实时显示"));
    //}
    //else {
    //    m_timer->stop();
    //    ui->pushButton_7->setText(codec->toUnicode("开始实时显示"));
    //}
    if (m_timer->isActive()) {
        // 停止实时显示
        m_timer->stop();
        ui->pushButton_7->setText(codec->toUnicode("开始实时显示"));
        // 注意：停止定时器后，相机仍然处于抓图状态，但不再采集帧。
        // 如果希望停止相机抓图以释放资源，可以调用 StopCapture()，但这样下次启动时需要重新 StartCapture。
        // 为了下次能快速启动，建议不停止相机抓图，仅停止定时器。
        // 如果之前在线测试时调用了 StopCapture()，那么下次启动前必须调用 StartCapture。
    }
    else {
        // 开始实时显示
        // 确保相机处于抓图状态（如果之前被 StopCapture 停止了，需要重新启动）
        sm::CVisionInterface::Ins().StartCapture();   // 关键：确保相机开始抓图

        m_timer->start(33);
        ui->pushButton_7->setText(codec->toUnicode("停止实时显示"));
    }
}
void MainWindow::updateFrame()
{
    cv::Mat img;
    if (sm::CVisionInterface::Ins().CameraCapture(img) != APIErrCode::SUCCESS) {
        // 采集失败可做简单处理，比如返回
        return;
    }

    QPixmap qpimage = QPixmap::fromImage(cvMat2QImage(img));

    if (!scene2) {
        scene2 = new QGraphicsScene(this);
        ui->graphicsView_2->setScene(scene2);
        ui->graphicsView_2->setDragMode(QGraphicsView::ScrollHandDrag);
        ui->graphicsView_2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    }
    else {
        scene2->clear();
    }

    pixmapItem2 = scene2->addPixmap(qpimage);
    scene2->setSceneRect(qpimage.rect());

    ui->graphicsView_2->resetTransform();
    ui->graphicsView_2->fitInView(pixmapItem2, Qt::KeepAspectRatio);
}
//相机初始化
void MainWindow::on_OpenCamera_test()
{
    sm::CVisionInterface::Ins().Init();
}

void MainWindow::on_CloseCamera_test()
{
    sm::CVisionInterface::Ins().CloseCamera();
}

void MainWindow::SearchAndConnectCamera()
{
    CTlFactory& tlFactory = CTlFactory::GetInstance();
    DeviceInfoList_t devices;
    tlFactory.EnumerateDevices(devices); // 枚举设备

    if (devices.empty()) {
        throw RUNTIME_EXCEPTION("No camera found.");
    }

    CInstantCamera camera(tlFactory.CreateDevice(devices[0]));
    camera.Open(); // 连接首台相机
}
