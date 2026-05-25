#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QVector>
#include "ImageWindow.h"
#include <QStringListModel>
#include <QStringList>
#include "detailsdialog.h"
#include <QTimer> 
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
    struct FeaturePoint {
        double x;
        double y;
    };

    struct MatchData {
        double match_x = 0.0;
        double match_y = 0.0;
        double angle_deg = 0.0;
        int w = 0;
        int h = 0;
        std::vector<FeaturePoint> features;
    };

    void SearchAndConnectCamera();
   
private slots:
    void on_actionw_triggered();      // 打开文件操作槽函数
    void on_pushButton_3_clicked();
    void on_pushButton_clicked();  // ← 添加这行
    void on_pushButton_5_clicked();
    void on_pushButton_2_clicked();
    void on_pushButton_8_clicked();
    void on_pushButton_4_clicked();
    void on_pushButton_9_clicked();
    void on_pushButton_10_clicked();
    void on_pushButton_11_clicked();
    void onFirstViewRectSelected(const QRectF& rect);
    void onSecondViewRectSelected(const QRectF& rect);
    void onFirstViewMenuAction(const QString& action);  // 第一个视图的菜单动作
    void onSecondViewMenuAction(const QString& action); // 第二个视图的菜单动作

    void on_Camera_test();
    void on_OpenCamera_test();//打开相机
    void on_CloseCamera_test();//关闭相机
    void updateFrame();   // 定时采集并刷新画面

    void on_pushButton_detailsDontCare_clicked();  // 细节忽略区域按钮

    void on_spinBox_exposure_valueChanged(double val);
    void on_spinBox_gain_valueChanged(double val);
private:
    Ui::MainWindow* ui;
    QGraphicsScene* scene = nullptr;
    QGraphicsPixmapItem* pixmapItem = nullptr;
    double scaleFactor1 = 1.0;  // 用于 graphicsView
    double scaleFactor2 = 1.0;  // 用于 graphicsView_2
    bool onlinetest = 0;//在线测试
    ImageWindow* imageWindow = nullptr;  // 子窗口指针

    QGraphicsScene* scene2 = nullptr;
    QGraphicsPixmapItem* pixmapItem2 = nullptr;
    QPixmap originalPixmap2;
    QPixmap binaryPixmap2;
    bool showingBinary2 = false;
    QString filePath;
    QString filePath_orgin;
    QString testFolderPath;                       // 连续测试文件夹路径
    QStringList testImageFiles;                   // 存储图像文件路径列表

    void loadImage(const QString& path);
    void updateZoom(QGraphicsView* view, double& factor, double scale);
    void loadImageToSecondView(const QString& path);
    void resetViewZoom();
    bool eventFilter(QObject* watched, QEvent* event);
    //QVector<FeaturePoint> loadFeaturesFromXml(const std::string& xmlFilePath);
   MatchData loadMatchesFromXml(const std::string& filePath);
   void drawFeatureTrajectory1(const QVector<QPointF>& features, double match_x, double match_y, double angle_deg, const QColor& color = Qt::red);
    void drawFeatureTrajectory(const QVector<QPointF>& features, double match_x, double match_y, double angle_deg, const QColor& color = Qt::red);
    //获取全局的点集
    void MainWindow::drawFeatureTrajectory(double match_x, double match_y, double angle_deg, const QColor& color);
    QVector<QColor> colorTable = { Qt::red, Qt::green, Qt::blue, Qt::yellow, Qt::magenta };
    QPixmap originalPixmap;  // 原图
    QPixmap binaryPixmap;    // 二值图
    bool showingBinary = false; // 当前是否显示的是二值图
    QPixmap generateBinaryPixmap(const QPixmap& pixmap);  // ← 添加这一行

    QStringListModel* logModel;
    QStringList logList;
    void appendLog(const QString& msg); // 声明日志函数
    bool hasSelectedRoi;       // 标记是否有选中的ROI区域
    QRectF selectedRoi;        // 存储选中的ROI区域（场景坐标）
     // 新增：第二个视图的ROI
    bool hasSelectedRoi2;       // 第二个视图是否有选中区域
    QRectF selectedRoi2;        // 第二个视图的选中区域
    QList<QGraphicsItem*> trajectoryItems; // 存储所有轨迹相关的图形项
    bool modelInitialized;  // 添加这个声明  模板初始化
    bool m_userScaled;  // 记录用户是否手动缩放过视图
    QTransform m_lastTransform;  // 保存最后一次用户变换

    QVector<QRectF> detailsIgnoreRegions;  // 存储细节忽略区域
    int convertMarkTypeToInt(const QString& markTypeStr);//标记类型

    QTimer* m_timer= nullptr;
protected:
    void wheelEvent(QWheelEvent* event) override; // 确保声明了 wheelEvent
    void mousePressEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

};

#endif // MAINWINDOW_H
