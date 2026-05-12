#include "detailsdialog.h"
#include "ui_detailsdialog.h"
#include "selectablegraphicsview.h"
#include <QWheelEvent>
#include <QTextCodec>
#include <QTime>
#include <QPen>
#include <QBrush>
#include <QTimer>

DetailsDialog::DetailsDialog(QWidget* parent) :
    QDialog(parent),
    ui(new Ui::DetailsDialog),
    scene(nullptr),
    pixmapItem(nullptr),
    scaleFactor(1.0)
{
    ui->setupUi(this);

    // 设置窗口标题和大小
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    setWindowTitle(codec->toUnicode("细节忽略区域设置"));
    resize(800, 600);

    // 初始化场景
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setRenderHint(QPainter::Antialiasing);
    ui->graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    ui->graphicsView->setMouseTracking(true);

    // 连接信号槽
    connect(ui->graphicsView, &SelectableGraphicsView::rectSelected,
        this, &DetailsDialog::onRectSelected);
    connect(ui->graphicsView, &SelectableGraphicsView::menuActionTriggered,
        this, &DetailsDialog::onMenuAction);
}

DetailsDialog::~DetailsDialog()
{
    delete ui;
}

void DetailsDialog::setPixmap(const QPixmap& pixmap)
{
    if (pixmap.isNull()) {
        return;
    }

    scene->clear();
    selectedRegions.clear();

    pixmapItem = scene->addPixmap(pixmap);
    scene->setSceneRect(pixmap.rect());

    scaleFactor = 1.0;
    ui->graphicsView->resetTransform();

    // 使用 QTimer 延迟执行 fitInView，确保窗口已完全显示
    QTimer::singleShot(0, this, [this]() {
        if (pixmapItem) {
            ui->graphicsView->fitInView(pixmapItem, Qt::KeepAspectRatio);
        }
        });
}

void DetailsDialog::setSelectedRegions(const QVector<QRectF>& regions)
{
    selectedRegions = regions;
    drawAllRegions();
}

void DetailsDialog::onRectSelected(const QRectF& rect)
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    QString log = QString(codec->toUnicode("选中区域: x=%1, y=%2, 宽=%3, 高=%4"))
        .arg(rect.x()).arg(rect.y())
        .arg(rect.width()).arg(rect.height());

    appendLog(log);
}

void DetailsDialog::onMenuAction(const QString& action)
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    if (action == QString(codec->toUnicode("save_region"))) {
        QRectF rect = ui->graphicsView->getSelectedRect();
        if (!rect.isEmpty()) {
            selectedRegions.append(rect);
            // 先通知 graphicsView 清除当前选择的临时矩形
            ui->graphicsView->clearSelection();
            // 然后重绘所有保存的区域
            drawAllRegions();
        }
    }
    else if (action == QString(codec->toUnicode("clear_region"))) {
        // 清除当前选中的区域（需要找到最近添加的）
        if (!selectedRegions.isEmpty()) {
            selectedRegions.removeLast();
            drawAllRegions();
        }
    }
}

void DetailsDialog::on_pushButton_clearAll_clicked()
{
    selectedRegions.clear();
    ui->graphicsView->clearSelection();
    drawAllRegions();
}

void DetailsDialog::on_pushButton_confirm_clicked()
{
    // 确保只执行一次
    if (result() == QDialog::Accepted) {
        return;
    }
    accept();  // 关闭对话框并返回 QDialog::Accepted
}

void DetailsDialog::on_pushButton_cancel_clicked()
{
    // 确保只执行一次
    if (result() == QDialog::Rejected) {
        return;
    }
    reject();  // 关闭对话框并返回 QDialog::Rejected
}

void DetailsDialog::drawAllRegions()
{
    if (!scene) return;

    // 清除旧的ROI绘制（保留原图）
    QList<QGraphicsItem*> items = scene->items();
    for (QGraphicsItem* item : items) {
        if (item != pixmapItem) {
            scene->removeItem(item);
            delete item;
        }
    }

    // 绘制所有ROI区域
    QPen pen(Qt::yellow, 2, Qt::DashLine);
    QBrush brush(QColor(255, 255, 0, 50));  // 半透明黄色填充

    for (int i = 0; i < selectedRegions.size(); ++i) {
        const QRectF& rect = selectedRegions[i];

        // 绘制矩形框
        QGraphicsRectItem* rectItem = scene->addRect(rect, pen, brush);

        // 添加序号标签
        QGraphicsTextItem* textItem = scene->addText(QString::number(i + 1));
        textItem->setDefaultTextColor(Qt::yellow);
        textItem->setPos(rect.topLeft());

        // 设置字体大小
        QFont font = textItem->font();
        font.setPointSize(12);
        font.setBold(true);
        textItem->setFont(font);
    }
}

void DetailsDialog::appendLog(const QString& msg)
{
    QString time = QTime::currentTime().toString("HH:mm:ss");
    QString logEntry = QString("[%1] %2").arg(time, msg);

    ui->textEdit_log->append(logEntry);
    ui->textEdit_log->ensureCursorVisible();
}

void DetailsDialog::wheelEvent(QWheelEvent* event)
{
    const double zoomStep = 1.15;

    QPoint globalPos = event->globalPos();
    QPoint posInView = ui->graphicsView->mapFromGlobal(globalPos);

    if (ui->graphicsView->rect().contains(posInView)) {
        if (event->angleDelta().y() > 0) {
            updateZoom(zoomStep);
        }
        else {
            updateZoom(1.0 / zoomStep);
        }
    }
}

void DetailsDialog::updateZoom(double scale)
{
    scaleFactor *= scale;
    ui->graphicsView->scale(scale, scale);
}

void DetailsDialog::resizeEvent(QResizeEvent* event)
{
    QDialog::resizeEvent(event);

    if (ui->graphicsView && pixmapItem) {
        QGraphicsScene* currentScene = ui->graphicsView->scene();
        if (currentScene && currentScene->items().contains(pixmapItem)) {
            QRectF itemRect = pixmapItem->boundingRect();
            if (!itemRect.isEmpty()) {
                currentScene->setSceneRect(itemRect);
                ui->graphicsView->setTransformationAnchor(QGraphicsView::AnchorViewCenter);
                ui->graphicsView->fitInView(itemRect, Qt::KeepAspectRatio);
            }
        }
    }
}