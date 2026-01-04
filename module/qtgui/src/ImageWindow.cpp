#include "ImageWindow.h"
#include <QVBoxLayout>
#include <QPixmap>
#include <QDebug>

ImageWindow::ImageWindow(QWidget* parent)
    : QWidget(nullptr),  // 注意：不传 parent，表示这是独立窗口
    view(new QGraphicsView(this)),
    scene(new QGraphicsScene(this)),
    pixmapItem(nullptr)
{
    setWindowTitle("图像窗口");
    resize(800, 600);

    view->setScene(scene);
    view->setRenderHint(QPainter::Antialiasing);
    view->setDragMode(QGraphicsView::ScrollHandDrag);
    view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(view);
    setLayout(layout);
}

void ImageWindow::loadImage(const QString& path)
{
    QPixmap pixmap(path);
    if (pixmap.isNull()) {
        qDebug() << "加载图像失败：" << path;
        return;
    }

    scene->clear();
    pixmapItem = scene->addPixmap(pixmap);
    scene->setSceneRect(pixmap.rect());
    view->fitInView(pixmapItem, Qt::KeepAspectRatio);
}
