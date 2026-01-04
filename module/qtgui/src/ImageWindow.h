#ifndef IMAGEWINDOW_H
#define IMAGEWINDOW_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>

class ImageWindow : public QWidget
{
    Q_OBJECT
public:
    explicit ImageWindow(QWidget* parent = nullptr);
    void loadImage(const QString& path);

private:
    QGraphicsView* view;
    QGraphicsScene* scene;
    QGraphicsPixmapItem* pixmapItem;
};

#endif // IMAGEWINDOW_H
