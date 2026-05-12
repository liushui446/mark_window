#ifndef DETAILSDIALOG_H
#define DETAILSDIALOG_H

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QVector>
#include <QRectF>

namespace Ui {
    class DetailsDialog;
}

class DetailsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DetailsDialog(QWidget* parent = nullptr);
    ~DetailsDialog();

    // 设置要显示的图像
    void setPixmap(const QPixmap& pixmap);

    // 获取所有选中的ROI区域
    QVector<QRectF> getSelectedRegions() const { return selectedRegions; }

    // 设置已有的ROI区域（用于编辑）
    void setSelectedRegions(const QVector<QRectF>& regions);

private slots:
    void onRectSelected(const QRectF& rect);
    void onMenuAction(const QString& action);
    void on_pushButton_clearAll_clicked();
    void on_pushButton_confirm_clicked();
    void on_pushButton_cancel_clicked();

protected:
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    Ui::DetailsDialog* ui;
    QGraphicsScene* scene;
    QGraphicsPixmapItem* pixmapItem;
    double scaleFactor;

    QVector<QRectF> selectedRegions;  // 存储所有选中的ROI区域

    void updateZoom(double scale);
    void drawAllRegions();  // 在视图上绘制所有ROI区域
    void appendLog(const QString& msg);
};

#endif // DETAILSDIALOG_H