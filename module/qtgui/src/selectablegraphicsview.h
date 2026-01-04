#ifndef SELECTABLEGRAPHICSVIEW_H
#define SELECTABLEGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <QMouseEvent>
#include <QMenu>
#include <QAction>

class SelectableGraphicsView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit SelectableGraphicsView(QWidget* parent = nullptr);
    ~SelectableGraphicsView() override;

    QRectF getSelectedRect() const;
    void clearSelection();  // 清除选中矩形

signals:
    void rectSelected(const QRectF& rect);       // 矩形选中信号
    void rectMoved(const QRectF& newRect);       // 矩形移动信号
    void menuActionTriggered(const QString& action); // 菜单动作信号
    void binaryActionTriggered(bool enable);     // 二值化动作信号

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override; // 右键菜单事件

private slots:
    // 菜单动作槽函数
    void onSaveRegionAction();        // 保存区域
    void onClearRegionAction();       // 清除区域
    void onMeasureRegionAction();     // 测量区域
    void onBinaryAction();            // 二值化区域
    void onCancelBinaryAction();      // 取消二值化

private:
    bool isSelecting;                 // 框选状态
    bool isMoving;                    // 移动状态
    QPoint selectStartPoint;          // 框选起始点（视图坐标）
    QPointF moveStartPos;             // 移动起始点（场景坐标）
    QPointF rectStartOffset;          // 鼠标与矩形左上角偏移量
    QGraphicsRectItem* selectionRect; // 框选矩形项
    QRectF selectedRect;              // 选中的矩形区域（场景坐标）
    QMenu* contextMenu;               // 右键菜单

    void createContextMenu();         // 创建右键菜单
    // 添加新变量：记录鼠标在矩形内的本地偏移
    QPointF rectLocalOffset;
};

#endif // SELECTABLEGRAPHICSVIEW_H