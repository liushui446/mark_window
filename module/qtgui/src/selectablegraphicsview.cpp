#include "selectablegraphicsview.h"
#include <QPen>
#include <QBrush>
#include <QGraphicsScene>
#include <QContextMenuEvent>
#include <QAction>
#include <QMessageBox>
#include <QTextCodec> 

SelectableGraphicsView::SelectableGraphicsView(QWidget* parent)
    : QGraphicsView(parent),
    isSelecting(false),
    isMoving(false),
    selectionRect(nullptr),
    contextMenu(nullptr)
{
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    createContextMenu();
}

SelectableGraphicsView::~SelectableGraphicsView()
{
    clearSelection();
    delete contextMenu;
}

QRectF SelectableGraphicsView::getSelectedRect() const
{
    return selectedRect;
}

void SelectableGraphicsView::clearSelection()
{
    if (selectionRect) {
        if (scene() ) {
            scene()->removeItem(selectionRect);
        }
        delete selectionRect;
        selectionRect = nullptr;
    }
    selectedRect = QRectF();
    isSelecting = false;
    isMoving = false;
}

void SelectableGraphicsView::createContextMenu()
{
    // 保持原有菜单创建逻辑不变
    contextMenu = new QMenu(this);
    QTextCodec* codec = QTextCodec::codecForName("GBK");

    QAction* saveAction = new QAction(codec->toUnicode("保存选中区域"), this);
    QAction* clearAction = new QAction(codec->toUnicode("清除选中区域"), this);
    QAction* measureAction = new QAction(codec->toUnicode("测量区域大小"), this);
    QAction* binaryAction = new QAction(codec->toUnicode("二值化区域"), this);
    QAction* cancelBinaryAction = new QAction(codec->toUnicode("取消二值化"), this);

    connect(saveAction, &QAction::triggered, this, &SelectableGraphicsView::onSaveRegionAction);
    connect(clearAction, &QAction::triggered, this, &SelectableGraphicsView::onClearRegionAction);
    connect(measureAction, &QAction::triggered, this, &SelectableGraphicsView::onMeasureRegionAction);
    connect(binaryAction, &QAction::triggered, this, &SelectableGraphicsView::onBinaryAction);
    connect(cancelBinaryAction, &QAction::triggered, this, &SelectableGraphicsView::onCancelBinaryAction);

    contextMenu->addAction(saveAction);
    contextMenu->addAction(clearAction);
    contextMenu->addSeparator();
    contextMenu->addAction(measureAction);
    contextMenu->addSeparator();
    contextMenu->addAction(binaryAction);
    contextMenu->addAction(cancelBinaryAction);
}

// 修复1：正确标记移动状态，确保点击矩形内时激活isMoving
void SelectableGraphicsView::mousePressEvent(QMouseEvent* event)
{
    if (!scene()) {
        QGraphicsView::mousePressEvent(event);
        return;
    }

    if (event->button() == Qt::LeftButton) {
        // 绘制新矩形前清理旧矩形
        if (selectionRect != nullptr) {
            clearSelection();
        }

        // 检查是否有已存在的矩形（用于移动）
        bool isRectValid = (selectionRect != nullptr)
            && (selectionRect->scene() == scene())
            && selectionRect->isVisible()
            && !selectionRect->rect().isEmpty();

        // 处理移动逻辑
        if (isRectValid) {
            QPoint viewPos = event->pos();
            if (viewPos.x() < 0 || viewPos.x() >= width() ||
                viewPos.y() < 0 || viewPos.y() >= height()) {
                QGraphicsView::mousePressEvent(event);
                return;
            }

            QPointF scenePos = mapToScene(viewPos);
            // 转换场景坐标到矩形本地坐标（关键：判断是否点击在矩形内）
            QPointF localPos = selectionRect->mapFromScene(scenePos);
            if (selectionRect->rect().contains(localPos)) {
                isMoving = true;
                moveStartPos = scenePos;
                // 计算鼠标在矩形内的偏移（本地坐标）
                rectLocalOffset = localPos;  // 记录鼠标在矩形内的相对位置
                return;
            }
            else {
                clearSelection();
                isRectValid = false;
            }
        }

        // 框选新矩形
        if (!isRectValid && dragMode() == QGraphicsView::ScrollHandDrag) {
            isSelecting = true;
            selectStartPoint = event->pos();

            selectionRect = new QGraphicsRectItem();
            QPen pen(Qt::red);
            pen.setStyle(Qt::DashLine);
            selectionRect->setPen(pen);
            selectionRect->setBrush(QBrush(QColor(255, 0, 0, 50)));
            selectionRect->setVisible(true);

            if (scene()) {
                scene()->addItem(selectionRect);
                if (!scene()->items().contains(selectionRect)) {
                    delete selectionRect;
                    selectionRect = nullptr;
                    isSelecting = false;
                }
            }
            else {
                delete selectionRect;
                selectionRect = nullptr;
                isSelecting = false;
            }
            return;
        }
    }

    QGraphicsView::mousePressEvent(event);
}

// 修复2：正确计算移动后的位置，基于场景坐标更新矩形
void SelectableGraphicsView::mouseMoveEvent(QMouseEvent* event)
{
    // 处理矩形移动（核心修复）
    if (isMoving && selectionRect) {
        QPointF currentScenePos = mapToScene(event->pos());  // 当前鼠标场景坐标
        // 计算新的矩形基准点（pos()）：当前鼠标位置 - 鼠标在矩形内的偏移
        QPointF newPos = currentScenePos - rectLocalOffset;

        // 保持矩形大小不变，只更新位置（通过pos()调整，rect()本地坐标不变）
        selectionRect->setPos(newPos);
        // 更新保存的场景矩形（pos() + rect() 才是完整场景坐标）
        selectedRect = QRectF(
            selectionRect->pos() + selectionRect->rect().topLeft(),
            selectionRect->rect().size()
        );
        return;
    }

    // 处理框选更新
    if (isSelecting && selectionRect) {
        QPoint currentPoint = event->pos();
        int x = qMin(selectStartPoint.x(), currentPoint.x());
        int y = qMin(selectStartPoint.y(), currentPoint.y());
        int width = qAbs(currentPoint.x() - selectStartPoint.x());
        int height = qAbs(currentPoint.y() - selectStartPoint.y());

        QRect viewRect(x, y, width, height);
        QRectF sceneRect = mapToScene(viewRect).boundingRect();
        // 框选时，rect() 是本地坐标，pos() 固定为场景原点（0,0）
        selectionRect->setPos(0, 0);  // 基准点固定
        selectionRect->setRect(sceneRect);  // 本地坐标等于场景坐标
        selectedRect = sceneRect;  // 直接保存场景矩形
        return;
    }

    QGraphicsView::mouseMoveEvent(event);
}

// 修复3：释放鼠标时确认移动状态
void SelectableGraphicsView::mouseReleaseEvent(QMouseEvent* event)
{
    if (isMoving && event->button() == Qt::LeftButton) {
        isMoving = false;
        emit rectMoved(selectedRect);  // 发射移动结束信号
        return;
    }

    if (isSelecting && event->button() == Qt::LeftButton) {
        isSelecting = false;
        if (selectionRect) {
            selectedRect = QRectF(
                selectionRect->pos() + selectionRect->rect().topLeft(),
                selectionRect->rect().size()
            );
            if (selectedRect.width() > 0 && selectedRect.height() > 0) {
                emit rectSelected(selectedRect);
            }
            else {
                clearSelection();
            }
        }
        return;
    }

    QGraphicsView::mouseReleaseEvent(event);
}

void SelectableGraphicsView::contextMenuEvent(QContextMenuEvent* event)
{
    if (selectionRect && selectedRect.isValid()) {
        contextMenu->exec(event->globalPos());
    }
    else {
        QGraphicsView::contextMenuEvent(event);
    }
}

void SelectableGraphicsView::onSaveRegionAction()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    if (selectedRect.isValid()) {
        emit menuActionTriggered(codec->toUnicode("save_region"));
        clearSelection();
        QMessageBox::information(
            this,
            codec->toUnicode("提示"),
            codec->toUnicode("保存选中区域成功！")
        );
    }
    else {
        QMessageBox::warning(
            this,
            codec->toUnicode("警告"),
            codec->toUnicode("无有效区域可保存！")
        );
    }
}

// 其他菜单函数保持不变...
void SelectableGraphicsView::onClearRegionAction()
{
    QTextCodec* codec = QTextCodec::codecForName("GBK");
    clearSelection();
    emit menuActionTriggered(codec->toUnicode("clear_region"));
    QMessageBox::information(
        this,
        codec->toUnicode("提示"),
        codec->toUnicode("选中区域已清除")
    );
}

void SelectableGraphicsView::onMeasureRegionAction()
{
    if (selectedRect.isValid()) {
        QTextCodec* codec = QTextCodec::codecForName("GBK");
        QString info = QString(
            codec->toUnicode("区域大小：宽=%.2f, 高=%.2f")
        ).arg(selectedRect.width()).arg(selectedRect.height());
        emit menuActionTriggered(codec->toUnicode("measure_region:") + info);
        QMessageBox::information(this, codec->toUnicode("区域测量"), info);
    }
}

void SelectableGraphicsView::onBinaryAction()
{
    if (selectionRect && selectedRect.isValid()) {
        QTextCodec* codec = QTextCodec::codecForName("GBK");
        emit binaryActionTriggered(true);
        emit menuActionTriggered(codec->toUnicode("binary_region"));
        QMessageBox::information(
            this,
            codec->toUnicode("提示"),
            codec->toUnicode("已对选中区域应用二值化")
        );
    }
}

void SelectableGraphicsView::onCancelBinaryAction()
{
    if (selectionRect && selectedRect.isValid()) {
        QTextCodec* codec = QTextCodec::codecForName("GBK");
        emit binaryActionTriggered(false);
        emit menuActionTriggered(codec->toUnicode("cancel_binary"));
        QMessageBox::information(
            this,
            codec->toUnicode("提示"),
            codec->toUnicode("已取消选中区域的二值化")
        );
    }
}