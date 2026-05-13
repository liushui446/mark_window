#ifndef __CORE_1_H__
#define __CORE_1_H__

#include <vector>

// 导出宏定义
#ifdef _WIN32
#ifdef CORE_EXPORTS
#define SM_EXPORTS __declspec(dllexport)
#else
#define SM_EXPORTS __declspec(dllimport)
#endif
#else
#define SM_EXPORTS __attribute__((visibility("default")))
#endif

struct Point_f {
    double x;
    double y;
};

struct IgnoreRect {
    double x;
    double y;
    double width;
    double height;
};

namespace sm {
    class SM_EXPORTS Core {
    private:
        Core();  // 私有构造函数

    public:
        int timeout;
        std::vector<Point_f> sub_features;
        std::vector<Point_f> temp_features;
        std::vector<Point_f> features;
        std::vector<IgnoreRect> ignore_regions;

        // 获取单例实例
        static Core* get_init();

        // 销毁单例（可选）
        static void destroy();
    };
}

// 全局函数声明
SM_EXPORTS void print();

#endif // __CORE_1_H__