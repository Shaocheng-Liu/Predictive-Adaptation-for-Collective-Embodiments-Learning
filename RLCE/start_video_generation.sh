#!/bin/bash

# 快速启动脚本 - 选择使用哪个版本

echo ""
echo "========================================="
echo "  🎬 视频批量生成工具"
echo "========================================="
echo ""
echo "请选择生成方式："
echo ""
echo "1) 🟢 基础版 (顺序生成，预计 3-7.5 小时)"
echo "2) 🔵 快速版 (并行x4，预计 45分钟-2小时)"
echo "3) 🟡 断点续传 (只生成缺失，自动跳过已有)"
echo "4) 🟣 自定义 (手动指定参数)"
echo "5) ❌ 退出"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "启动 ✓ 基础版 (顺序生成)"
        echo "机械臂: 9个 | 任务: 10个 | 总数: 90个视频"
        read -p "确认启动？(y/n): " confirm
        if [ "$confirm" = "y" ]; then
            bash generate_all_videos.sh
        fi
        ;;
    2)
        echo ""
        echo "启动 ✓ 快速版 (并行x4)"
        read -p "确认启动？(y/n): " confirm
        if [ "$confirm" = "y" ]; then
            bash generate_all_videos_advanced.sh --parallel 4
        fi
        ;;
    3)
        echo ""
        echo "启动 ✓ 断点续传模式"
        echo "将跳过已生成的视频，只生成缺失部分"
        read -p "确认启动？(y/n): " confirm
        if [ "$confirm" = "y" ]; then
            bash generate_all_videos_advanced.sh --skip-existing
        fi
        ;;
    4)
        echo ""
        echo "自定义参数"
        read -p "并行数 (默认1): " parallel
        parallel=${parallel:-1}
        read -p "跳过已有? (y/n, 默认n): " skip
        
        args="--parallel $parallel"
        [ "$skip" = "y" ] && args="$args --skip-existing"
        
        echo ""
        echo "启动 ✓ 自定义版"
        echo "参数: $args"
        read -p "确认启动？(y/n): " confirm
        if [ "$confirm" = "y" ]; then
            bash generate_all_videos_advanced.sh $args
        fi
        ;;
    5)
        echo "退出"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "完成！视频保存在: outputs/*/video/"
echo "==========================================="
