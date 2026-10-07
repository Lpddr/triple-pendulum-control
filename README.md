# 基于强化学习的三级倒立摆控制系统

基于 MuJoCo、Gymnasium 与 PyTorch 构建小车三级倒立摆仿真系统，采用轨迹优化与 CEM 策略搜索，仅通过小车水平驱动力完成自主摆起、稳定直立及指定外力扰动后的恢复。

在自然下垂附近随机初态的 100 局独立仿真测试中全部通过，连续直立超过 10 秒。

本工程基于 [Zac Westbrook 的原项目](https://github.com/zw22x/triple-pendulum-rl) 扩展，运行方法见[中文运行指南](RUN_GUIDE_ZH.md)，算法及验证范围见[技术说明](docs/HR_DELIVERY.md)。
