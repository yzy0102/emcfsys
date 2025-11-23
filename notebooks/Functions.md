# 功能开发

1. 支持训练Unet等分割模型，用于分割电镜数据集

   1. Loss指标
      1. [X] 支持交叉熵 CrossEntropy
      2. [X] 支持Dice loss
      3. [ ] 支持多指标加权
      4. [ ] Focal loss
   2. Metric指标
      1. [X] IoU Acc F1Score
   3. 模型保存：
      1. [X] 每次保存最优模型IoUbest  保存后删除前面的模型
      2. [X] 训练完成后保存最终模型
      3. [ ] 早停策略
   4. 训练指标可视化
      1. [X] 可视化Loss下降
      2. [ ] 可视化IoU
2. 支持导入模型推理分割，注意导入的权重中需要包含模型本身

   1. [X] 导入模型
   2. [X] 推理分割
3. Dataset类

   1. [X] 各种transforms
   2. [ ] 支持多类数据类型输入
   3. [ ] 支持多种数据集标注形式
4. 模型类

   1. [X] 使用timm接口导入模型训练
   2. 分割模型：
      1. [X] Unet
      2. [X] PSPNet
      3. [X] DeepLabv3+
      4. [X] UperNet


