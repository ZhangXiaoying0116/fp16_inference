`pb_auto_tools.py`  

### 功能描述：该文件涵盖pb文件的常用处理，针对所有pb模型自动化工具
#### 功能1：统计pb中算子的个数
#### 功能2：将ckpt的权重文件转换成pb文件
#### 功能3：检查ef32算子的转换，对于矩阵乘、卷积、可分离卷积是否完成相应的前向和反向的转换；前向input和filter插入转换，反向grad的输入插入转换
#### 功能4：手工转换fp32的原始模型至fp16，包含全部节点转换；对于无法转换的节点进行检查，并根据错误信息插入cast节点  
#### 功能5：自动转换fp32模型至混合精度模型，不是全部节点转换，图结构可导出；功能5与功能4可进行同步分析，确定模型最合适的fp16模型转换方法  
