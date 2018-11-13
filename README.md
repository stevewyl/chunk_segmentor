# Chunk分词器使用指南

环境依赖：python 3.6 (暂时只支持python3)

## 主要功能

1. 能够输出名词短语
2. 支持词性输出，名词短语词性为np
3. 支持名词短语以限定词+中心词的形式输出

>不可分割的名词短语是不存在限定词+中心词的形式的，如“机器学习”，而“经典机器学习算法”可拆解为“经典_机器学习_算法”

## Step 1 安装软件包

推荐新建一个python的虚拟环境（可跳过）

```bash
conda create --name chunk_seg python=3.6.5
```

### pip安装

```bash
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install chunk-segmentor
```

### 手动安装

```bash
git clone https://github.com/stevewyl/chunk_segmentor
cd chunk_segmentor
pip install -r requirements.txt
python setup.py install
```

### 额外安装
```bash
# 若你的机器安装有GPU，利用GPU加速预测速度
pip install tensorflow-gpu==1.9.0
```
### 安装错误
1. ImportError: cannot import name 'normalize_data_format'
```bash
pip install -U keras
```

## Step 2 如何使用

* 第一次import的时候，会自动下载模型和字典数据  
* 支持单句和多句文本的输入格式，建议以列表的形式传入分词器

```python
from chunk_segmentor import Chunk_Segmentor
cutter = Chunk_Segmentor()
s = '这是一个能够输出名词短语的分词器，欢迎试用！'
res = [item for item in cutter.cut([s] * 10000)] # 1080ti上耗时12s

# 提供两个版本，accurate为精确版，fast为快速版但召回会降低一些，默认精确版
cutter = Chunk_Segmentor(mode='accurate')
cutter = Chunk_Segmentor(mode='fast')
# 限定词+中心词的形式, 默认开启
cutter.cut(s, qualifier=False)
# 是否输出词性， 默认开启
cutter.cut(s, pos=False)

# 输出格式（词列表，词性列表，chunk集合）
[
    (
        ['这', '是', '一个', '能够', '输出', '名词_短语', '的', '分词器', ',', '欢迎', '试用', '!'],
        ['rzv', 'vshi', 'mq', 'v', 'vn', 'np', 'ude1', 'np', 'w', 'v', 'v', 'w'],
        ['分词器', '名词_短语']
    )
    ...
]
```

## Step 3 后续更新

若存在新的模型和字典数据，会提示你是否需要更新

## To-Do Lists

1. 提升限定词和名词短语的准确性 ---> 新的模型
2. char模型存在GPU调用内存溢出的问题 ---> 使用cnn提取Nchar信息来代替embedding的方式，缩小模型规模
3. 自定义字典，支持不同粒度的切分
4. 多进程模型加载和预测
