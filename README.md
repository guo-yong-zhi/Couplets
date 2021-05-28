# Couplets
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guo-yong-zhi/Couplets/master?filepath=run.ipynb)  
基于["Attention is All You Need"](https://arxiv.org/abs/1706.03762)中的Transformer实现自动对对联。较大的改动有：更改了DecoderLayer中第二层attention的query, key, value以捕获上联的模式和下联的语义，增加了一层memory的线性叠加以捕获上联的语义，调整了PositionalEncoding的方式。另外还实现了greedy、一维beam search、二维beam search等解码方式。
## 输出示例
| 上联                                                     | 参考下联                                         | 系统输出                                                 |
| -------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| 锦瑟无弦惊日月                                           | 铜钱有脚走街衢                                   | 瑶琴有韵醉春秋                                           |
| 金龙辞旧，秦岭山河千般秀                                 | 银蛇迎春，神州日月万象新                         | 银蛇迎新，神州大地万里春                                 |
| 天涯苦远，讯信迷茫，唯凭一纸传吟，纵有相思难会见         | 雅座逢缘，人情淡薄，独具千般才智，何因惜玉又怜香 | 海角逍遥，心怀寂寞，只把三杯对饮，已无寂寞不相逢         |
| 提刀上马，江山如画                                       | ——                                               | 把酒临风，岁月似歌                                       |
| 兰亭临帖，行书如行云流水                                 | ——                                               | 柳岸抚琴，弄曲似弄月吟风                                 |
| 只有刚强的人，才有神圣的意志，凡是战斗的人，才能取得胜利 | ——                                               | 若无敬畏之事，业无天地之精神，大为人民之事，业可成就辉煌 |

## 使用方法
1. 使用预训练模型对对联：`python run.py`。如需指定下联部分字，中间用`|`隔开，下联空字用空格或减号占位。[在线试试](https://mybinder.org/v2/gh/guo-yong-zhi/Couplets/master?filepath=run.ipynb)  
    > 输入示例：
    >* 白日依山尽  
    >* 白日依山尽|-河-海  
    >* 白日依山尽|明月  
2. 训练过程见`couplets.ipynb`  

## 规模
* 训练集770491（77万）条，测试集4000条
* 模型可训练参数52825311（50M）
* 词表（汉字表）容量6745
* 词向量512维，其中位置编码10维
* FFN内部层2048维

## to do
* [ ] beam search使用batch mode  
* [ ] 在beam search中，根据字频降低相似对联的得分  
* [ ] decoder共享使用encoder的注意力权重或参数，直接捕获上下联的对应关系 
* [ ] 使用大型预训练模型的词嵌入向量或浅层特征 

*大量代码copy自[annotated-transformer](https://github.com/harvardnlp/annotated-transformer)*  
*训练数据来自[couplet-dataset](https://github.com/wb14123/couplet-dataset)*