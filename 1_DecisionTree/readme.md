## Sklearn built-in DecisionTreeClassfier v.s. My DecisionTree

|Accuracy(%)  |  Fold1  |  Fold2  |  Fold3  |  Fold4  |  Fold5  |  Avg.  |
|:------------|:-------:|:-------:|:-------:|:-------:|:-------:|:------:|
|Built-in     |  92.98  |  95.61  |  91.22  |  96.49  |  92.92  |  93.85 |
|Mine         |  93.86  |  95.61  |  92.98  |  94.73  |  95.57  |  94.55 |

|Train time(s)|  Fold1  |  Fold2  |  Fold3  |  Fold4  |  Fold5  |  Avg.  |
|:------------|:-------:|:-------:|:-------:|:-------:|:-------:|:------:|
|Built-in     |  0.0081 |  0.0069 |  0.0064 |  0.0062 |  0.0080 | 0.0071 |
|Mine         |  4.9146 |  5.5333 |  5.1658 |  5.1595 |  5.2357 | 5.2018 |

|Test time(ms)|  Fold1  |  Fold2  |  Fold3  |  Fold4  |  Fold5  |  Avg.  |
|:------------|:-------:|:-------:|:-------:|:-------:|:-------:|:------:|
|Built-in     |  0.4890 |  0.6101 |  0.3319 |  0.2789 |  0.3757 | 0.4171 |
|Mine         |  0.2398 |  0.2649 |  0.2441 |  0.2868 |  0.2698 | 0.2611 |

## Reference

1. https://github.com/JyiHUO/algorithm-from-scratch/tree/master/DecisionTree

2. 《统计学习方法(第二版)》, 李航
