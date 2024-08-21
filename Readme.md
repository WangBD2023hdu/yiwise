## 模型训练
`python train.py`

PS:
- 训练过程中会自动保存在验证集上效果最好的模型，文件`best_f1_micro_model.pth`和`best_f1_macro_model.pth`

- label_map.txt保存了模型训练时使用的标签映射关系，方便推理使用，文件第一行不能修改。
