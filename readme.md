采用cnn对mnist分类。实现了bn（滑动平均）。

运行：python main.py

为了记录训练过程的loss值，cnn和mlp的main.py中都修改了train_epoch函数，另外为了记录训练时间及准确率，训练和测试的代码中也增加了一些语句。

每次训练会生成loss.txt文件记录每5次迭代的loss、log.txt文件记录训练时长，测试会在log.txt中记录准确率。
