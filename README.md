# alexnet
这是一个图像识别的神经网络(CNN卷积神经网络)  

my_layers:神经层  
alexnet: main函数  
read_data:读取图片和标签进行训练  
image_to_tfcode: 写入数据   

由于现在的训练基本都在GPU上进行，普通的计算机训练模型可能需要 3-4天的样子才可以把模型跑出来(前提是你的内存够用，至少8G吧)，训练次数需要增加  

首先对于这个模型是经典的alexnet模型，分为5个卷积层，3个全连接层。但是貌似google用卷积层代替了全连接层，而且效果还挺明显的。人家google已经实现好的模型我们就不再写一个了，我们写一个经典的alexnet模型吧。  

这个模型是在alexnnet的基础上做了一定的修改，就是我们在卷积层里面加入了BN算法，这样可以快速线性回归也可以不用太担心过拟合。代码里面文档很清楚了。
