import matplotlib.pyplot as plt


plt.figure(figsize=(10, 7), dpi=100)
game = ['2', '4', '6', '8', '10', ]
scores = [1.00,0.977 ,0.962 , 0.94 ,0.923]
plt.plot(game, scores, c='black')
plt.scatter(game, scores, c='black')
y_ticks = [0.75,0.80,0.85,0.90,0.95,1.00,1.05]
plt.yticks(y_ticks[:])
# plt.grid(True, linestyle='--', alpha=0.5)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("     {0,1}         {0,1,2,3}     {0,1,2,3,4,5}     {0,1,2,3      {0,1,2,3,4,  \n                                                   ,4,5,6,7}      5,6,7,8,9}  ",fontdict={'size': 16})
plt.ylabel("平均测试准确率", fontdict={'size': 16})
# plt.title("NBA2020季后赛詹姆斯得分", fontdict={'size': 20})
plt.show()
