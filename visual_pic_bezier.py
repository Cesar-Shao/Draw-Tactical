#❓❓❓❓❓SeqCond 表现比 RealCond好

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.interpolate import make_interp_spline
from Bezier import bezier_curve
from matplotlib.patches import FancyArrowPatch

def get_res(seqcond_,seq_):
    for i in range(seq_.shape[0]):
        for j in range(seq_.shape[1]):
            if np.count_nonzero(1) == 1:
                # 获取元素为 1 的索引
                location_1 = np.where(seqcond_[i, j] == 1)[0]
                if not location_1.size == 0:
                    if not location_1[0] == 5:
                        index = location_1[0]
                # print("1 的位置是:", index)
                    seq_[i, j, 0] = seq_[i, j, index]
    return seq_



def cut_seq(seq,seqcond):
    flag = True
    pass_ls=[]
    shoot_ls=[]
    temp=[]
    for i in range(seqcond.shape[0]):
        if np.all(seqcond[i,:] == 0):
            if flag:
                temp.append(i)
                flag=False
        elif seqcond[i,5]==1 :
            shoot_ls.append(i)
        
        else:
            
            if not flag:
                temp.append(i)
                flag=True
                pass_ls.append(temp)
                temp=[]
                handler=np.argwhere(seqcond[i,:] == 1)
    return pass_ls,shoot_ls
            
def draw_tac(player_data,ball_data,handler_data,i_):

    # 创建图形对象
    fig, ax = plt.subplots(figsize=(6, 6))  # 调整图形对象的大小
    print(i_,'!!!!!!!!!')
    # 载入篮球场地背景图像
    court_img = plt.imread('data/court.png')
    height=court_img.shape[0]*0.1
    width=court_img.shape[1]*0.1
    ax.imshow(court_img, extent=[0,width,0,height])

    for i in range(len(player_data)):
                
            x=bezier_curve(player_data[i])[:,0]
            y=bezier_curve(player_data[i])[:,1]
            plt.plot(x, y,color='black',linestyle='-')
            
            # 添加箭头
            try:
                arrow_start = (x[len(x)//2], y[len(x)//2])  # 箭头起点为倒数第二个点
                arrow_end = (x[len(x)//2+1], y[len(x)//2+1])  # 箭头终点为最后一个点
                
                #arrow_start = (x[0], y[0])  # 箭头起点为倒数第二个点
                #arrow_end = (x[-1], y[-1])  # 箭头终点为最后一个点

                plt.annotate('', xy=arrow_end, xytext=arrow_start, arrowprops=dict(arrowstyle='->', color='black',mutation_scale=20),annotation_clip=False)
                
                '''arrow_direction=(player_data[-1][0]-player_data[0][0],player_data[-1][1]-player_data[0][1])
                arrow = FancyArrowPatch(arrow_start, arrow_direction, arrowstyle='->', color='orange',mutation_scale=20)

                plt.gca().add_patch(arrow)'''
            except:
                pass
            
    
    for i in range(len(ball_data)):
            
            x=bezier_curve(ball_data[i])[:,0]
            y=bezier_curve(ball_data[i])[:,1]
            
            plt.plot(x, y,color='red',linestyle='dotted')
            # 添加箭头
            try:
                arrow_start = (x[len(x)//2], y[len(x)//2])  # 箭头起点为倒数第二个点
                arrow_end = (x[len(x)//2+1], y[len(x)//2+1])  # 箭头终点为最后一个点

                plt.annotate('', xy=arrow_end, xytext=arrow_start, arrowprops=dict(arrowstyle='->', color='red',mutation_scale=20),annotation_clip=False)
            except:
                pass
        
    x=bezier_curve(handler_data)[:,0]
    y=bezier_curve(handler_data)[:,1]
    plt.plot(x, y,color='orange',linestyle='-')
    try:
        arrow_start = (x[len(x)//2], y[len(x)//2])  # 箭头起点为倒数第二个点
        arrow_end = (x[len(x)//2+1], y[len(x)//2+1])  # 箭头终点为最后一个点

        plt.annotate('', xy=arrow_end, xytext=arrow_start, arrowprops=dict(arrowstyle='->', color='orange',mutation_scale=20),annotation_clip=False)
    except:
        pass

    # Add labels and title to the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curve Plot')
    plt.savefig(f'tactic_{i_}.png')
    print(i_,'end!!!!!!!!!')








# 载入数据
seqcond = np.load('data/SeqCond.npy')
seq = np.load('data/50Real.npy')
seq = seq[:,:,:6,:2]
data = seq[1,:,:]
seqcond=seqcond[1,:,:]  

# 提取球员和球的坐标数据
player_data = data[:,1:,:].reshape(50, 5, 2)
ball_data = data[:,:1,:].reshape(50, 2)



pass_ls,shoot_ls=cut_seq(seq,seqcond)

if len(pass_ls)>1:

    for i in range(len(pass_ls)-1):
        ball_cut=[]
        player_cut=[]
        if i==0:
            
            ball_cut.append(ball_data[pass_ls[i][0]:pass_ls[i][1],:])
            ball_cut.append(ball_data[pass_ls[i+1][0]:pass_ls[i+1][1],:])

            
            for j in range(player_data.shape[1]): 
                if j== np.argwhere(seqcond[pass_ls[i][1]]):
                    handler_cut=player_data[:pass_ls[i+1][1],j,:]
                    #player_cut.append(np.empty_like(player_data[:pass_ls[i+1][1],j,:]))
                    #player_cut.append(None)
                else:
                    player_cut.append(player_data[:pass_ls[i+1][1],j,:])
            
            draw_tac(player_cut,ball_cut,handler_cut,i)
        else:
            ball_cut.append(ball_data[pass_ls[i][0]:pass_ls[i][1],:])
            ball_cut.append(ball_data[pass_ls[i+1][0]:pass_ls[i+1][1],:])
            
            for j in range(player_data.shape[1]): 
                if j== np.argwhere(seqcond[pass_ls[i][1]]):
                    handler_cut=player_data[pass_ls[i][0]:pass_ls[i+1][1],j,:]
                    #player_cut.append(np.empty_like(player_data[pass_ls[i][0]:pass_ls[i+1][1],j,:]))
                    #player_cut.append(None)
                else:
                    player_cut.append(player_data[:pass_ls[i+1][1],j,:])
            
            
            draw_tac(player_cut,ball_cut,handler_cut,i)
        
        
