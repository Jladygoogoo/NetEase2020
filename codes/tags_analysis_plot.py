import numpy as np  
import matplotlib.pyplot as plt
from powerlaw import plot_pdf, Fit, pdf

# from tags_analysis import ClusetrsSet,TagsCluster

from numpy import logspace, histogram, floor, unique,asarray
from scipy.optimize import leastsq
from math import ceil, log10
# 我是看不懂他在干什么了
def power_law_plot(data, title, save=False, save_path=None):
    data = asarray(data)

    avg = np.mean(data)
    xmin, xmax = np.min(data), np.max(data)
    log_min_size, log_max_size = log10(xmin), log10(xmax)
    number_of_bins = ceil((log_max_size-log_min_size)*10)
    bins = logspace(log_min_size, log_max_size, num=number_of_bins)
    bins[:-1] = floor(bins[:-1])
    bins[-1] = ceil(bins[-1])
    bins = unique(bins)
    # print(bins)

    hist, edges = histogram(data, bins, density=False)
    # hist[hist==0] = 1
    y = hist
    x = (edges[1:] + edges[:-1])/2
    y = [p[0] for p in list(filter(lambda p:p[0]>0, zip(y,x)))]
    x = [p[1] for p in list(filter(lambda p:p[0]>0, zip(y,x)))]
    
    if len(y)<=3: return None, None, None

    logy = np.log10(y)
    logx = np.log10(x)

    # print('\n',logy)
    # print(logx,'\n')

    beta = np.polyfit(logx,logy,1)
    logy_hat = beta[0]*np.array(logx) + beta[1]
    std_sigma = (np.mean((logy-logy_hat)**2))**0.5/np.mean(logy)

    plt.scatter(logx,logy)
    plt.plot(logx,logy_hat,'r')

    plt.title(title)
    plt.ylim(-0.1,1.5)
    plt.xlim(2,3.5)
    plt.annotate('\nalpha={:.2f},  sigma={:.2f}, avg={:.2f}'.format(beta[0],std_sigma,avg),
                xy=(0.2,0.2), xycoords="figure fraction")
    if save:
        plt.savefig(save_path)
    plt.close()

    return beta[0], std_sigma, avg


def draw_donut(all_rank_tags):
    cols = 2
    rows = math.ceil(len(all_rank_tags)/2)

    pie = Pie(init_opts={'width':'1200px','height':'1000px'})
    for i,rank_tags in enumerate(all_rank_tags):
        rank_s,rank_e = ranks_list[i]

        pie.add(
            '[{},{}]'.format(rank_s*100,rank_e*100-1),
            rank_tags,
            center=['{}%'.format(30+30*(i%2)), '{}%'.format(15+25*(i//2))],
            radius=[40,70]
        )

    pie.set_global_opts(
        title_opts=opts.TitleOpts(title='clusters distribution in reviews-rank'),
        legend_opts=opts.LegendOpts(type_="scroll", pos_top="20%", pos_left="80%", orient="vertical")
    ).set_series_opts(
        label_opts=opts.LabelOpts(formatter="{b}: {c}")
    )

    pie.render(path=image_save_path)


def draw_heap(all_ranks_tags,ranks_list):
    attr = list(map(lambda x:str(x), ranks_list))

    tags_rank_dist = {}
    for i,rank_tags in enumerate(all_ranks_tags):
        total = np.sum([x[1] for x in rank_tags])
        for tag,num in rank_tags:
            if tag not in tags_rank_dist: 
                tags_rank_dist[tag] = [0]*i + [num/total]
            else:
                tags_rank_dist[tag] += [0]*(i-len(tags_rank_dist[tag])) + [num/total]
    for k,v in tags_rank_dist.items():
        tags_rank_dist[k] += [0]*(len(ranks_list)-len(v))
        # print(tags_rank_dist[k])
        # print("{}: {}".format(k,tags_rank_dist[k]))
    tags_rank_dist = dict(sorted(tags_rank_dist.items(), key=lambda x:np.sum(x[1]), reverse=True))

    bar = Bar(init_opts={'width':'1000px','height':'800px','theme':'roma'})
    bar.add_xaxis(attr)
    for k,v in tags_rank_dist.items():
        bar.add_yaxis(k,v,stack='stack1')
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title='clusters distribution in reviews-rank [heap]'),
        xaxis_opts=opts.AxisOpts(name='reviews_rank'),
        yaxis_opts=opts.AxisOpts(name='clusters_dist',max_=1.3),
        legend_opts=opts.LegendOpts(type_="scroll", pos_top="10%", pos_left="20%", orient="horizontal")
    ).set_series_opts(
        label_opts=opts.LabelOpts(formatter=" ")
    )
    bar.render(path=image_save_path)



def plplot(data, title, save=False, save_path=None):
    data = np.array(data)

    fig = plt.figure(figsize=(18,6))
    fig.suptitle(title)
    
    # === A ===
    ax1 = fig.add_subplot(1,3,1)

    # 线性x轴
    x, y = pdf(data, linear_bins=True)
    ind = y>0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    ax1.scatter(x, y, color='r', s=.5)

    # 双log-绘制概率密度曲线
    plot_pdf(data[data>0], ax=ax1, color='b', linewidth=2)

    ax1.set_xlabel('A')
    
    # 绘制histogram小图
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax1in = inset_axes(ax1, width = "30%", height = "30%", loc=3)
    ax1in.hist(data, normed=True, color='b')
    ax1in.set_xticks([])
    ax1in.set_yticks([])

    # === A ===

    # === B ===
    
    annotation = ''
    ax2 = fig.add_subplot(1,3,2, sharey=ax1)

    # 双log-绘制概率密度曲线
    print(title)
    print(pdf(data))
    print()
    plot_pdf(data, ax=ax2, color='b', linewidth=2)

    # 拟合power-law函数并绘图
    fit = Fit(data, xmin=1, discrete=True, parameter_range={'alpha':[None,None]})
    fit.power_law.plot_pdf(ax=ax2, linestyle=':', color='g')
    params1 = (fit.power_law.alpha, fit.power_law.xmin, fit.power_law.sigma)

    # alpha为拟合系数
    # xmin表示最小的x值(使不为0)，此处指定为1
    # sigma为标准差
    annotation += '\':\' - alpha={:.2f}, xmin= {}, sigma={:.2f}'.format(*params1)
    # p = fit.power_law.pdf()
    
    fit = Fit(data, discrete=True, parameter_range={'alpha':[-5,10]})
    # 区别于ax2中的第一条拟合线 - 此处的xmin并非指定，而是自动计算的optimal
    fit.power_law.plot_pdf(ax=ax2, linestyle='--', color='g')
    params2 = (fit.power_law.alpha, fit.power_law.xmin, fit.power_law.sigma)
    annotation += '\n\'--\' - alpha={:.2f}, xmin= {}, sigma={:.2f}'.format(*params2)

    ax2.set_xlabel('B')
    ax2.set_ylabel(u"p(X)")# (10^n)")
    ax2.set_xlim(ax1.get_xlim())
    annotate_coord = (0.05, 0.88)
    ax2.annotate(annotation, annotate_coord, xycoords="axes fraction")
        
    # === B ===

    # === C ===

    ax3 = fig.add_subplot(1,3,3, sharey=ax1)#, sharex=ax1)#, sharey=ax2)
    plot_pdf(data[data>0], ax=ax3, color='b', linewidth=2)
    fit.power_law.plot_pdf(ax=ax3, linestyle='--', color='g')
    fit.exponential.plot_pdf(ax=ax3, linestyle='--', color='r')

    
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax1.get_xlim())

    ax3.set_xlabel('C')

    # === C ===

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    return params1, params2


def test():
    data = np.array(ClusetrsSet.load('../models/clusters/100nt10a6_tags_clusters_model.pkl').clusters[1].reviews_nums)
    # plplot(data,'test','DC marvel loki thor')
    fit_power_law(data)

if __name__ == '__main__':
	test()
