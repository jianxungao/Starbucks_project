# 星巴克数据分析项目

## 项目简介
理解顾客购买行为, 对以提供产品和服务为主要盈利目的大多数企业都是至关重要的. 通过对顾客购买行为的分析, 可以帮助企业提供更好的产品和服务, 进而获取更多的市场竞争优势. 那么如何更好的理解顾客的购买行为, 一种有效的方式是通过对顾客消费记录进行数据挖掘，使用机器学习的手段, 来**发现**数据背后隐藏的信息.

### 背景
原始数据是Udacity数据科学家项目提供的. Udacity提供的一些模拟 Starbucks rewards 移动 app 上用户行为的数据. 这个数据集是从星巴克 app 的真实数据简化而来. 此模拟器仅产生了一种饮品, 实际上星巴克的饮品有几十种. 一般情况下, 每隔几天, 星巴克会向 app 的用户发送一些推送.

### 数据信息
1. 推送信息，可能仅仅是一条饮品的广告或者是折扣券或 BOGO（买一送一）。一些顾客可能一连几周都收不到任何推送。每种推送都有有效期。例如，买一送一（BOGO）优惠券推送的有效期可能只有 5 天。你会发现数据集中即使是一些消息型的推送都有有效期，哪怕这些推送仅仅是饮品的广告，例如，如果一条消息型推送的有效期是 7 天，你可以认为是该顾客在这 7 天都可能受到这条推送的影响。
2. 交易信息，数据集中还包含 app 上支付的交易信息，交易信息包括购买时间和购买支付的金额。交易信息还包括该顾客收到的推送种类和数量以及看了该推送的时间。顾客做出了购买行为也会产生一条记录。
3. 备注信息，这个数据集里有一些地方需要注意。即，这个推送是自动生效的；顾客收到推送后，即使没有看到，满足了条件，推送的优惠依然能够生效。比如，一个顾客收到了"满10美元减2美元优惠券"的推送，但是该用户在 10 天有效期内从来没有打开看到过它。该顾客在 10 天内累计消费了 15 美元。数据集也会记录他满足了推送的要求，然而，这个顾客并没被受到这个推送的影响，因为他并不知道它的存在。另外，有可能顾客购买了商品，但没有收到或者没有看推送。例如，一个顾客在周一收到了满 10 美元减 2 美元的优惠券推送。这个推送的有效期从收到日算起一共 10 天。如果该顾客在有效日期内的消费累计达到了 10 美元，该顾客就满足了该推送的要求。

### 项目目标
此任务是将交易数据、顾客统计数据和推送数据结合起来判断哪一类人群会受到某种推送的影响。需要考虑到某类人群即使没有收到推送，也会购买的情况。从商业角度出发，如果顾客无论是否收到推送都打算花 10 美元，我们并不希望给他发送满 10 美元减 2 美元的优惠券推送。所以可能需要分析某类人群在没有任何推送的情况下会购买什么。


### 项目工作流
1. 数据清洗
2. 数据汇总
3. 特征抽取
4. 特征工程
5. 机器学习建模


# 数据基本信息
一共有三个数据文件：

* portfolio.json – 包括推送的 id 和每个推送的元数据（持续时间、种类等等）
* profile.json – 每个顾客的人口统计数据
* transcript.json – 交易、收到的推送、查看的推送和完成的推送的记录

以下是文件中每个变量的类型和解释 ：

**portfolio.json**
* id (string) – 推送的id
* offer_type (string) – 推送的种类，例如 BOGO、打折（discount）、信息（informational）
* difficulty (int) – 满足推送的要求所需的最少花费
* reward (int) – 满足推送的要求后给与的优惠
* duration (int) – 推送持续的时间，单位是天
* channels (字符串列表)

**profile.json**
* age (int) – 顾客的年龄 
* became_member_on (int) – 该顾客第一次注册app的时间
* gender (str) – 顾客的性别（注意除了表示男性的 M 和表示女性的 F 之外，还有表示其他的 O）
* id (str) – 顾客id
* income (float) – 顾客的收入

**transcript.json**
* event (str) – 记录的描述（比如交易记录、推送已收到、推送已阅）
* person (str) – 顾客id
* time (int) – 单位是小时，测试开始时计时。该数据从时间点 t=0 开始
* value - (dict of strings) – 推送的id 或者交易的数额

# 数据清洗
1. 发现缺失数据
<img src="./images/check_null.png" alt="check_null" width="500">

2. 删除缺失数据
<img src="./images/delete_null.png" alt="delete_null" width="500">

3. 发现离群值
<img src="./images/outliers.png" alt="outliers" width="500">

# 数据分析
数据一般的统计学分析：

年龄分布:

1. 顾客平均年龄54岁，中位数是55岁，最小18岁，最大101岁，标准差17.38。
2. 其中男性顾客平均年龄52岁，中位数是52岁，最小18岁，最大100岁，标准差17.41。
3. 其中女性顾客平均年龄57岁，中位数是58岁，最小18岁，最大101岁，标准差16.88。
<img src="./images/age_dist.png" alt="age" width="500">

性别分布:

所有顾客中，57.2%是男性， 41.3%是女性， 其他是1.43%

<img src="./images/gender.png" alt="gender" width="500">

年龄和消费水平的关系(男性):

1. 男性顾客随着年龄的增加，对于平均完成推送一定的正向线性关系。
<img src="./images/male_offer_complete.png" alt="male_offer_complete" width="500">
2. 男性顾客随着年龄的增加，对于平均购买交易次数有负向的线性关系，但是对于平均购买量有正向对线性关系。
<img src="./images/male_age_amount.png" alt="male_age_amount" width="500">
<img src="./images/male_age_tnx.png" alt="male_age_tnx" width="500">

年龄和消费水平的关系(女性):

1. 女性顾客随着年龄的增加，对于平均完成推送相关性不明显。
<img src="./images/female_offer_complete.png" alt="female_offer_complete" width="500">
2. 女性顾客随着年龄的增加，对于平均购买交易次数有负相关的趋势，但是60岁以后持平。
<img src="./images/female_age_tnx.png" alt="female_age_tnx" width="500">

3. 女性顾客随着年龄的增加，对于平均购买量是随年龄增加而增加, 但是70岁以后有下降的趋势。
<img src="./images/female_age_amount.png" alt="female_age_amount" width="500">
 
收入水平和消费的关系：

* 收入低于50k以下的男顾客和女顾客在消费量，完成推送交易，购买次数上有显著性。说明这两个组购买行为有明显的不同。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|10.7606|0.0|
|`完成推送交易`|11.4290|0.0|
|`购买次数`|4.2466|1.1115282947171679e-05|

* 收入在50k和75k之间的男顾客和女顾客在消费量，完成推送交易上有显著性。在消费次数上没有显著性。说明此二组在购买次数上相当，在购买量和完成推送上行为不一样。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|12.7067|0.0|
|`完成推送交易`|14.2658|0.0|
|`购买次数`|-0.1511|0.56|

* 收入高于75k的男顾客和女顾客在消费量，完成推送交易，在消费次数上都没有显著性。说明此二组购买行为有一定的类似性。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|-0.4629|0.68|
|`完成推送交易`|0.8434|0.20|
|`购买次数`|-0.6398|0.74|


* 收入低于50k的顾客和收入在50k和75k之间顾客在消费量，完成推送交易，存在显著性。在消费次数上都没有显著性。说明此二组在购买次数上类似，但在花费和完成推送交易上不同。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|22.0496|0.0|
|`完成推送交易`|17.9547|0.0|
|`购买次数`|-11.1395|1.0|

* 收入低于50k的顾客和收入高于75k的顾客在消费量，完成推送交易，存在显著性。在消费次数上都没有显著性。说明此二组在购买次数上类似，但在花费和完成推送交易上不同。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|38.0860|0.0|
|`完成推送交易`|34.3149|0.0|
|`购买次数`|-35.4485|1.0|

* 收入在50k和75k之间顾客和收入高于75k的顾客在消费量，完成推送交易，不存在显著性。在消费次数上有显著性。说明此二组在购买次数上不同，但在花费和完成推送交易上类似。

|特征|t-score|p-value|
|---|---|---|
|`消费量`|-21.9972|1.0|
|`完成推送交易`|-19.2598|1.0|
|`购买次数`|29.7061|0.0|



# 特征抽取
从交易数据中提取了以下特征:

特征描述：
| feature name | description | 
|---|---|
|`age`                                 |年龄|
|`income`                              |收入|
|`membership_since_in_days`            |会员天数|
|`amount                          `    |消费总计|
|`reward_x                        `    |消费后获得的奖励|
|`offer completed                 `    |推送完成总计|
|`offer received                  `    |收到推送总计|
|`offer viewed                    `    |查看推送总计|
|`transaction                      `   |交易总计|
|`bogo_1_received                  `   |收到买一送一1总计|
|`bogo_1_viewed                    `   |查看买一送一1总计|
|`bogo_1_completed                 `   |完成买一送一1总计|
|`bogo_2_received                  `   |收到买一送一2总计|
|`bogo_2_viewed                    `   |查看买一送一2总计|
|`bogo_2_completed                 `   |完成买一送一2总计|
|`informational_1_received         `   |收到信息推送1总计|
|`informational_1_viewed           `   |查看信息推送1总计|
|`bogo_3_received                  `   |收到买一送一3总计|
|`bogo_3_viewed                    `   |查看买一送一3总计|
|`bogo_3_completed                 `   |完成买一送一3总计|
|`discount_1_received              `   |收到折扣1总计|
|`discount_1_viewed                `   |查看折扣1总计|
|`discount_1_completed             `   |完成折扣1总计|
|`discount_2_received              `   |收到折扣2总计|
|`discount_2_viewed                `   |查看折扣2总计|
|`discount_2_completed             `   |完成折扣2总计|
|`discount_3_received              `   |收到折扣3总计|
|`discount_3_viewed                `   |查看折扣3总计|
|`discount_3_completed             `   |完成折扣3总计|
|`informational_2_received         `   |收到信息推送2总计|
|`informational_2_viewed           `   |查看信息推送2总计|
|`bogo_4_received                  `   |收到买一送一4总计|
|`bogo_4_viewed                    `   |查看买一送一4总计|
|`bogo_4_completed                 `   |完成买一送一4总计|
|`discount_4_received              `   |收到折扣4总计|
|`discount_4_viewed                `   |查看折扣4总计|
|`discount_4_completed             `   |完成折扣4总计|
|`bogo_received                    `   |收到所有4种买一送一总计|
|`bogo_viewed                      `   |查看所有4种买一送一总计|
|`bogo_completed                   `   |完成所有4种买一送一总计|
|`discount_received                `   |收到所有4种折扣总计|
|`discount_viewed                  `   |查看所有4种折扣总计|
|`discount_completed               `   |完成所有4种折扣总计|
|`informational_received           `   |收到所有2种信息推送总计|
|`informational_viewed             `   |查看所有2种信息推送总计|
|`transaction_completed_ratio      `   |交易次数与推送完成比|
|`offer_viewed_rate                `   |查看推送与收到推送比|
|`offer_completed_received_rate    `   |完成推送与收到推送比|
|`offer_completed_viewed_rate      `   |完成推送与查看推送比|
|`bogo_viewed_rate                 `   |所有买一送一查看与收到比|
|`bogo_completed_received_rate     `   |所有买一送一完成与收到比|
|`bogo_completed_viewed_rate       `   |所有买一送一完成与查看比|
|`discount_viewed_rate             `   |所有折扣查看与收到比|
|`discount_completed_received_rate`   |所有折扣完成与收到比|
|`discount_completed_viewed_rate  `   |所有折扣完成与查看比|
|`informational_viewed_rate       `   |信息查看与收到比|
|`F                               `   |女性one-hot|
|`M                               `   |男性one-hot|
|`O                               `   |性别其他|


# 机器学习建模
## PCA降维
通过PCA帮助理解数据：

前30个PCA Components可以解释原始数据97%的变异数
<img src="./images/pca.png" alt="pca" width="500">

### 前3个PCA Components的特征权重
<img src="./images/pca1.png" alt="First_component" width="500">
<img src="./images/pca2.png" alt="Second_component" width="500">
<img src="./images/pca3.png" alt="Third_component" width="500">

## 无监督学习 - KMeans Clustering
### Sihouette分数
Silhouette数值度量组内样本，与组间样本相比的紧密程度。It is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). 值越大越好。

### Inertia指标
Inertia度量所有组内样本与组中心的距离。
Inertia calculates the sum of distances of all the points within a cluster from the centroid of that cluster. 值越小越好。

### 如何决定K值
如图所示，通过计算不同K-Means聚类的Sihouette分数和Inertia指标，K=13比较理想。
<img src="./images/sihouette.png" alt="k_value" width="500">

## 监督学习 - 预测推送完成比率
监督学习建模：

<img src="./images/supervised.png" alt="supervised" width="500">


|Model|Performance|
|---|---|
|`Random Forest`|R^2 0.9927, MSE 0.007258|
|`Neural Network`|R^2 0.9998, MSE 0.000181|



# 发现与结论

顾客聚类分析结果：

10 推送概览：

<img src="./images/10_offers.png" alt="10_offers" width="500">

一般情况：
1. 各个群组的平均年龄比较接近。
2. 群组9平均消费水平和收到的奖励都是最高的一组。
3. 各个群组的平均收入水平也相对接近，群组0，2，6平均收入相对比较低， 群组6的平均收入相对最低。
4. 群组0，2平均消费水平和收到奖励都是最低的两组。
<img src="./images/age_income.png" alt="age_income" width="500">
5. 各个群组的平均会员时间比较接近， 群组6相对较长。
6. 群组0，2，6男性顾客比女性顾客多。群组6的男性顾客为80%，女性为20%。其他组的男性和女性顾客比率比较接近。
<img src="./images/member_gender.png" alt="membership_gender" width="500">

对各种推送：
1. 各个群组收到的各种推送总体而言比较接近。
2. 群组9是推送完成和平均消费最高的一组。
3. 群组0，2完成推送和平均消费都是最低组。
<img src="./images/offer_general.png" alt="offer_general" width="500">

对折扣推送：
1. 群组11完成折扣1推送最高。
2. 群组10完成折扣2推送最高。
3. 群组12完成折扣3推送最高。
4. 群组8完成折扣4推送最高。
5. 群组7是几乎没有收到折扣推送的一组。
6. 群组0和群组2，大约10%顾客完成折扣推送， 是完成折扣推送比率最低的组，说明此二组顾客对折扣推送没有兴趣。
<img src="./images/discount_general.png" alt="discount_general" width="500">

折扣1：
1. 群组7几乎没有收到折扣1的推送。
2. 群组11接收，查看和完成折扣1推送比较高。
<img src="./images/discount_1.png" alt="discount_1" width="500">

折扣2:
1. 群组7几乎没有收到折扣2推送。
2. 群组10接收，查看和完成折扣2推送比较高。
3. 群组0，2，完成折扣2推送比较低。
<img src="./images/discount_2.png" alt="discount_2" width="500">

折扣3:
1. 群组7几乎没有收到此种折扣3推送。
2. 群组12接收，查看和完成折扣3推送比较高。
3. 群组0，2，完成折扣3推送比较低。
<img src="./images/discount_3.png" alt="discount_3" width="500">

折扣4:
1. 群组7几乎没有收到折扣4推送。
2. 群组8接收，查看和完成折扣4推送比较高。
3. 群组0，2，完成折扣4推送比较低。
<img src="./images/discount_4.png" alt="discount_4" width="500">



对买一送一推送：
1. 群组9完成买一送一1和2推送完成最多。
2. 群组3完成买一送一3推送完成最多。
3. 群组5完成买一送一4推送完成最多。
4. 群组7完成4种买一送一推送次多。
5. 群组4，是几乎没有收到买一送一的推送的一组，
6. 群组9，是收到买一送一的推送最多的组，完成此推送的也是最多的。
7. 群组1，3，5，9接收完成比超过75%。说明此4组对买一送一推送比较有兴趣。
8. 群组0，2，6接收并且查看了，但是完成比率较低。 说明此3组对买一送一推送响应度比较低。
<img src="./images/bogo_general.png" alt="bogo_general" width="500">

买一送一1:
1. 群组9完成买一送一1推送完成最多。
2. 群组4几乎没有收到买一送一1推送。
<img src="./images/bogo_1.png" alt="bogo_1" width="500">

买一送一2:
1. 群组9完成买一送一2推送完成最多。
2. 群组4几乎没有收到买一送一2推送。
<img src="./images/bogo_2.png" alt="bogo_2" width="500">

买一送一3:
1. 群组3完成买一送一3推送完成最多。
2. 群组4几乎没有收到买一送一3推送。
<img src="./images/bogo_3.png" alt="bogo_3l" width="500">

买一送一4:
1. 群组5完成买一送一4推送完成最多。
2. 群组4几乎没有收到买一送一4推送。
<img src="./images/bogo_4.png" alt="bogo_4" width="500">


对于信息对推送：
1. 群组0和1接近100%的顾客查看了信息推送，
2. 群组4，7，对信息推送的查看率接近于80%。
3. 其他群组查看率低于40%。
<img src="./images/info_general.png" alt="info_general" width="500">

信息1:
第一类信息较多发送给群组1。
信息2:
第二类信息发送给群组0比较多，群组2几乎没有接收到次类信息推送。
<img src="./images/info.png" alt="info" width="500">



# 回顾与总结
1. Pandas Dataframe使用技巧在数据处理中有所提高。
2. 深入了解了PCA和K-Means聚类结合来发现数据背后的真相。
3. 更好的理解聚类算法，加深对评价指标的了解。

# 感谢
感谢Udacity提供的精心设计的学习项目。谢谢。

# References
1. [Sklearn Cluster](https://scikit-learn.org/stable/modules/clustering.html) 

2. PULKIT SHARMA, [The Most Comprehensive Guide to K-Means Clustering You’ll Ever Need](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/#k-means-clustering-python-code)

3. [Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

4. [Jeffri Sandy](https://medium.com/@jeffrisandy/investigating-starbucks-customers-segmentation-using-unsupervised-machine-learning-10b2ac0cfd3b)






