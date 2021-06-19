## 混合精度

**学习资料**

- [混合精度-维基百科](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
- [Experiments with Mixed Prevision Algorithms in Linear Algebra](https://smc2021.ornl.gov/sessions/jack-dongarra-university-of-tennessee/)
- [NVIDIA DEVELOPER BLOG](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/)
- [浅谈混合精度训练](https://zhuanlan.zhihu.com/p/103685761)



**1. 为什么要混合精度计算**

- 减少所需要的内存，降低所需的内存可以训练更大的模型或训练更大的小批量。

- 缩短训练或推理时间。与单精度相比，NVIDIA GPU提供高达8倍以上的半精度运算吞吐量。

  

**2. 数值精度的表示**

![image-20210521153146822](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/image-20210521153146822.png)

由上图可知单精度数（float 32 ）大小为32bit，占据4个Byte。半精度数（float 16）大小为16bit，占据2个Byte。在数据量相同的情况下，存储半精度数所需要的内存空间是单精度数的一半。以IEEE 标准的半精度数（16位）为例，数值由三个部分组成：

1. 最高位表示符号位
2. exponent表示指数部分，在半精度数中用5位表示
3. fraction表示小数部分，取值范围在$[1,2)$之间，在半精度数中用10位表示

fraction 的位数越多，浮点数就越精确。而指数部分 exponent 的位数越多，浮点数能够表示的范围也就越大

![image-20210521161146699](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/数值表示.png)

1. **如果 exponent 位全部为0：**

   - 如果 fraction 位 全部为0，则表示数字 0

   - 如果 fraction 位 不为0，则表示一个非常小的数字（subnormal numbers），其计算方式：

     ![](<img src="https://latex.codecogs.com/svg.image?(-1)^{\text&space;{signbit&space;}}&space;\times&space;2^{-14}&space;\times\left(0&plus;\frac{\text&space;{&space;fraction&space;}}{1024}\right)&space;" title="(-1)^{\text {signbit }} \times 2^{-14} \times\left(0+\frac{\text { fraction }}{1024}\right) " />)

2. **如果 exponent 位全部位1：**

   - 如果 fraction 位 全部为0，则表示 ±inf

   - 如果 fraction 位 不为0，则表示 NAN
   
3. **exponent 位的其他情况：**

   计算方式为：$(-1)^{\text {signbit }} \times 2^{(\text {exponent-15 })} \times\left(1+\frac{\text { fraction }}{1024}\right)$



以IEEE Stardard半精度数(16位)为例，特殊的几个数：

- 最小小数的表示 (smallest  positive  subnormal  number)
  $$ 0 00000 00000000012 = 00011_{6} = {\displaystyle 2^{-14}\times (0+{\frac {1}{1024}})} ≈ 0.000000059604645$$ 

- 最大整数(largest normal number)

    $$0 11110 11111111112 = 7bff16 = {\displaystyle 2^{15}\times (1+{\frac {1023}{1024}})}{\displaystyle 2^{15}\times (1+{\frac {1023}{1024}})} = 65504$$
    
- 小于零的最大数 (largest number less than one)
$$ 0 01110 11111111112 = 3bff16 = {\displaystyle 2^{-1}\times (1+{\frac {1023}{1024}})}{\displaystyle 2^{-1}\times (1+{\frac {1023}{1024}})} ≈ 0.99951172$$

其他更多的边界情况可以参考[wiki](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)



**3. 混合精度计算**

降低数值进度进行运算在对精度要求比较高的应用场景下会导致精度的丢失，进而影响到程序的运算结果。但是降低运算精度会带来吞吐、矩阵计算速度的成倍提升，同时也可能会减少内存的占用。为了解决降低运算精度带来的损失，设计了对应的计算方法，使得混合精度的运算结果在一些任务能和双精度的运算结果相媲美。在高性能计算机上，使用混合精度计算也能很大程度上降低电量的使用。

![image-20210521205747527](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/各种数值.png)

**3.1 FP32 权重备份(FP32 MASTER COPY OF WEIGHTS)**  
   FP16相比于FP32具有更好的吞吐量。但相比于FP32数值表示，FP16数值表达的数值范围约为$2^{-24} {\sim} 65504$，远小于FP32的数值表达范围，在实际使用过程中，可能会遇到**溢出错误**和**舍入误差**问题，进而影响到计算的精度。文章作者通过备份FP 32的权重解决了这个问题，具体做法如下图所示，对weights, activations, gradients 的计算使用FP16，对于weights的更新采用FP32精度。 

$$
weights = weights + lr * gradients
$$
        gradients和lr通常会比较小，对于weights的更新会出现舍入误差，若采用FP16精度更新可能会因为最小间隔问题导致weights更新无效。

![image-20210522164846855](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/FP32备份.png)

**3.2 损失缩放（Loss Scale）**  
   下图是SSD网络在训练时的梯度分布，67%梯度在FP16精度下的值为0。为了解决这个下溢出的问题，采用Loss Scale方法将梯度整体右移。如果将梯度$\times 8$（相当于右移三位），这样就可以采用FP16表示$2^{-27}$的数。具体的做法是在计算的loss上进行缩放，根据**反向传播过程中的链式法则**，这个缩放因子会作用于每个梯度。在更新weights时，将缩放因子取消即可。
    
    	在作者的实验中，**采用$\times 8$缩放可以达到使用FP32精度相同的结果**。进一步证明了，模型**参数存在着大量的冗余**。

<img src="https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/梯度问题.png" alt="image-20210522171216814" style="zoom:67%;" />

**3.3 算术精度(ARITHMETIC PRECISION)**  
    在混合计算过程中，NVIDIA的解决方案为，**利用FP16进行乘法和存储，利用FP32来进行加法计算**。 这么做的原因主要是为了减少加法过程中的舍入误差，保证精度不损失。常见的计算场景有下面三种。

1. **在向量乘法中**，将向量乘积以FP32累加，写入内存之前将其转换为FP16  
2. **在Reduction操作中**，读写采用FP16精度，计算采用FP32  
3. **在Point-wise操作中**，内存带宽有限，算术精度不是速度受限的主要因素，FP16或者FP32都可以使用。  

<img src="https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/NVIDIA解决方案.png" alt="image-20210522180341889" style="zoom:67%;" />



### A Mixed-Precision Fused Multiply and ADD (ACSSC 2012,Nicolas Brunie  et al.)

[论文链接](https://ieeexplore.ieee.org/abstract/document/6189977)

​        提出Mixed-Precision FMA 算子，采用binary k 和 binary 2k两种精度计算 $R = AB + C$ , 并证明了采用两种精度计算的合理性。经过查阅C++文档后，从C++  11开始，在标准库中已经实现了相应的算子 [fma docs](https://en.cppreference.com/w/cpp/numeric/math/fma)。要计算$R = AB + C$，在早期的CPU中先做乘法再做加法,该过程需要进行两次rounding，后来CPU中实现了一个新的算子FMA(测试了一下，确实是有了)，可一步完成这个运算，只需进行一次rounding，计算更快精度也更高。

附上作者在文中给出的一段可行性证明

```
float A[],B[] // binary 32
double C // binary 64

C = 0
for(i = 0;i < N;I++)
	C = C + A[i]*B[i]
```

进行计算时，有两种方法。将A，B转换成double，进行binary 64精度的计算，或者先计算float精度乘法，再将结果转换成double。

可行性证明：

1. 将float转换成double 没有错误

2. float有效位是24位，double 有效位是53位，所以double可以存下计算的结果

3. 不会出现上溢出或者下溢出

   

##### 个人实验：

​		按照文章给的算法步骤，自己用C++代码验证了一下。将float A[] ， B[]转换成double后计算所得结果和C++标准库中的fma所得结果一样，但在N = 1000000的数据规模下，标准库的fma执行速度是自己实现的no_fma_mixed_precision()函数的3倍左右，推测是C++ fma在 CPU 中有对应的汇编指令实现，而自己代码实现的计算结果相同的no_fma_mixed_precision()函数缺乏对应硬件的支持。同时由混合精度的no_fma_mixed_precision函数,float精度的no_fma_float函数和double精度的no_fma_double三个函数的执行时间可以看出来，no_fma_mixed_precision()和no_fma_float()程序执行时间相差无几且都远快于no_fma_double()。精度方面采用混合精度计算的结果和采用全double计算得出的结果相差0.01左右，但是fma单次rounding相比于先相乘在相加的两次rounding产生的误差没有从实验结果中体现出来。

![image-20210608152551052](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/混合计算结果.png)

<img src="https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/混合精度fma.png" alt="image-20210608152939630" style="zoom:50%;" />



### Optimizing the Fast Fourier Transform Using Mixed Precision on Tensor Core Hardware (HiPCW 2018 ，Anumeena Sorna et al.)

[论文链接](https://ieeexplore.ieee.org/abstract/document/8634417)

​         提出一种利用动态分裂的混合精度算法，利用NVIDIA Valta GPU支持的FP 16和FP 32两种精度的数据加速FFT计算（NVIDIA Valta GPU上已经实现了上篇文章中的混合精度FMA）。改进后的FFT算法仅在计算矩阵乘法时采用FP16，在算法其他部分采用FP32计算 。为了尽可能的保持计算精度，采用动态分割的算法。算法主要分为三个部分：

1. 拆分数据。将FP32数据拆分成两个FP16数据的比例和，$FP_{32} = s1\times FP_{16} + s2\times FP_{16}$

2. 混合精度。使用tensor core 进行混合精度运行FFT算法计算拆分后的数据。

3. 合并数据。GPU上计算的结果采用FP32精度进行合并。

   ![image-20210609143531248](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/动态分割.png)

​        论文的一个关键点是动态拆分，将FP32数据拆分成两个FP16数据的比例和，采用缩放向量$s_{1}$ 和 $s_{2}$来降低将FP32数据转换成FP16数据时，出现的误差,具体分为以下5个步骤。

1. 按列计算输入矩阵中的绝对值范数来确定$s_{1}$ ，再将每列除以$s_{1}$ 。

![image-20210609144854060](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/动态step1.png)

2. 将FP32矩阵$x_{fp_{32}(i,j)}$转化成FP16 矩阵$x1_{fp_{16}}$ 。

3. 计算步骤2中由于精度转换产生的误差，并将误差保存为FP32矩阵$x2_{fp_{32}}$。

   ![image-20210609145446559](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/动态step2.png)

4. 采用误差矩阵采用作为输入，采用和计算$s_{1}$相同的过程计算$s2$

   ![image-20210609145819592](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/动态step4.png)

5. 将FP32残差矩阵$x2_{fp_{32}(i,j)}$转化成FP16 矩阵$x2_{fp_{16}}$ 。



#####   本文算法要达到FFT加速的效果，有一个重要的前提条件就是：花费在数据拆分和合并上的时间开销，小于Tensor Core提供的加速比。

  论文中从精度、速度两个方面对比了cuBLAS库中实现的FFT算法。误差评价指标计算方式如下所示。


​                     $$ Error  =  \quad { \sum| Result - BaselineResult |\over InputRange} $$


![image-20210609151007987](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/GPUFFT实验结果.png)

实验结果如上面三个图所示。在图1中，cuFFT采用16位进度计算的结果Error远大与本文算法计算的结果，证明了本文算法相比于单一FP16在精度上的改进。图2表示了本文算法在拆分、合并两个阶段在Input Size增大时，所耗费的时间比例很小，可能具有加速效果。图3中对比了本文算法实现和cuFFT库中FFT算法的执行时间，可以看出来已经高度优化的cuFFT计算速度还是快于本文算法实现。



##### 个人心得：

​        本文实验中Tensor Core 实现的FFT算法虽然在执行时间上并没有超过现有的cuFFT中FFT算法实现，但是提出了一种利用GPU Valta架构优化FFT算法的思路个人感觉，本文在动态分割步骤中计算的$s_{1}$和$s_{2}$的过程可以在程序执行之前采用一定的预处理方式提前计算出来，并在后续的程序中使用预处理出来的$s_{1}$和$s_{2}$来替代程序运行过程中对$s_{1}$和$s_{2}$的计算，采用这种预处理的方式会降低精度，但能对FFT的执行速度能起到进一步的加速作用。



### A mixed precision semi-Lagrangian algorithm and its performance on accelerators (HPCS 2016 Lukas Einkemmer)

[论文链接](https://ieeexplore.ieee.org/abstract/document/7568318)

​	    这篇文章的**Introduction部分感觉写的很好**。混合精度计算在线性代数算法中有很大的研究空间。在迭代计算方法中，可以采用单精度计算近似解必要时采用双精度算法对运算结果进行细化。也可在预处理阶段采用单精度算法进行计算，来提高算法整体的速度表现。也有对于自动化混合精度的研究，即将算法程序的某些部分自动转换成单精度。
​         在混合精度方面，文章中对于重要的系数采用双精度存储对于重要程度低的系数采用单精度格式存储。

![image-20210611094201963](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/拉格朗日.png)



### AMPT-GA: automatic mixed precision floating point tuning for GPU applications (ICS 2019 Pradeep V Kotipalli)

[论文链接](https://dl.acm.org/doi/abs/10.1145/3330345.3330360)

​        这篇文章是介绍自动化精度的文章，将整个程序作为一个整体来评估性能，而不是针对程序中某一部分的算法进行改进。论文的主要思想是采用改进的遗传算法(GA)来搜索出一个在精度允许范围内，对浮点数变量执行计算效率最高（不一定是全局最优）的精度向量(PA)。对于PV性能得分采用时间周期衡量，比如一个时钟周期计算一个FP64数据，2个FP32数据，一个强转换耗费4个时钟周期。文章采用有向无环图（DAG）来分析程序的执行步骤，采用变量分组的方式来避免过大的搜索域同时减少数据类型强制转换造成的时间消耗，采用遗传算法中的变异和交叉重组来跳出一些局部最优的情况，采用Execution Filtering策略来过滤掉大部分没有形成当前最优解的情况。
算法步骤图下图所示，其中蓝色的部分是文章中程序进行混合精度优化的地方。

![image-20210610232329750](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/算法步骤.png)

​         从输入输出的角度来看，用户向AMPT-GA提供程序源代码，定义测试输入，定义如何测量误差以及目标阈值。AMPT-GA输出满足精度要求的所有浮点变量的精度向量，为用户提供混合精度配置参考方案。

​		文章实验在LULESH程序上进行，实验结果如下图，Fig. 3中在T = 0.65的case中，AMPT-GA效果没有Precimonious好，作者说这个原因可能是AMPT-GA评估方法的不精确导致。在Fig. 4中可以很明显的看见加了Execution Filtering的AMPT-GA方法的执行次数最少。其他几个图依次说明了，AMPT-GA相比于其他两种方法效率更高，在大量数据上收敛更快，满足误差约束的情况，以及AMPT-GA方法的置信度。

![image-20210610234053453](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/自动混合精度.png)

​        最后作者在LavaMD,Backprop,CFD三种算法上进行了测试，AMPT-GA算法在Backprop和CFD两种算法的大部分情况在FOM最高或接近最高的情况下，达到了最高效率。作者认为，AMPT-GA在LavaMD中FOM低于Precimonious的原因是，LavaMD中浮点变量较少，算法都能充分探索搜索空间发挥不出来AMPT-GA的优势。

##### 个人心得：

​        这篇文章的思路很好，提供了一种有效的自动化混合精度的方法。这篇文章存在着许多可以进行改进的地方，比如说现阶段该方法不会对循环计数进行分析，对于程序的分析也是静态分析没有进行动态分析，搜索域的减小可以提高程序的搜索效率但在本文的问题场景中会导致漏掉较优情况的问题。个人感觉这种针对特定测试用例进行优化的方法可能存在过拟合的问题，即AMPT-GA搜索出来的精度向量在改变了输入测试用例后的表现可能会大幅下降。

### Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers(SC 2018 Azzam Haidar)

[论文链接](https://ieeexplore.ieee.org/abstract/document/8665777)

​        这篇文章研究和实现了GPU Tensor Core 加速的高性能混合精度迭代求精算法框架，研究了不同类型矩阵的结果和分析，文章相关代码工作在MAGMA库发布。总的来说，这篇文章进行了非常充分的实验，对工程实践有很重要的参考价值。

​		文章考虑了两种从低精度因子分解中提取高精度解的方法，一种是$IR$，一种是$IRGM$。采用迭代求精(Iterative Refinement)方法提高计算$Ax = b$ 的精度时，迭代步骤为以下三步:

1. 计算残差（Residual）: $r = b - Ax_{i}$
2. 计算修正 （Correction）：求解$Ac = r$
3. 更新 （update）: 采用$x_{i+1} = x_{i} + c$来修正现有解

​        采用混合精度的算法步骤如下图所示，在LU 分解的时候，采用低精度进行，计算残差和更新解的部分采用高精度计算以保证结果的精度。目前也有人在研究三精度甚至四精度的$IR$。

![image-20210610151923488](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/IRstep.png)


​        这篇文章进行了充分的实验，作者在不同类型的矩阵上测试了算法的数值行为，作者在实验中发现，**当迭代次数超过200次的时候**，算法的性能会大幅下降。   

![image-20210610151923488](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/矩阵类型.png)

​        由于实验结果太多了就不贴在文章里了，总的来说，作者在实验过程中得到两个Lesson ，FP16-TC相比于FP16跟稳定也更快，效果会更好，迭代求精算法能起到很好的加速效果 。

![image-20210610152726753](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/papers/Mixed%20precision/images/lesson.png)

   文章最后希望能在未来能够实现多GPU环境下实现4倍的加速和对应的能效提高。

##### 个人心得：

​         这篇文章进行了充分的实验，在科研和工程领域都具有很高的借鉴价值，目前没有完全消化掉这篇文章的内容，如果以后有机会能    进行混合精度的研究，这篇文章一定要再拿出来吃透。



### 个人思考：

​		设计适用于GPU硬件架构的算法在HPC中是一项很有意义的事情。Tensor Core主要运用在深度学习中，较低的精度是可以容忍的，但是数值计算等对精度要求较高的应用不适合直接将算法移植到GPU上利用Tensor Core和FP 16 混合精度加速。目前很多混合精度相关的工作都是针对某一特定的算法进行混合精度的改进, 对于FFT这类应用面很广的算法进行定制化的改进意义重大，但在数值计算领域中各种各样的算法太多，要是逐一混合精度的设计是一项工作量很大的研究，但是目前没有统一的混合精度框架适用于对大部分算法的改进，只能依靠在实践中积累的经验或其他人工作中探索出的混合精度设计来对某一算法进行改进。经验性的东西可一定程度上被人工智能算法替代，数值计算算法、应用程序或许可以拆分成多个步骤，采用一定的智能算法来评估每一部分算法采用怎么样的精度计算以达到较优的速度- 精度平衡，也不一定要局限在双精度，引入更多种的精度可能也是一个可以探索的方向 , 强化学习方法可能是一条可行的道路。目前强化学习在NN量化和模型混合精度推理领域已经有了一些研究,也取得了不错的试验效果，但是HPC、数值计算对数据精度要求较高，这也是与NN一个很大的不同，这个区别决定了NN领域的一些混合精度的做法不一定能很好的移植到HPC领域。

1. [HAQ: Hardware-Aware Automated Quantization With Mixed Precision](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html)，CVPR 2019

   模型量化是压缩和加速深度神经网络（DNN）推理的一种广泛使用的技术，本文利用强化学习来自动确定量化策略，并且在设计循环中获取了硬件加速器的反馈，可以在不同的资源约束下，不同硬件架构上工作。

2. [Mixed Precision Quantization for ReRAM-based DNN Inference Accelerators](https://ieeexplore.ieee.org/abstract/document/9371610/keywords#keywords)，ASPDAC 2021

   文章采用强化学习方法为DNN 推理提出了一种混合精度量化方案。 采用深度强化学习方法，在大型设计空间中搜索最佳量化配置。 

3. [ReLeQ: An Automatic Reinforcement Learning Approach for Deep Quantization of Neural Networks](https://par.nsf.gov/servlets/purl/10111602) ，NeurIPS ML for Systems workshop, 2018

4. [Rethinking Differentiable Search for Mixed-Precision Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Cai_Rethinking_Differentiable_Search_for_Mixed-Precision_Neural_Networks_CVPR_2020_paper.html) ，CVPR2020
