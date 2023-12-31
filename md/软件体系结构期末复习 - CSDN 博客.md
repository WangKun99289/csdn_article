> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_56110281/article/details/131353531?ops_request_misc=&request_id=&biz_id=102&utm_term=%E8%BD%AF%E4%BB%B6%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84%E6%9C%9F%E6%9C%AB%E5%A4%8D%E4%B9%A0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-131353531.nonecase&spm=1018.2226.3001.4187)

#### 文章目录

*   [第一章](#_1)
*   *   [1、软件危机产生的原因](#1_3)
    *   [2、软件重用](#2_10)
    *   [3、青鸟构建模型](#3_20)
    *   [4、获取构件的途径](#4_44)
    *   [5、** 构建的分类方法 **](#5_54)
    *   [6、** 软件体系结构发展的四个阶段 **](#6_62)
    *   [7、软件的元素包括](#7_76)
*   [第二章](#_82)
*   *   [1、 软件体系结构模型](#1__84)
    *   [2、“4+1” 视图](#241_96)
    *   [3、软件体系结构的核心模型](#3_101)
    *   [4、软件提信息结构的抽象模型](#4_115)
*   [第三章](#_122)
*   *   [1、体系结构风格是指](#1_124)
    *   [2、常用体系结构风格分类](#2_128)
    *   [3、每种风格的构件和连接件](#3_136)
    *   [4、C2 风格的四条组织规则](#4C2_188)
    *   [5、三层 C/S 体系结构将应用功能分为 == 表示层、功能层、数据层 ==](#5CS_197)
    *   [6、与 C/S 体系结构相比 B/S 结构的不足之处](#6CSBS_199)
    *   [7、基于层次消息总线的体系风格](#7_210)
    *   [8、SIS 互联系统是指什么](#8SIS_219)
    *   [9、层次系统最广泛的应用 书](#9__229)
    *   [10、黑板系统由三部分组成（不展开）](#10_233)
    *   [11、特定领域软件体系结构 DSSA](#11DSSA_239)
*   [第四章](#_282)
*   *   *   [1、软件体系结构的描述方法](#1_284)
    *   [2、软件体系结构描述语言的种类](#2_294)
    *   [3、ADL（体系结构描述语言）的能力与特性，与其他语言的区别](#3ADL_298)
    *   [4、（MIL）模块内连接语言定义（一句话）](#4MIL_308)
    *   [5、C2 风格例子](#5C2_314)
    *   [6、UML 图](#6UML_322)
    *   [7、Wright 语言的特点](#7Wright_339)
*   [第五章](#_343)
*   *   [1、基于构件的动态系统结构模型](#1_345)
    *   [2、分层](#2_349)
    *   [3、动态体系结构的动态性（详细）](#3_358)
    *   [4、CBDSAM 模型定义，分层，更新执行的步骤，局部更新和全局更新（每种更新的五个步骤）](#4CBDSAM_391)
*   [第六章](#_429)
*   *   [1、SOA 关键技术](#1SOA_431)
    *   [2、服务描述语言 WSDL](#2WSDL_436)
    *   [3、统一描述、发现和集成协议 UDDI](#3UDDI_440)
    *   [4、SOAP 消息封装协议](#4SOAP_444)
    *   [5、**Web Service 模型 **](#5Web_Service_451)
*   [英文缩写](#_464)

第一章
---

### 1、软件危机产生的原因

*   **用户需求不明确**
*   **缺乏正确的理论指导**
*   **软件规模越来越大**
*   **程序复杂度越来越高**

### 2、软件重用

软件重用是指 == **在两次或多次不同的软件开发过程中重复使用相同或相近软件元素的过程。**== 软件元素包括程序代码、测试用例、[设计文档](https://so.csdn.net/so/search?q=%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3&spm=1001.2101.3001.7020)、设计过程、需求分析文档甚至领域 (domain) 知识。通常，把这种可重用的元素称作软构件 (software component), 简称为构件可重田的软件元素越大，就说重用的粒度(granularity) 越大。

**优点：**

*   减少软件开发过程中大量的重复性工作，提高生产率，降低成本，缩短开发周期（**减少大量重复性工作，提高效率，降低成本，缩短开发周期**）
*   ** 有助于改善软件质量（** 由于软构件经过严格的质量认证，且在实际的运行环境中得到检验）
*   **提高软件的灵活性与标准化程度**

### 3、青鸟构建模型

![](https://img-blog.csdnimg.cn/a5bc7ad6e4d64ca3bd3f3e4498b4a702.png#pic_center)

*   **外部接口：**
    
    *   **构建名称**
        
    *   **功能描述**
        
    *   **参数化属性**
        
    *   **对外功能接口**
        
    *   **所需的构建**
        
*   **内部接口：**
    

*   **具体成员**
    
*   **虚拟成员**
    
*   **成员关系（内部、外部接口）**
    

### 4、获取构件的途径

*   **从现有构件中获得符合要求的构件，直接使用或作适应性 (flexibility) 修改，得到可重用的构件。**
    
*   **通过遗留工程 (legacy engineering), 将具有潜在重用价值的构件提取出来，得到可重用的构件。**
    
*   **从市场上购买现成的商业构件。**
    
*   **开发新的符合要求的构件。**
    

### 5、**构建的分类方法**

**1. 关键字分类法**

**2. 刻面分类法**

**3. 超文本组织方法**

### 6、**软件体系结构发展的四个阶段**

**纵观软件体系结构技术的发展过程，从最初的 “无结构” 设计到现行的基于体系结构的软件开发，可以认为经历了 4 个阶段：**

**(1)== 无体系结构设计阶段。== 以汇编语言进行小规模应用程序开发为特征**

**(2) 萌芽阶段。出现了程序结构设计主题，以控制流图和数据流图构成软件结构为特征**

**(3) 初期阶段。出现了从不同侧面描述系统的结构模型，以 UML 为典型代表。**

**(4)== 高级阶段。以描述系统的高层抽象结构为中心，不关心具体的建模细节，划分了体系结构模型与传统软件结构的界限，该阶段以 Kruchten 提出的 “4 十 1”== 模型为标志。**

### 7、软件的元素包括

**程序代码、测试用例、设计文档、设计过程、需求分析文档甚至领域知识**

第二章
---

### 1、 软件体系结构模型

**①结构模型（** 最直观普遍。以构件、连接件家等语义概念来刻画结构， 力图通过结构来反映系统的重要的语义内容。研究该模型的核心是体系结构描述语言

**②框架模型**（更侧重于描述整体的结构。它以一些特殊问题为目标建立只针对和适应该问题的结构）

**③动态模型** (对结构或框架的补充，研究系统的 “大颗粒” 的行为性质)

**④过程模型**（研究构造系统的步骤和过程）

**⑤功能模型**（体系结构是由一组功能构件按层次组成，下层向上层提供服务）

### 2、“4+1” 视图

![](https://img-blog.csdnimg.cn/73911fa599ef49d09eb7a5488b44f807.png#pic_center)

### 3、软件体系结构的核心模型

![](https://img-blog.csdnimg.cn/afbdb9f773f74258927a948e3c9c9631.png#pic_center)

⑴== **构件** == 是具有某种功能的可重用的软件模板单元，表示系统中主要的计算元素和数据存储…

⑵== **连接件** == 表示了构件之间的交互，如管道过程调用

⑶== **配置** == 表示了构件和连接件的拓扑逻辑和约束

⑷== **端口** == 即构件的接口，表示了构件和外部环境的交互点

⑸** 角色 ** 即连接件的接口，角色定义了该连接件所表示的交互的参与者

### 4、软件提信息结构的抽象模型

![](https://img-blog.csdnimg.cn/09071f6e98be4d659db368ea3b452a3a.png#pic_center)

第三章
---

### 1、体系结构风格是指

软件体系结构风格是 == **描述某一特定应用领域中系统组织方式的惯用模式** ==

### 2、常用体系结构风格分类

**(1) 数据流风格：** 批处理序列、管道与过滤器。  
**(2) 调用 / 返回风格：** 主程序与子程序、面向对象风格、层次结构。  
**(3) 独立构件风格：** 进程通信、事件系统。  
**(4) 虚拟机风格：** 解释器、基于规则的系统。  
**(5) 仓库风格：** 数据库系统、超文本系统、黑板系统。

### 3、每种风格的构件和连接件

1.  管道过滤风格
    
    *   **构件：****过滤器**
    *   **连接件：****管道**
2.  数据抽象和面向对象系统
    
    *   **构件：****对象**
        
    *   **连接件：过程调用**
        
3.  基于事件的系统
    
    *   **构件：模块（过程、集合）**
        
    *   **连接件：隐式调用**
        
4.  分层系统
    
    *   **构件：每个子层**
        
    *   **连接件：层间协议**
        
5.  仓库系统及知识库
    
    *   **构件：中央数据结构、独立构件**
        
    *   **连接件：控制策略**
        
6.  C2 风格
    

*   **构件：构件**
*   **连接件：连接件**

7.  客户 / 服务器风格

*   **构件：数据库服务器、客户应用程序**
*   **连接件：网络**

8.  浏览器 / 服务器风格

*   **构件：浏览器，web 服务器, 数据库服务器**
*   **连接件：网际交互协议**

9.  公共对象请求代理

*   **构件：应用对象、通用服务、对象服务**
*   **连接件：对象请求代理**

### 4、C2 风格的四条组织规则

(1) 系统中的 == **构件和连接件都有一个顶部和一个底部。**==  
(2) **构件的顶部应连接到某连接件的底部，构件的底部侧应连接到某连接件的顶部，**而构件与构件之间的直接连接是不允许的。  
(3) 一**个连接件可以和任意数目的其他构件和连接件连接。**  
(4) 当两个连接件进行直接连接时，**必须由其中一个的底部到另一个的顶部。**

### 5、三层 C/S 体系结构将应用功能分为表示层、功能层、数据层

### 6、与 C/S 体系结构相比 B/S 结构的不足之处

*   **与 C/S 体系结构相比，B/S 体系结构也有许多不足之处，例如：**
    
    **(1)B/S 体系结构缺乏对动态页面的支持能力，没有集成有效的数据库处理功能。**  
    **(2)B/S 体系结构的系统扩展能力差，安全性难以控制。**  
    **(3) 采用 B/S 体系结构的应用系统，在数据查询等响应速度上，要远远低于 C/S 体系结构。**  
    **(4)B/S 体系结构的数据提交一般以页面为单位，数据的动态交互性不强，不利于在线**  
    **事务处理 (OnLine Transaction Processing,OLTP) 应用。**
    

### 7、基于层次消息总线的体系风格

![](https://img-blog.csdnimg.cn/bd3075a8fffe4035befaffae2e08351b.png#pic_center)

**消息总线是系统的连接件，负责消息的分派传递和过滤以及处理结果的返回。**各个构件挂接在消息总线上，向总线登记感兴趣的消息类型。构件根据需要发出消息，由消息总线负责把该消息分派到系统中所有对此消息感兴趣的构件，**消息是构件之间通信的唯一方式。**构件接收到消息后，根据自身状态对消息进行响应，并通过总线返回处理结果。由于构件通过总线进行连接，并不要求各个构件具有相同的地址空间或局限在一台机器上。**该风格可以较好地刻画分布式并发系统**，以及基于 CORBA、DCOM 和 EJB 规范的系统。

如图 3-22 所示，系统中的 == **复杂构件可以分解为比较低层的子构件** ==，这些子构件通过局部消息总线进行连接，这种复杂的构件称为 ** 复合构件。** 如果子构件仍然比较复杂，可以进一步分解，如此分解下去，整个系统形成了树状的拓扑结构，== 树结构的末端结点称为叶结点，它们是系统中的原子构件，== 不再包含子构件，原子构件的内部可以采用不同于 HMB 的风格，例如前面提到的数据流风格、面向对象风格及管道 - 过滤器风格等，这些属于构件的内部实现细节。但要集成到 HMB 风格的系统中，必须满足 HMB 风格的构件模型的要求，主要是在接口规约方面的要求。另外，整个系统也可以作为一个构件，通过更高层的消息总线，集成到更大的系统中。于是，可以采用统一的方式刻画整个系统和组成系统的单个构件。

### 8、SIS 互联系统是指什么

SIS 是指 == **系统可以分成若干个不同的部分，每个部分作为单独的系统独立开发。**== 整个  
系统通过一组互连系统实现，而互连系统之间相互通信，履行系统的职责。**其中一个系统体**  
**现整体性能，称为上级系统 (superordinate system); 其余系统代表整体的一个部分，称为从**  
** 属系统 (subordinate system)。** 从上级系统的角度来看，从属系统是子系统，上级系统独立  
于其从属系统，每个从属系统仅仅是上级系统模型中所指定内容的一个实现，并不属于上级  
系统功能约束的一部分。互连系统构成的系统的软件体系结构（Software Architecture for  
SIS,SASIS) 如图 3-30 所示。

### 9、层次系统最广泛的应用 书

层次系统最广泛的应用是 == **分层通信协议** ==

### 10、黑板系统由三部分组成（不展开）

*   **知识源**
*   **黑板数据结构**
*   **控制**

### 11、特定领域软件体系结构 DSSA

![](https://img-blog.csdnimg.cn/9ea4ac44c61442cd851e90b0301e287f.png#pic_center)

认识每种体系结构的示意图

![](https://img-blog.csdnimg.cn/d6d9e30852584f44a4d221a291cdf190.png#pic_center)

![](https://img-blog.csdnimg.cn/91053a24295a4b288ab8fa0f3910f96a.png#pic_center)

![](https://img-blog.csdnimg.cn/ad9c1e793c7a49ae8679e978b38e5198.png#pic_center)

![](https://img-blog.csdnimg.cn/b6c20b400759446e8e41b2efe4d945e2.png#pic_center)

![](https://img-blog.csdnimg.cn/d2b1e65709ff425883bd9896802900dd.png#pic_center)

![](https://img-blog.csdnimg.cn/2b3eb753dddf4896b4261c0219bb63b3.png#pic_center)

![](https://img-blog.csdnimg.cn/6fbab92a969c4e76966f41f0fb5c9cbd.png#pic_center)

![](https://img-blog.csdnimg.cn/3c37ec8b030e4d6da8afc567e8edb42f.png#pic_center)

![](https://img-blog.csdnimg.cn/ebc4c4bf74414a6cb34dec36fcca4198.png#pic_center)

![](https://img-blog.csdnimg.cn/971eddd93eba42848b0fa2bc602a0805.png#pic_center)

![](https://img-blog.csdnimg.cn/84e97cc998514325b4173b9da3be0fa2.png#pic_center)

![](https://img-blog.csdnimg.cn/0e6ee6b0ad2d4aa9965760f6da1de561.png#pic_center)

第四章
---

#### 1、软件体系结构的描述方法

*   **图形表达工具**
    
*   **模块内连接语言**
    
*   **基于软构件的系统描述语言**
    
*   **软件体系结构描述语言**
    

### 2、软件体系结构描述语言的种类

主要的体系结构描述语言有 Aesop、MetaH、C2、Rapide、SADL、Unicon 和 Wright 等（认识即可）

### 3、ADL（体系结构描述语言）的能力与特性，与其他语言的区别

*   **构造能力 抽象能力 重用能力 组合能力 异构能力 分析和推理能力**

1.  ADL 与需求语言：需求语言描述问题空间，ADL 根植于解空间
    
2.  ADL 与建模语言：建模语言对整体行为的关注大于对部分的关注，ADL 集中在构件的表示上
    
3.  ADL 与传统的程序设计语言：有许多相同和相似之处，又各自有着很大的不同。（求同存异）
    

### 4、（MIL）模块内连接语言定义（一句话）

软件体系结构的第二种描述和表达方法是采用将一种或几种传统程序设计语言的模块连接起来的模块内连接语言（Module Interconnection Language，MIL）。

定义：**采用将一种或几种传统程序设计语言的模块连接起来的语言**。

### 5、C2 风格例子

![](https://img-blog.csdnimg.cn/5d2cdc1d8db04cec9116213b0b045edd.png#pic_center)

![](https://img-blog.csdnimg.cn/c6b611bbdf6a4696aee929beadc0da61.png#pic_center)

### 6、UML 图

![](https://img-blog.csdnimg.cn/9b5312447df54f7ca8a177a0c7a7b901.png#pic_center)

![](https://img-blog.csdnimg.cn/2e6e867dbac64fd28132947e69ae01f3.png#pic_center)

![](https://img-blog.csdnimg.cn/28b8631cf9924bf3a1c24163dfe433ad.png#pic_center)

![](https://img-blog.csdnimg.cn/50f477dde5044531b912cbc7bf42d8ac.png#pic_center)

![](https://img-blog.csdnimg.cn/77b70a654086453f92cb1037e69878c3.png#pic_center)

### 7、Wright 语言的特点

Wright 的主要特点为：对体系结构和抽象行为的精确描述、定义体系结构风格的能力和一组对体系结构描述进行一致性和完整性的检查。体系结构通过构件、连接件以及它们之间的组合来描 述，抽象行为通过构件的行为和连接件的胶水来描述。

第五章
---

### 1、基于构件的动态系统结构模型

CBDSAM (Component Based Dynamic System Architecture Model)

### 2、分层

![](https://img-blog.csdnimg.cn/39d0ddca0efb489aae101e5e80a137ba.png#pic_center)

*   应用层处于最底层，包括构件连接构件接口和执行。构件连接定义了连接件如何与构件相连接；构件接口说明了构件提供的服务，例如消息、操作和变量等。在这一层，可以添加新的构件、删除或更新已经存在的构件。
*   中间层包括连接件配置构件配置，构件描述和执行。连接件配置主要是管理连接件及接口的通信配置；构件配置管理构件的所有行为；构件描述对构件的内部结构、行为、功能和版本等信息加以描述。在这一层，可以添加版本控制机制和不同的构件装载方法。
*   体系结构层位于最上层，控制和管理整个体系结构，包括体系结构配置、体系结构描述和执行。其中，体系结构描述主要是描述构件以及它们相联系的连接件的数据；体系结构配置轻制整个分布式系统的执行，并且管理配置层；体系结构描述主要对体系结构层的行为进行描述。在这一层，可以更改和扩展更新机制，更改系统的拓扑结构，以及更改构件到处理元素之间的映射。

### 3、动态体系结构的动态性（详细）

**包括三类：**

1.  **交互式动态性**
2.  **结构化动态性**
3.  **体系结构动态性**

*   **可构造动态性特征**
    
    构造性动态特征通常可以通过结合动态描述语言、动态修改语言和一个动态更新系统来实 现。动态描述语言用于描述应用系统软件体系结构的初始配置；当体系结构发生改变的时候，这种 改变可以通过动态修改语言进行描述，该语言支持增加或删除、激活或取消体系结构元素和系统遗 留元素；而动态更新可以通过体系结构框架或者中间件实现。可构造性动态特征如图 7-11 所示。
    

![](https://img-blog.csdnimg.cn/b3cb0383ffdd4e14b6d9643e54ce5a6a.png#pic_center)

*   **适应性动态特征**
    
    适应性动态特征 某些应用程序必须有立即反映当前事件的能力，此时程序不能进行等待，必须把初始化、选择 和执行整合到体系结构框架或中间件里面。 适应性动态特征是基于一系列预定义配置而且所有事件在开发期间已经进行了评估。执行期 间，动态改变是通过一些预定义的事件集触发，体系结构选择一个可选的配置来执行系统的重配 置。如图 7-12 描述了由事件触发改变的适应性动态特征。
    

![](https://img-blog.csdnimg.cn/1edf12441dbc4038bda2026a9eb0ad7b.png#pic_center)

*   **智能性动态特征**

智能性动态特征 智能性动态特征是用一个有限的预配置集来移除约束。如图 7-13 所示，它描述的是一个具有智 能性动态特征的应用程序体系结构

![](https://img-blog.csdnimg.cn/03682d91eb514071abf802b8a1ca68d3.png#pic_center)

对比适应性体系结构特征，智能性体系结构特征改善了选择转变的功能，适应性特征是从一系 列固定的配置中选择一个适应体系结构的配置，而智能性特征是包括了动态构造候选配置的功能。 但是由于智能特征的复杂性，在实际的软件体系结构中并不是太多的系统能够用到这种方法。

### 4、CBDSAM 模型定义，分层，更新执行的步骤，局部更新和全局更新（每种更新的五个步骤）

> 基于构件的动态系统结构模型（Component Based Dynamic System Architecture Model， CBDSAM）支持运行系统的动态更新，该模型分为三层，分别是应用层、中间层和体系结构层。

*   **更新执行步骤**
    
    *   **== 检测更新的范围：== 局部、全局;**
        
    *   **== 更新准备工作：== 应用层需要等待参与进程发出可安全更新信号; 配置层需要等待连接件其他构件已完成更新;**
        
    *   **== 执行更新：== 告知发起者更新的结果;**
        
    *   **== 存储更新：== 将所作的更新存储到构件或体系结构描述中。**
        

![](https://img-blog.csdnimg.cn/064451faf6bc407282e92c72ad2635b7.png#pic_center)

![](https://img-blog.csdnimg.cn/f542027d9c4e4dfbb995b38b87445778.png#pic_center)

*   **局部更行**
    
    局部更新由于只作用于需要更新的构件内部，不影响系统的其它部分，因此比全局更新要简单
    
    *   第一步，更新发起者发出一个更新请求，这个请求被送到构件 A 的配置器中，构件配置器将分析更新的类型，从而判断它是对象的局部更新。
    *   第二步，由于更新为局部更新，构件 A 的配置器发出一个信号给连接件以隔离构件 A 的通信，准备执行更新。
    *   第三步，构件 A 的配置器开始执行更新。
    *   第四步，更新执行完毕后，构件 A 的构件描述被更新，并且构件 A 发送一个消息给连接件 B，两者间的连接被重新存储起来。
    *   第五步，将更新结果返回给更新发起者。
*   **全局更行**
    
    *   第一步，Server 构件配置器接收到更新发起者提出的更新请求后，向体系结构配置器提出更新 请求。
    *   第二步，体系结构配置器对更新请求的类型进行分析，判断是否在更新请求限制（属于全局更 新还是局部更新）范围内，不在更新范围内的更新不予执行；如果在更新限制范围内，体系结构配 第 7 章：动态软件体系结构 πADL 动态体系结构 第 7 章：动态软件体系结构 动态体系结构的描述 第 7 章：动态软件体系结构 动态体系结构的特征 置器对更新所涉及的连接件和构件（本例中为 Client 构件和连接件）发出消息，要求它们做好更新准 备工作。
    *   第三步，准备工作完成后，Client 构件配置器和连接件向体系结构配置器返回就绪信息。
    *   第四步，一切准备就绪后，体系结构配置器通知 Server 构件进行更新。
    *   第五步，更新执行完毕后，向 Server 构件配置器、体系结构配置器和更新发起者通知更新执行 完毕并返回更新结果；同时，体系结构配置器通知 Client 构件和连接件更新结束，可继续正常运行。 这样，在没有影响系统运行的情况下，按照更新发起者的要求对系统进行了更新，并维护了系 统的一致性

第六章
---

### 1、SOA 关键技术

![](https://img-blog.csdnimg.cn/6f7212a6f58f43d7bb9b6351b169fe8b.png#pic_center)

### 2、服务描述语言 WSDL

WSDL 是对服务进行描述的语言，它有一套基于 XML 的语法定义。WSDL 描述的重点是服务，它包含 Service Implementation Definition（服务实现定义）和 Service Interface Definition（服务接口定义）。

### 3、统一描述、发现和集成协议 UDDI

UDDI 是一种用于描述、发现、集成 Web 服务的技术，它是 Web 服务协议栈的一个重要部分。通过 UDDI, 企业可以根据自己的需要动态查找并使用 Wb 服务，也可以将自己的 Web 服务动态地发布到 UDDI 注册中心，供其他用户使用。

### 4、SOAP 消息封装协议

1.  ==SOAP 封装结构。== 定义一个整体框架用来表示消息中包含什么内容，谁来处理这些内容，以及这些内容是可选的或是必需的。
2.  ==SOAP 编码规则。== 用以交换应用程序定义的数据类型的实例的一系列机制。
3.  ==SOAP RPC 表示。== 定义一个用来表示远程过程调用和应答的协议。
4.  ==SOAP 绑定。== 定义一个使用底层传输协议来完成在结点间交换 SOAP 信封的约定。

### 5、**Web Service 模型**

![](https://img-blog.csdnimg.cn/6cbe356a1877407ea618ae925537cb64.png#pic_center)

Web Service 模型中的操作包括 == 发布、查找和绑定，== 这些操作可以单次或反复出现。

*   发布。**为了使用户能够访问服务，服务提供者需要发布服务描述，以便服务请求者可以查找它。**
*   查找。** 在查找操作中，服务请求者直接检索服务描述或在服务注册中心查询所要求的服务类型。** 对服务请求者而言，可能会在生命周期的两个不同阶段中涉及查找操作，首先是在设计阶段，为了程序开发而查找服务的接口描述；其次是在运行阶段，为了调用而查找服务的位置描述。
*   绑定。== **在绑定操作中，服务请求者使用服务描述中的绑定细节来定位、联系并调用服务，从而在运行时与服务进行交互。**== 绑定可以分为动态绑定和静态绑定。在动态绑定中，服务请求者通过服务注册中心查找服务描述，并动态地与服务交互；在静态绑定中，服务请求者已经与服务提供者达成默契，通过本地文件或其他方式直接与服务进行绑定。
*   服务提供者。服务提供者是服务的所有者，该角色负责定义并实现服务，使用 WSDL 对服务进行详细、准确、规范的描述，并将该描述发布到服务注册中心，供服务请求者查找并绑定使用。
*   服务请求者。服务请求者是服务的使用者，虽然服务面向的是程序，但程序的最终使用者仍然是用户。从体系结构的角度看，服务请求者是查找、绑定并调用服务，或与服务进行交互的应用程序。服务请求者角色可以由浏览器来担当，由人或程序（例如，另外一个服务）来控制。
*   服务注册中心。服务注册中心是连接服务提供者和服务请求者的纽带，服务提供者在此发布他们的服务描述，而服务请求者在服务注册中心查找他们需要的服务。不过，在某些情况下，服务注册中心是整个模型中的可选角色。例如，如果使用静态绑定的服务，服务提供者则可以把描述直接发送给服务请求者。

英文缩写
----

1.UML : 统一建模语言（Unified Modeling Language，UML）

2.XSL : 可扩展样式语言（Extensible Style Language，XSL）

3.CBDSAM：基于构件的动态系统结构模型（Component Based Dynamic System Architecture Model， CBDSAM）

4.UDDI：一种发现服务层的规范 （UDDI 是 “统一描述、发现和集成协议” 的缩写，用于在网络上注册、发现和集成 Web 服务的协议和规范 gpt）

5.WSDL：一种描述服务层的规范（WSDL 是 “Web 服务描述语言”（Web Services Description Language）的缩写，是一种用于描述 Web 服务接口和其可用操作的 XML 格式标准。 gpt）

6.CSP：通信顺序进程

7.DSSA ：特定领域的软件体系结构（Domain Specific Software Architecture， DSSA）

8.HMB：层次消息总线（Hierarchy Message Bus，HMB）

9.PCL：多变配置语言（Proteus Configuration Language，PCL）

10.SOA：面向服务的体系结构（Service-Oriented Architecture，SOA）

11.OMG：是 OMG（Object Management Group，对象管理组织）

12.CORBA ：CORBA （Common Object Request Broker Architecture，通用对象请求代理结构）

13.API：及 API（Application Programming Interface，应用编程接口）

14.OOD：（Object-Oriented Design，面向对象的设计）

15.OLTP：在线事务处理（OnLine Transaction Processing，OLTP）

16.SIS：互连系统构成的系统（System of Interconnected Systems, SIS）

17.UML： 统一建模语言（Unified Modeling Language，UML）

18.XLL ： 可扩展链接语言 （Extensible Linking Language，XLL）

19.SOAD： 面向服务的分析与设计（Service-Oriented Analysis and Design，SOAD）

（Common Object Request Broker Architecture，通用对象请求代理结构）

13.API：及 API（Application Programming Interface，应用编程接口）

14.OOD：（Object-Oriented Design，面向对象的设计）

15.OLTP：在线事务处理（OnLine Transaction Processing，OLTP）

16.SIS：互连系统构成的系统（System of Interconnected Systems, SIS）

17.UML： 统一建模语言（Unified Modeling Language，UML）

18.XLL ： 可扩展链接语言 （Extensible Linking Language，XLL）

19.SOAD： 面向服务的分析与设计（Service-Oriented Analysis and Design，SOAD）

20.DSA：动态软件体系结构（Dynamic Software Architecture，DSA）