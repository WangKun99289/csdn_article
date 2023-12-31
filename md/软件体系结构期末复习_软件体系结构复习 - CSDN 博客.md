> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/Deeper_blue123/article/details/131209809)

概述
--

### 软件危机的表现

软件成本日益增长

开发进度难以控制

[软件质量](https://so.csdn.net/so/search?q=%E8%BD%AF%E4%BB%B6%E8%B4%A8%E9%87%8F&spm=1001.2101.3001.7020)差

软件维护困难

### [软件体系结构](https://so.csdn.net/so/search?q=%E8%BD%AF%E4%BB%B6%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84&spm=1001.2101.3001.7020)概念

软件体系结构由一定形式的结构化元素组成，即是**构件的集合**。

包括处理构件、数据构件和连接构件

**处理构件**负责加工数据

**数据构件**代表被加工的信息

**连接构件**则负责组合连接不同的构件

基于视图的体系结构建模
-----------

**架构元素**–软件或硬件中存在的一组实际元素

**视图**–由系统相关者编写和读取的一组连贯的架构元素的表示

视图是一组架构元素及其关联关系的表示

但并非涵盖所有架构元素。

视图绑定了体系结构描述时感兴趣的元素类型和关系类型，并显示了它们。

#### “4+1” 的视图模型

Kruchten 在 1995 年提出了 “4+1” 的视图模型。

“4+1” 视图模型从 5 个不同的视角包括逻辑视图、进程视图、物理视图、开发视图和场景视图来描述软件体系结构。

每一个视图只关心系统的一个侧面，5 个视图结合在一起才能反映系统的软件体系结构的全部内容。

![](https://img-blog.csdnimg.cn/449f675409a740a9b41d56924189bc53.png) 

![](https://img-blog.csdnimg.cn/907fc6b7c7fa424e926a15fc0f2bfa49.png)

[用例图](https://so.csdn.net/so/search?q=%E7%94%A8%E4%BE%8B%E5%9B%BE&spm=1001.2101.3001.7020)：从用户角度描述系统功能。

类图：描述系统中类的静态结构。

对象图：系统中的多个对象在某一时刻的状态。

状态图：是描述状态到状态控制流，常用于动态特性建模

活动图：描述了业务实现用例的工作流程

顺序图：对象之间的动态合作关系，强调对象发送消息的顺序，同时显示对象之间的交互

协作图：描述对象之间的协助关系

构件图：一种特殊的 UML 图来描述系统的静态实现视图

部署图：定义系统中软硬件的物理体系结构

体系结构风格
------

![](https://img-blog.csdnimg.cn/c3427a05c1984b12b6553b70869f2704.png)

### 数据流风格

#### 批处理

优点：

提供了更简单的子系统划分。每个子系统可以是独立的程序，处理输入数据并生成输出数据。

缺点：

提供了较高的延迟和较低的吞吐量。不提供并发性和交互式界面。实现需要外部控制。

#### 管道 - 过滤器

优点：

为大量数据处理提供并发性和高吞吐量。提供可重用性并简化系统维护。提供可修改性和低耦合度之间的过滤器提供简单性，通过管道连接的任何两个过滤器之间提供明确的划分。提供灵活性，支持顺序和并行执行。

缺点：

不适用于动态交互。过滤器之间的数据转换开销大。不提供过滤器合作交互以解决问题的方式。难以在动态环境下配置这种体系结构。

#### 过程控制

使用循环结构来控制环境变量。

### 调用 - 返回风格

#### 主程序 - 子程序

优点：

促进模块化和函数重用

黑箱（功能的封装）允许库函数轻松地集成到程序中。

缺点：

扩展性差

子程序可能会以意想不到的方式更改数据。

在执行期间，子程序可能会受到其他子程序进行的数据更改的影响。

最适合计算集中的小型系统。

#### 面向对象

优点：

面向对象的架构将应用程序映射到现实世界的对象，使其更加易于理解。

由于程序的重用，它易于维护并提高系统质量。

这种架构通过多态和抽象提供了可重用性。

它能够在执行过程中管理错误（健壮性）。

它能够扩展新功能而不影响系统。

它通过封装提高了可测试性

面向对象的架构可以减少开发时间和成本。

缺点：

面向对象的架构很难确定系统所需的所有必要类和对象。

由于面向对象的架构提供了新的项目管理方式，因此很难在预估的时间和预算内完成解决方案。

在没有明确的重用程序的情况下，这种方法可能无法在大规模上成功重用。

#### 层次结构

优点：

用户可以执行复杂任务而无需了解底层的层次结构。

不同层可以在不同级别的授权或特权下运行。通常，顶部层被认为是 “用户空间”，没有权限分配系统资源或直接与硬件通信。这种“沙盒化” 提供了内核的安全性和可靠性。

由于分层架构遵循最小知识原则，设计将更加松散耦合。

它可以直观而强大。许多组织和解决方案都有分层结构，因此很容易在适当的地方应用分层架构。

它支持关注点的分离，因为每个层都是一组具有类似责任或目的的组件。

它是模块化和松散耦合的，因为每个层仅与一个或两个其他层通信，可以轻松地交换不同的实现。

它可以适应分层不总是严格的，这有助于管理设计复杂性或提供系统结构的起点。

缺点：

没有明确地涉及系统整体架构，可能会造成 “过度设计” 和复杂性。

在实现时需做好管理接口和协议的工作，以保证每层之间的沟通。

层与层之间的性能开销可能导致系统性能下降。

### 虚拟机风格

#### 解释器 Interpreters

解释器风格是一种常见的软件系统架构风格，它基于一个解释器来实现应用程序逻辑。与传统的编译器不同，解释器将整个应用程序代码解释为一组指令，并直接在运行时对程序进行解释执行。解释器通常包括解释器核心和一组解释器插件，通过引入不同的插件来支持多种编程语言。

解释器风格的核心思想是提供一种灵活的逻辑实现方式，通过解释器来执行程序代码，将应用程序逻辑从底层的硬件和操作系统中解耦出来。解释器风格适用于需要快速开发和调试各种应用程序的场景，尤其是用于开发一些动态应用程序，如脚本应用程序、网页应用程序、日志分析应用程序等。它可以实现代码的热加载，动态调试和灵活的应用程序实现方式，并且有较好的跨平台性。

与传统的编译器风格相比，解释器风格的主要优点在于可以实现快速开发、易修改、快速部署和可定制化的应用程序。解释器风格可以实现高度灵活、易于扩展和自定义的应用程序，因为它仅需要在运行时执行代码，而能够快速反应用户的需求变化。此外，解释器风格还提供了较好的交互性、实时性和可编程性，可以让用户更好地控制应用程序的运行。

然而，解释器风格的缺点在于它需要处理大量的解释负担，因为解释器需要在运行时进行逐行解释和执行程序代码，从而导致一定的性能损失。此外，解释器在执行代码时需要访问较多的内存和磁盘资源，在某些情况下可能会造成安全风险和逻辑错误。

#### 基于规则的系统 Rule based systems

### 独立构件风格

#### 进程通讯 communicating-processes

#### 事件系统 event system

### 以数据为中心风格

数据访问者之间的交互或通信仅通过数据存储进行。在客户端之间，数据是唯一的通信方式。控制流将架构分为两类

*   存储库体系结构样式
    
*   黑板体系结构样式
    

#### 仓库风格

优点：

提供数据完整性，备份和还原功能。

提供可伸缩性和代理的可重用性，因为它们彼此之间没有直接通信。

减少软件组件之间瞬态数据的制约开销。

缺点：

它更容易遭受故障，数据副本或重复是可能的。

数据存储和其代理之间的数据结构高度依赖。

数据结构的更改高度影响客户端。

数据的进化是困难且昂贵的。

分布式数据网络传输数据的成本较高。

#### 黑板风格

优点：

提供可伸缩性，易于添加或更新知识源。

提供并发性，允许所有知识源并行工作，因为它们彼此独立。

支持知识源代理的可重用性。

缺点：

黑板的结构变化可能会对其所有代理产生重大影响，因为黑板和知识源之间存在密切的依赖关系。

多个代理同步的问题。

设计和测试系统的主要挑战。

### 异构架构

异构架构是几种风格的组合，组合方式可能有如下几种：

（1）使用层次结构。一个系统构件被组织成某种架构风格，但它的内部结构可能是另一种完全不同的风格。

（2）允许单一构件使用复合的连接件。

### 云体系结构风格

云架构：基础设施层、平台层和应用层三个层次的。

对应名称为 IaaS，PaaS 和 SaaS

软件体系结构文档化
---------

主要目标：

传达软件架构的结构。

理解整体情况。

创建共享愿景：团队和利益相关者。

共同的词汇。

描述软件的构建方式和实现。

聚焦于关于新功能的技术交流。

提供地图来浏览源代码。

证明设计决策的正当性。

帮助新加入团队的开发人员。

### 原则：

以读者的角度编写文档：

• 确定读者及其期望

避免不必要的重复（DRY 原则）

避免歧义

• 解释符号（或使用标准符号）

• 对于图表，使用图例

使用标准组织或模板

• 必要时添加 TBD / To Do

• 组织方便参考 / 链接。

基于体系结构的软件开发
-----------

### 不良设计的特征及避免

◦刚性

◦脆弱性

◦不动性

◦粘性

◦不必要的复杂性

◦不必要的重复

◦不透明度

### OO 设计原则：

◦SRP - 单一职责原则

◦OCP - 开闭原则 : 对扩展开放，对修改关闭

◦LSP - 里氏替换原则 : 子类可以完全替换父类，且用户无需知道它们之间的差异，只需知道父类的接口即可。

◦ISP - 接口分离原则 : 一个类不应该强制性地去实现它所不需要的接口

◦DIP - 依赖倒置原则: 高层模块不应该依赖低层模块，两者都应该依赖**抽象;** 抽象不应该依赖细节，细节应该依赖**抽象**

### 设计模式 **Types of Design Patterns:**

#### ◦ Creational 创建型模式

这些设计模式都涉及类实例化或对象创建。这些模式可以进一步分为类创建模式和对象创建模式。类创建模式在实例化过程中有效地使用继承；

对象创建模式有效地使用委托完成工作。创建型设计模式包括工厂方法模式、抽象工厂模式、建造者模式、单例模式和原型模式。

#### ◦ Structural 结构型模式

这些设计模式关于组织不同的类和对象以形成更大的结构并提供新功能。结构型设计模式包括适配器模式、桥接模式、组合模式、装饰模式、外观模式、享元模式、私有类数据模式和代理模式。

#### ◦ Behavioral 行为型模式

行为型模式是关于识别对象之间常见的通信模式并实现这些模式的技术。行为型设计模式包括职责链模式、命令模式、解释器模式、迭代器模式、中介者模式、备忘录模式、空对象模式、观察者模式、状态模式、策略模式、模板方法模式和访问者模式。

软件体系结构测试与评估
-----------

软件体系结构测试与程序测试有所不同，软件体系结构测试是检查软件设计的适用性，这种测试不考虑软件的实现代码。

软件体系结构测试和程序测试均是软件测试的重要部分，它们之间的区别主要体现在以下几个方面：

1.  目标不同：软件体系结构测试主要关注的是验证软件体系结构是否能够满足系统的功能和质量属性，如性能、可靠性、安全性、可维护性等等。而程序测试则主要关注程序代码是否符合规范，是否能够正确地实现系统需求。
    
2.  侧重点不同：软件体系结构测试侧重于测试系统整体的架构，包括各个功能模块之间的接口、数据流和控制流等。而程序测试则侧重于测试程序代码的每个行为、函数和方法的正确性和一致性。
    
3.  测试技术不同：软件体系结构测试通常采用静态分析、模型检测和形式化验证等技术，以发现结构上的问题和不一致性。而程序测试则通常采用黑盒测试、白盒测试、集成测试等技术，以发现程序代码的各种缺陷和错误。
    
4.  测试对象不同：软件体系结构测试主要针对软件体系结构进行测试，而程序测试则测试各个程序模块、函数和方法的行为是否符合规范和预期。
    

综上所述，虽然软件体系结构测试和程序测试在软件测试中都起着重要的作用，但它们的目标、侧重点、测试技术和测试对象都有所不同，需要针对不同的测试目标采取不同的测试方法和策略。

### 质量属性场景

◦ 通用场景提供了一个框架，用于生成大量的通用、系统独立、特定于质量属性的场景。

◦ 包括六个部分：

◦ 刺激源

◦ 刺激

◦ 环境

◦ 制品

◦ 响应

◦ 响应度量

在环境中，刺激源发出刺激并击中制品中的系统。

可用性关注的是系统故障和系统故障时间的长短。系统故障是指无法提供正确服务的准备工作，当系统无法提供其原本预期的服务时。

可修改性是关于系统变更的成本，包括时间和金钱。

性能关注的是及时性。事件发生时，系统必须及时做出响应。

安全性指系统在为合法用户提供访问权限时，防止或抵御未经授权的访问的能力。攻击是试图违反安全的尝试。

可测试性指软件可以轻松地展示其缺陷或完好无损程度的能力。为了具有可测试性，系统必须控制输入并能够观察输出。

可用性是指用户完成任务的简易程度以及系统为用户完成此任务提供何种支持。可用性的不同方面包括：

*   学习系统功能
    
*   高效地使用系统
    
*   最小化错误的影响
    
*   使系统适应用户的需求
    
*   提高用户的信心和满意度。
    

### 架构权衡分析方法（ATAM）

*   ATAM 是 SAAM 的后继者，也在得到广泛的应用。
    
*   该方法采用质量属性效用树和质量属性类别，对架构进行分析。
    
*   SAAM 没有明确涉及质量属性之间的交互作用，ATAM 则考虑了这一点。因此，权衡考虑的是竞争性的质量属性。
    
*   ATAM 是 SAAM 的一种特化方法，专注于可修改性、性能、可用性和安全性。
    

ATAM 方法的表述　　

商业动机的表述

构架的表述

识别构架方法

生成质量属性效用树

分析构架方法

集体讨论并确定场景优先级

再次分析构架方法

结果的表述

![](https://img-blog.csdnimg.cn/a8fc2007ab2f4364808b1038a546ef26.png)