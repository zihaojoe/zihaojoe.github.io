---
layout:     post
title:      Original-命令行语句汇总
subtitle:   Command Line Cheat Sheet
date:       2020-04-20
author:     Joe Zhang
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - hdfs
    - shell
    - command
    - 终端
    - terminal
    - git
    - ssh 
---

## 1. Bash
Source: <https://www.cs.cmu.edu/~15131/f17/>

### 1.1 Command
* ls / tree: -a 显示所有, a代表all; -l显示详细内容（权限）
* cat / less <filename>: 显示文件内容; /banana表示查抄; 按q键退出
* cp <source> <destination>: 拷贝文件
* mv <source> <destination>: 移动文件
* rm <filename>: 删除文件; rm -r folder2/ : 删除文件夹
* mkdir <directory>: 建立路径（文件夹）
* touch <file>: 建立空文件
* echo <text>: 打印文本
* fg: foreground任务。例如fg 1; 用jobs查看当前任务。
* chmod及权限: <https://www.cnblogs.com/peida/archive/2012/11/29/2794010.html>
* vim: https://www.runoob.com/linux/linux-vim.html
* |: 管道操作
* grep(global reg ex print) / sed(stream editor): 正则表达式和自动文件操作
* &gt;&gt;: append stdout; >: overwrite stdout; 2>, 2>>: stderr
    - cat asdf hello.txt 2>&1 > alloutput.txt:把2句柄赋值传递给1句柄的地址（0是stdin, 1是stdout, 2是stderr）; 功能为把stderr发送到stdout, 然后把stdout发送到文件
    - 2>&1 要写到后面: ls a 1>&2 2> b.txt; ls a > b.txt 2>&1
    - /dev/null: 不打印任何东西, 可以理解为垃圾桶位置
* $: 把输出变成输入的一部分, 类比pipe; touch myfile-$(date +%s).txt
* make

### 1.2 Points
* setting variables:
    - myvariable="hello”: 不可被其他程序引用。获得变量值用$, 例如echo $myvariable,  echo lone${another_var}s    
    - export anothervar="some string”: 可被外部或者其他程序引用
    - permission: <https://www.cnblogs.com/peida/archive/2012/11/29/2794010.html>
* {} 生成序列: mv {1.txt,2.txt}; mv Bash/2{.txt,}

## 2. Hadoop
### 2.1 hdfs
* hdfs dfs -cat fileName
* hdfs dfs -mkdir msia
* hdfs dfs -ls
* hdfs dfs -copyFromLocal <localdir> <serverdir>

### 2.2 MapReduce
* ant: compile Java file, in the dir of build.xml

### 2.3 Example - 1st Homework
1. build and compile java code
ant
2. mkdir in hdfs  
    * hdfs dfs -mkdir wc   
    * hdfs dfs -mkdir wc/input   
    * hdfs dfs -copyFromLocal text.txt wc/input   
3. run local(linux system) java using yarn [hdfs input dir] [hdfs output dir]  # output folder should not exist previously   
    * yarn jar WordCount.jar /user/zzu8431/wc/input /user/zzu8431/wc/output   
4. check output  
    * hdfs dfs -ls wc/output  
    * hdfs dfs -cat wc/output part-00000   

## 3. SSH
### 3.1 ssh connection without entering password
* eval \`ssh-agent\`  
* ssh-add 

### 3.2 Add SSH key to the server, so as to skip password
* ssh-copy-id -i .ssh/id_rsa.pub  username@192.168.x.xxx

### 3.3 Remote server connection
* ssh zzu8431@wolf.analytics.private
* ssh zzu8431@msia423.analytics.northwestern.edu
* ssh -T git@github.com

### 3.4 SCP - secure copy
* scp [OPTION] [user@]Local host: file1 [user@]Remote host: file2

### 3.5 additional tutor
* <https://meineke.github.io/workflows/>   # TA Session
* <https://blog.csdn.net/xlgen157387/article/details/50282483>   # Server Config (Renaming)
* <https://blog.csdn.net/liu_qingbo/article/details/78383892>   # SSH key config