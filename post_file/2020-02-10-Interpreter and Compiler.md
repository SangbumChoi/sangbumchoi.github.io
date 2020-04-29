---
title: 'Interpreter and Compiler'
date: 2020-02-10
permalink: /posts/2020/02/Interpreter and Compiler/
tags:
  - Study

---

First, these two differences are comming from due to translating high-level language such as Python, C++ to machine languages which is scripted with '0' and '1's.

###Interpreter
* Translates program one statement at a time.	
* It takes less amount of time to analyze the source code but the overall execution time is slower.	
* No intermediate object code is generated, hence are memory efficient.	
* Continues translating the program until the first error is met, in which case it stops. Hence debugging is easy.	
* Programming language like Python, Ruby use interpreters.

###Compiler
* Scans the entire program and translates it as a whole into machine code.
* It takes large amount of time to analyze the source code but the overall execution time is comparatively faster.
* Generates intermediate object code which further requires linking, hence requires more memory.
* It generates the error message only after scanning the whole program. Hence debugging is comparatively hard.
* Programming language like C, C++ use compilers.

So in order to use interpreter as translator if you did not pass through the error code, then you will not notice the error in your script because it translates program one statement at a time. In compiler because translator change all script into machine langauge level, if there is a small conflict between script you won't get an fine outcome.


Reference

[https://www.programiz.com/article/difference-compiler-interpreter](https://www.programiz.com/article/difference-compiler-interpreter)

Github
