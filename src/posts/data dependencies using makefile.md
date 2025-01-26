---
layout: post
title: Data dependencies using Makefile
excerpt: 
date: 2024-12-20
tags:
  - post
  - ai
published: true
cssclasses:
  - page-manila
---
Organization in data science projects can make a massive difference in the experience, especially when working on a team.

Over the last semester in my Masters in AI, I've iterated on project structure and developed some opinions, which I'd like to summarize in a future post, but for now let's focus on one very cool tool: the Makefile.

I first used Makefiles in my undergraduate degree working with C++. C++ is a compiled language, and it's important that code gets compiled in the right order, since one module may depend on another. This could be done manually by carefully compiling each file from the most basic up, but it would be extremely tedious -- if you made a change to a low level file, you'd have to recompile all the dependent files.

It could also be done with a script that compiles all modules with a glob, but how would you enforce the order? Well this is where a Makefile comes in, allowing you to specify how one module is dependent on others. That way, when you compile a module, it first triggers the compilation of dependent modules. It even checks if the dependent modules are unchanged, and skips them if so.
### Makefiles in Data Science

I hadn't seen makefiles in years, until I was looking for a better way to organize data science projects and stumbled upon [cookie cutter data science](). This structure recommends using a Makefile to manage *data dependencies*. I was intrigued.

It turns out, dependencies in 
