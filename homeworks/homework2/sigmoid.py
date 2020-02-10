#!/usr/bin/env python
# encoding: utf-8
"""
@author: Leo
@contact: lldlld0574@gmail.com
@file: sigmoid.py
@time: 2020/1/26 20:51
@desc:
"""
from sympy import symbols, solve, exp
x = symbols('x')
solve(1./(1+exp(-x)) + 0.001, 0.5 + 0.25*x)
