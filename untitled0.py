#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:24:14 2019

@author: yiqing
"""
#Computer Project 1
from numpy import *
from pylab import *
#Problem 1
def is_mult_table(x):
    if not isinstance(x,list):
        return False
    if len(x) <= 0:
        return False
    if isinstance(x[0],list):
        sub_len=len(x[0])
    for i in x:
        if not isinstance(i,list):
            return False
        if len(i) != len(i):
            return False
        if len(i) != sub_len:
            return False
        for m in i:
            if not isinstance(m,int):
                return False
            if m<0 or m>=sub_len:
                return False
    return True
                   
#Problem 3
import random
def random_mult_table(n):
    assert isinstance(n, int)
    assert n > 0
    out = []
    for i in range(n):
        out.append([])
        for j in range(n):
            out[i].append(random.randint(0,n-1))
    return out

#Problem 4
#There are n**(n**2)
#Because there are n elements, and for each element there are n possibilities, so there are n**(n**2) multiplication table.

#Problem 5
def is_associative_mult_table(x):
    if not is_mult_table(x):
        return False
    for i in range(len(x)):
        for j in range(len(x)):
            for k in range(len(x)):
                if x[x[i][j]][k] != x[i][x[j][k]]:
                    return False
    return True

#Problem 7
def row_mult_table(x,i):
    assert is_mult_table(x)
    assert isinstance(i, int)
    assert 0 <= i and i < len(x)
    return x[i]
def col_mult_table(x,i):
    assert is_mult_table(x)
    assert isinstance(i, int)
    assert 0 <= i and i < len(x)
    return [row[i] for row in x]
def identity_mult_table(x): 
    if not is_mult_table(x):
        return -1
    axis = []
    for i in range(len(x)):
        axis.append(i)    
    for i in range(len(x)):
        for j in range(len(x)):
            if col_mult_table(x, j) == axis and row_mult_table(x, i) == axis:
                return x[i][j]
    return -1            

#Problem 9
#Part a
import pickle, requests
tabs = pickle.loads(bytes(requests.get("http://goo.gl/ZtqICh").text, 'utf-8'))
def Fails_one_triple(tabs):
    for i in tabs:
        if not is_mult_table(i):
            return False
        sum = 0
        for t in range(len(i)):
            for j in range(len(i)):
                for k in range(len(i)):
                    if i[i[t][j]][k] != i[t][i[j][k]]:
                        sum = sum + 1
        if sum == 1:
            return True
    return False  
Fails_one_triple(tabs)
#returns True

#Part b
def Fails_every_triple(tabs):
    for i in tabs:
        if not is_mult_table(i):
            return False
        sum = 0
        for t in range(len(i)):
            for j in range(len(i)):
                for k in range(len(i)):
                    if i[i[t][j]][k] != i[t][i[j][k]]:
                        sum = sum + 1
        if sum == 27:  
            return True
    return False
Fails_every_triple(tabs)
#returns True

#Part c
def Fails_exact_m(tabs, m):
    for t in tabs:
        if m < 0:
            return False
        if m >27:
            return False
        if not is_mult_table(t):
            return False
        sum = 0
        for i in range(len(t)):
            for j in range(len(t)):
                for k in range(len(t)):
                    if t[t[i][j]][k] != t[i][t[j][k]]:
                        sum = sum + 1
        if sum == m:
            return True
    return False
Fails_exact_m(tabs,3)
#returns True

#Problem 10
import pickle, requests
tabs = pickle.loads(bytes(requests.get("http://goo.gl/ZtqICh").text, 'utf-8'))

def is_group_mult_table(x):
    if not is_mult_table(x):
        return False
    if not is_associative_mult_table(x):
        return False
    e = identity_mult_table(x)
    if e == -1:
        return False
    for i in range(len(x)):
        for j in range(len(x)):
            if e in row_mult_table(x, i) and e in col_mult_table(x,j):
                continue
            else:
                return False
    return True

#Part a
def a_a(tabs):
    count = 0
    for i in tabs:
        if is_group_mult_table(i):
            count = count +1
    return count
a_a(tabs)
#returns 3

#Part b
def is_abelian(x):
    for m in range(len(x)):
        for n in range(len(x)):
            if x[m][n]==x[n][m]:
                continue
            else:
                return False
    return True
   
def b_b(tabs):
    count = 0
    for i in tabs:
        if is_group_mult_table(i) and is_abelian(i):
            count = count + 1
    return count
b_b(tabs)
#return 3

#Part c
def c_c(tabs):
    count = 0
    for i in tabs:
        if is_abelian(i) and not is_associative_mult_table(i):
            count = count + 1
    return count

def c_c2(tabs):
    count = 0
    for i in tabs:
        if is_associative_mult_table(i) and not is_abelian(i):
            count = count + 1
    return count
c_c(tabs)
#returns 666
c_c2(tabs)
#returns 50

#Part d
def identity_more_table(x): 
    if not is_mult_table(x):
        return -1
    axis = []
    identities = []
    for i in range(len(x)):
        axis.append(i)    
    for i in range(len(x)):
        for j in range(len(x)):
            if col_mult_table(x, j) == axis and row_mult_table(x, i) == axis:
                identities.append(x[i][j])
    if identities:
        return identities
    else:
        return -1
        
def d_d(tabs):
    for i in tabs:
        if not isinstance(i, list):
            if len(identity_more_table(i)) > 1:
                return True
    return False
d_d(tabs)
#returns False

#Part e
def e_e(tabs):
    for i in tabs:
        if is_associative_mult_table(i) and identity_mult_table(i)==-1 :
            return True
    return False
e_e(tabs)
#returns True

#Part f
def inverse(x):
    e = identity_mult_table(x)
    if e == -1:
        return False
    for i in range(len(x)):
        for j in range(len(x)):
            if e in row_mult_table(x, i) and e in col_mult_table(x,j):
                continue
            else:
                return False
    return True

def f_f(tabs):
    for i in tabs:
        if identity_mult_table(i)!=-1 and not is_associative_mult_table(i) and not inverse(i):
            return True
    return False
print(f_f(tabs))
#returns True


#computering project 2
#Problem 1
import requests
open("groups.py", "wb").write(requests.get("https://goo.gl/Kbhn8a").content)
from groups import *    
def order_perm(x):
    assert IsPerm(x), "the argument must be a permutation"
    t=SymmetricGroup(x.degree()).identity()
    d=1
    for i in range(0,100,1):
        if t!=x**d:
            d=d+1
    return d

#Problem 3
import gzip, pickle, requests
from groups import *
open("mhp.p.gz", "wb").write(requests.get("https://goo.gl/h4kHnt").content)
p = pickle.load(gzip.open("mhp.p.gz", "r"))

def find_order(p):
    assert IsPerm(p), "the argument must be a permutation"
  



#Problem 4
def is_subgroup(G, H):
        assert IsSymmetricGroup(G), "the first argument should be a symmetric group"
        assert isinstance(H, list), "the second argument should be a list"
        for h in H:
            if not h in G:
                return False
            if not h**-1 in H:
                return False
            for m in H:
                if not m*h in H:
                    return False
        return True

#Problem 5
def is_cyclic_subgroup(G, H):
    if not is_subgroup(G, H):
        return False
    t=len(H)
    for h in H:
        if order_perm(h)==t:
            return True
    return False
       
            
#Problem 7
#a
def is_abelian_subgroup(G, H):
    if not is_subgroup(G, H):
        return False
    for i in H:
        for j in H:
            if i*j!=j*i:
                return False
    return True
#b
print(is_abelian_subgroup(SymmetricGroup(11),[Perm((1,2,3)), Perm(), Perm((1,3,2))]))
print(is_abelian_subgroup(SymmetricGroup(2),[Perm((1,2,3)), Perm(), Perm((1,3,2))]))
print(is_abelian_subgroup(SymmetricGroup(4),[Perm((1,2)), Perm(), Perm((3,4))]))
print(is_abelian_subgroup(SymmetricGroup(5),[Perm((1,2)), Perm(), Perm((3,4)), Perm((1,2),(3,4))]))
#c
#Perm(), Perm((1,2)), Perm((3,4)), Perm((5,6)), Perm((7,8)), Perm((1,2),(5,6)), Perm((1,2),(3,4)), Perm((1,2),(7,8)), Perm((3,4),(5,6)),Perm((3,4),(7,8)), Perm((5,6),(7,8)), Perm((1,2),(3,4),(5,6)), Perm((1,2),(5,6),(7,8)), Perm((1,2),(3,4),(7,8)), Perm((3,4),(5,6),(7,8)), Perm((1,2),(3,4),(5,6),(7,8))
#this is when n=4,16 elements inside
#d
def non_cyclic_but_abelian(n):
    x=[]
    y=[]
    z=[]
    a=[]
    for i in range(2,n*2+1,2):
        j=i-1
        x.append(Perm((j, i)))
    for k in x:
            for t in x:
                if not k*t in y and k != t:
                    y.append(k*t)
    for w in x:
        for s in x:
            for q in x:
                if not w*s*q in z and w !=s and s!=q and q!=w:
                    z.append(w*s*q)
                    
    for v in x:
        for b in x:
            for j in x:
                for l in x:
                    if not v*b*j*l in a and v !=b and v!=j and v!=l and b!=j and b!=l and j!=l:
                        a.append(v*b*j*l)
    lucky = x + y + z + a
    lucky.append(Perm(()))
    
    return lucky
print(non_cyclic_but_abelian(5))

print(is_abelian_subgroup(SymmetricGroup(10),non_cyclic_but_abelian(5)))   

   
o = Perm((1,2,3,4,5,6))*Perm((31,32,33,34,35,36))  

print(is_subgroup(SymmetricGroup(36),[Perm((1,2,3,4,5,6),(31,32,33,34,35,36)),Perm()]))

               
                
                    
            

        
            
            

