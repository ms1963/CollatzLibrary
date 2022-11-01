"""
Distributed with:
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

Library for analyzing the Collatz-Problem/(3n+1)-Problem
including functionality for analyzing alternative Collatz 
Sequences:
while (not in infinite loop):
    even(n) => n = n // 2
    odd(n)  => n = base * n + delta
The library defines a toolkit for investigating and
visualizing Collatz sequences. 
EH (Event Horizons) represent infinite loops which 
most Collatz sequences eventually end up with (such as 
[1,2,4] for base = 3, delta = 1).
When run as a program this module will execute a set of
demos to illustrate some of the available functionality
respectively use cases.
Collatz Library depends on the following packages:
math, cmath, time, datetime, matplotlib{.myplot}, numpy, 
pandas and treelib.{Node, Tree}
    
(c) 2021/2022 by Michael Stal
"""

# dependencies

import math
import cmath
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from treelib import Node, Tree # need to install Treelib

# Helper functions

def even(n):
    return n % 2 == 0
    
def odd(n):
    return not even(n)
    
"""
divmod()
returns (r,q) with number == r + q * base, r < divisor, 
default divisor is 4
"""

def divmod(number, divisor = 4):
    r = number % divisor
    q = number // divisor
    return (r,q)
    
"""
create_range() creates a range starting from 
the lower_bound up to the upper_bound both of 
which are determined first
create_range(100, -5) => range(-5, 100)
create_range(10, 500) => range(10, 500)
"""
    
def create_range(one, two):
    if one <= two:
        return range(one, two)
    else:
        return range(two, one)
        
"""
collatz_generator() takes a number and iteratively calculates the 
corresponding collatz sequence using lazy evaluation:
Sample usage: 
for i in collatz_generator(11):
    print(i)
"""

def collatz_generator(number, base = 3, delta = 1):
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both even or odd")
    num = number
    completed = False
    cseq = {num}
    yield num # generate first element
    while not completed: # generate successors
        num = succ(num, base, delta)
        if num in cseq:
            completed = True
        else:
            cseq.add(num)
            yield num
 
"""
creates large number that increases resulting collatz sequences.
n should be large, init should be a prime number
"""
     
def create_large_number(n, init): # creates a number with high amount of (3+2)-ops
    pow4n = pow(4,n)
    return pow4n - 1 + pow4n * init

        
# Other generalized Collatz functions:
    
# apply Collatz function to floating point number r    
def float_collatz(r):
    return 0.25 * ((7 * r + 2) - cmath.cos(math.pi * r) * (5 * r + 2))

# apply Collatz function to complex number z
def complex_collatz(z):
    j = complex(0,1)
    return 0.25 * ((7 * z + 2) - cmath.exp(j * math.pi * z) * (5 * z + 2))
    
# apply Collatz function to integer n   
def natural_collatz(n):
	# Unification of 3x+1 for odd and x/2 for even
	return 0.25*(2 + 7*n + (-1)*(2 + 5*n)*((-1) ** n))
    

# Atomic Collatz operations    
def succ(n, base = 3, delta = 1): # successor of n wrt. Collatz conjecture 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    if n <= 1:
        return n
    else:
        if n % 2 == 0:
            return int(n // 2)
        else:
            return int(base * n + delta)
            
# succ_with_predicate does the same as succ, i.e. it calculates the 
# next element of the collatz sequence following n.
# In addition, it calls the function check(n) on the argument.
# For example, check() could calculate n % NUMBER.
# the result is a tuple consisting of (result, check(result))
            
def succ_with_check(n, check, base = 3, delta = 1): # successor of n wrt. Collatz conjecture 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    check(n)
    if n <= 1:
        return (n,check(n))
    else:
        if n % 2 == 0:
            return ((int(n // 2), check(int( n // 2))))
        else:
            return ((int(base * n + delta), check(int(base * n + delta))))
    pass

 
# kth successor of n wrt. Collatz conjecture        
def kth_succ(n, k, base = 3, delta =  1): 
    res = n
    if n > 0:
        for j in range(k): res = succ(res, base, delta)
    return res
    
"""
seq_to_set() converts sequence of numbers to set of nodes
assuming (seq[i], seq[i+1]) become edges
"""

def seq_to_set(sequence):
    s = set()
    for i in range(0, len(sequence)-1):
        s.add((sequence[i], sequence[i+1]))  
    return s

"""
call all ops in lambda_list with num as argument and measures 
the times required. Used for benchmarking.
iterations defines the quantity hof often the functions are called 
returns list of total times for all function calls per lambda. The
lambdas require to have one integer argument and no return values.
times[0]: summed up time for calling lambda 1 
times[1]: summed up time for calling lambda 2 
...
"""

def runtime_checker(lambda_list, num, iterations = 1):
    times = []
    for my_lambda in lambda_list:
        start_time = time.time()
        for iter in range(iterations): my_lambda(num)
        times += [time.time() - start_time]
    return times

# REDUCE operations to create Collatz conjectures

"""
regular Collatz operations without any shortcuts.
binary == True => numbers are displayed in binary format
"""

def reduce(number, binary = False, verbose = True): 
    num = number
    div2_ctr = 0
    mult3plus1_ctr = 0
    while num != 1:
        if binary:
            print(bin(num).replace("0b", ""))
        else:
            print("{0:20.0f}".format(num))
    
        if num % 2 == 0: # even number
            num = num // 2
            if verbose: print("-- div 2 ->")
            div2_ctr += 1
        else:            # odd  number
            num = 3 * num + 1
            if verbose: print("-- 3 * n + 1 ->")
            mult3plus1_ctr += 1
    if binary:
        print(bin(num).replace("0b", ""))
    else:
        print("{0:20.0f}".format(num))
        
    print("Regular Algorithm")
    print("div(2)        counted = {0:6d}".format(div2_ctr))
    print("mult(3).inc() counted = {0:6d}".format(mult3plus1_ctr))
    print("Total operations      = {0:6d}".format(div2_ctr + mult3plus1_ctr))
            


# Create EH for reduce_general() -> see below
def find_EH_general(first, base = 3, delta = 1):
    if even(delta) != even(base):
        raise ValueError("delta and base must both be evil or odd")
    EH = {first}
    num = first
    while True:
        if even(num):
            num = num // 2;
        else:
            num = base * num + delta
        if num in EH:
            break 
        else:
            EH.add(num)
            continue
    return sorted(list(EH))
        
    
"""
Collatz function with different base and delta values
delta and base must both be odd or even 
<=> even(delta) == even(base)
even(n_old) => n_new = n_old / 2   
odd(n_old) => n_new = base * n_old + delta
base = 3, delta = 1 => Standard-Collatz
MAXLOOPS: number of loops after which calculation stops
no matter whether it calculated the Collatz Sequence
or not
"""

def reduce_general(number, MAXLOOPS, base = 3, delta = 1, verbose = True): # Collatz with different base and delta 
    loop_ctr = 0
    
    if even(delta) != even(base):
        raise ValueError("delta and base must both be odd or evil")
    else:
        if verbose:
            print("BASE  is {0:6d}".format(base))
            print("DELTA is {0:6d}".format(delta))
    num = number
    div2_ctr = 0
    mult_base_plus_delta_ctr = 0
    collatz_seq = {num}
    completed = False
    while (not completed) and (loop_ctr < MAXLOOPS):
        print("{0:20.0f}".format(num))
        loop_ctr += 1
        if verbose: print("Loop nr. " + str(loop_ctr))
    
        if num % 2 == 0: # even number
            num = num // 2
            if verbose: print("-- div 2 ->")
            div2_ctr += 1
        else:            # odd  number
            num = base * num + delta
            if verbose: print("-- " + str(base) + " * n + " + str(delta) + " ->")
            mult_base_plus_delta_ctr += 1 
        if not num in collatz_seq:
            collatz_seq.add(num)
        else: # num is in sequence => must be point of EH 
            completed = True
            print("*** Element of EH found: {0:6d} ***".format(num))
    if loop_ctr == MAXLOOPS:
        print("Reached maximum number of loops = " + str(MAXLOOPS))
        return
    EH = find_EH_general(num, base, delta)
    print("EH = ", EH)
    print("Regular Algorithm")
    print("div(2)               counted = {0:6d}".format(div2_ctr))
    print("mult_base_plus_delta counted = {0:6d}".format(mult_base_plus_delta_ctr))
    print("Total operations             = {0:6d}".format(div2_ctr + mult_base_plus_delta_ctr))
            
      
"""
reduce_opt minimizes number of collatz operations
to a minimum - helpful to determine quickly whether
collatz assumption holds for number passed
returns a 4-tuple: with occurrences in % of modulo 4 
ops, i.e., mod4 = 0,1,2,3
"""

def reduce_opt(number, verbose = False): # collatz uses modulo 4 calculation for shortcuts
    num = number
    n0 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    completed = (num == 1)
    
    
    while not completed:
        if verbose: print("{0:20.0f}".format(num))
        (r,q) = divmod(num, 4) # num = r + 4 * q
        
        if verbose: print(" mod 4 = {0:1.0f}".format(r))
    
        if q != 0:
            if r == 0: # even number that can be divided by 4
                num = q
                if verbose: print("--22->")
                n0 += 1
            elif r == 1:
                num = 3 * q + 1
                if verbose: print("--322->")
                n1 += 1
            elif r == 2:
                num = 3 * q + 2
                if verbose: print("--232->")
                n2 += 1
            else: # r == 3
                num = 9 * q + 8
                if verbose: print("--3232->")
                n3 += 1
        else: # q == 0
            completed = True
            # r cannot be 0
            # r == 1 => target reached
            if r == 2:
                n2 += 1
            elif r == 3:  
                n3 += 2
                n2 += 1
                n0 += 2
                
    if verbose: 
        print("{0:20.0f}".format(num))
    
    nsum = n0 + n1 + n2 + n3
    if verbose:
        print("n0                    = {0:6d}".format(n0))
        print("n1                    = {0:6d}".format(n1))
        print("n2                    = {0:6d}".format(n2))
        print("n3                    = {0:6d}".format(n3))
        print("Total operations      = {0:6d}".format(nsum))
    return (n0, n1, n2, n3)
    
    
    
"""
Calculation of Standard Collatz EVENT HORIZON respectively 
traverses Collatz conjecture until an 
element in EH is reached
"""

def reach_eh(n, base = 3, delta = 1): 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    e = n
    cseq = {n} # init collatz sequence
    completed = False
    while not completed:
        if even(e):
            e = e // 2
        else:
            e = base * e + delta
        if not e in cseq:
            cseq.add(e)
        else:
            completed = True
    return len(cseq), e


"""
calls reach_EH to find an element in EH
and takes this element to calculate the whole EH
"""

def calc_eh(n, base = 3, delta = 1): 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    size, e = reach_eh(n, base, delta) # get an(y) element of EH
    EH = {e}
    completed = False
    while not completed: # calculate all successors
        if even(e):
            e = e // 2
        else:
            e = base * e + delta
        if not e in EH:
            EH.add(e)
        else:
            completed = True
    return size, sorted(list(EH)) # return result
    
    
# Collatz with ALTERNATE BASES AND EVENT HORIZONS

"""
traverses Collatz conjecture until an 
element in EH is reached. Treshold used to avoid
diverging sequences
"""

def generic_reach_eh(n, base = 3, delta = 1): 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    treshold = pow(10,100)
    e = n
    
    cseq = {n} # init collatz sequence
    completed = False
    while not completed:
        diverging = ((e < 0) and (e < -treshold)) or ((e > 0) and (e > treshold))
        if diverging: ## very likely diverging
            return -1, -1
        if even(e):
            e = e // 2
        else:
            e = base * e + delta
        if not e in cseq:
            cseq.add(e)
        else:
            completed = True
    return len(cseq), e


"""
calls generic_reach_EH to find an element in EH
and takes this element to calculate the whole EH.
Treshold used to avoid diverging sequences.
"""

def generic_calc_eh(n, base = 3, delta = 1): 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    treshold = pow(10,100)
    
    size, e = generic_reach_eh(n, base, delta) # get an(y) element of EH
    if size == -1: return -1, {}
    EH = {e}
    completed = False
    while not completed: # calculate all successors
        diverging = ((e < 0) and (e < -treshold)) or ((e > 0) and (e > treshold))
        if diverging: ## very likely diverging
            return -1, {}
        if even(e):
            e = e // 2
        else:
            e = base * e + delta
        if not e in EH:
            EH.add(e)
        else:
            completed = True
    return size, sorted(list(EH)) # return result
    

"""
this function calculates the generic Collatz sequence of n.
it checks whether the sequence diverges to (-) infinity using
the treshold value 10^100.
"""

def generic_collatz_with_treshold(n, base = 3, delta = 1, verbose = True):
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    treshold = pow(10,100)
    num = n
    size, EH = generic_calc_eh(n, base, delta)
    
    div2_ctr = 0
    mult_base_plus_delta_ctr = 0
    while not (num in EH):
        print("{0:20.0f}".format(num))
        diverging = ((num < 0) and (num < -treshold)) or ((num > 0) and (num > treshold))
        if diverging: ## very likely diverging
            print("exceeds all limits")
            break
        if num % 2 == 0: # even number
            num = num // 2
            if verbose: print("-- div 2 ->")
            div2_ctr += 1
        else:            # odd  number
            num = base * num + delta
            if verbose: print("-- " + str(base) + " * n + " + str(delta) + " ->")
            mult_base_plus_delta_ctr += 1
    print("{0:20.0f}".format(num))
    print("EH = " + str(EH))
    print("Regular Algorithm")
    print("div(2)        counted = {0:6d}".format(div2_ctr))
    print("mult(base).inc() counted = {0:6d}".format(mult_base_plus_delta_ctr))
    print("Total operations      = {0:6d}".format(div2_ctr + mult_base_plus_delta_ctr))
    

"""
searches in given range for all EHs
displays list of all EHs found 
empty list {} in result indicates that
collatz function for some values in range
diverged
"""

def search_for_ehs(rng, base = 3, delta = 1, verbose = False):
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    list_of_ehs = list()
    for n in rng:
        size, eh = generic_calc_eh(n, base, delta)
        if verbose:
            if verbose: print("Number used  -> " + str(n))
        if eh not in list_of_ehs:
            list_of_ehs.append(eh)
            if verbose:
                print("EH of number -> "+ str(n))
    for ehor in list_of_ehs:
        print(str(ehor))
        
                          
"""
BRUTE FORCE CALCULATOR
calculates Collatz sequences including EH 
for a range of numbers rng.
Results are written to a .csv file that 
can be imported into Excel
"""

def search_range(rng, base = 3, delta = 1, filename = "CollatzCalculations"):
    SEP = "; "
    NEWLINE = "\n"
    
    fname = filename+".csv"
    output_file = open(fname, "w")
    output_file.write("Analyzed range: " + str(rng) + NEWLINE);
    
    print("Brute Force Calculation of Collatz Conjectures with Range = " + str(rng))
    
    for i in rng:
        print(i)
        
        len, res = calc_eh(i, base, delta)
        output_file.write("Number, Sequence Length, Resulting EH, Collatz?" + NEWLINE);
        if not 1 in res:
            print("Collatz Assumption does not hold for " + str(tmp))
            break
            print(res)
            output_file.write(str(i) + SEP + str(len) + SEP + str(res) + SEP + "Collatz(" + str(i) + ") = " + str(False) + NEWLINE);
        else: 
            print("Standard EH reached")
            output_file.write(str(i) + SEP + str(len) + SEP + str(res) + SEP + "Collatz(" + str(i) + ") = " + str(True) + NEWLINE);

    output_file.close()        


"""
Method to find number with biggest length of Collatz sequence
in range rng
"""
def search_max_in_range(rng, base = 3, delta = 1):
    maximum = (1,1)
    
    for i in rng:
        result = calc_collatz_sequence_opt(i)
        length = len(result)
        if length > maximum[1]:
            maximum = (i, length)
            
    print("Search in " + str(rng) + " completed")
    print("Found maximum length of " + str(maximum[1]) + " for number " + str(maximum[0]))
        
        
# SINGLE and MULTI step operations

    
# Calculates whole Collatz conjecture and returns it to caller
def calc_collatz_sequence(n, base = 3, delta = 1): # return Collatz sequence as a list 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    num = n
    list = [num]
    while(num != 1):
        num = succ(num, base, delta)
        list.append(num)
    return list
    
"""
same as above but optimized by combining application of Collatz rules
depending on n mod 4. This is only applicable to standard Collatz
"""

def calc_collatz_sequence_opt(n):
    num = n
    list = []

    while (num != 1):
        (r,q) = divmod(num, 4)
        list.append(num)
        if r == 0:
            num = 2 * q
            list.append(num)
            num = q
            if num == 1: break
        else:
            if r == 1:
                num = 4 + 12 * q
                list.append(num)
                num = 2 + 6 * q
                list.append(num)
                num = 1 + 3 * q
                if num == 1: break
            else:
                if r == 2:
                    num = 1 + 2 * q
                    if num == 1: break
                    list.append(num)
                    num = 4 + 6 * q
                    list.append(num)
                    num = 2 + 3 * q
                else: # r == 3
                    num = 10 + 12 * q
                    list.append(num)
                    num = 5 + 6 * q
                    list.append(num)
                    num = 16 + 18 * q
                    list.append(num)
                    num = 8 + 9 * q
                    
    list.insert(len(list), 1)
    return list
    



"""
GRADIENT calculations for Collatz sequences
    The gradient for the Collatz sequence of n is calculated
    by 
        n / (number of Collatz steps for n to reach 1)
        
The gradient counts the steps required for the Collatz sequence 
to go from number to 1
Note: gradient() is based on reduce()
"""

def gradient(number, base = 3, delta = 1): # counter of required steps 
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    num = number
    steps = 0
    
    while num != 1:
        steps += 1
        if num % 2 == 0: # even number
            num = num // 2
        else:            # odd  number
            num = base * num + delta
    return steps
        
"""
optimized gradient calculation that requires approx. 
25% up to  50% less time than the unoptimized version
this depends on number used, memory, CPU. The higher 
the number the more benefit can be gained.   
gradient_opt counts the steps required for the Collatz sequence 
to go from number to 1
Note: gradient_opt() is based on reduce_opt()
"""

def gradient_opt(number): # optimized counter of steps
    num = number
    steps = 0
    completed = False
    
    while (num != 1):
        (r,q) = divmod(num, 4)
        # num = q * 4 + r
    
        if q != 0:
            if r == 0: # even number that can be divided by 4
                num = q
                steps += 2
            elif r == 1:
                num = 3 * q + 1
                steps += 3
            elif r == 2:
                num = 3 * q + 2
                steps += 3
            else:  # r == 3
                num = 9 * q + 8
                steps += 4
        else: # q == 0
            if r == 0: # invalid number
                return -1
            elif r == 1:
                steps += 0
            elif r == 2:
                steps += 1
            else: # r == 3
                steps += 7 #3->10->5->16->8->4->2->1
            num = 1
    return steps

 
# gradient_calculator() calculates the gradient := log_base(num)/steps       
def gradient_calculator(num, log_base = 2):
    steps = gradient_opt(num)
    if steps == 0: steps == 1 # prevent zero
    if (False <= 1): # for bases <= 1 use ln() instead
        return math.log(num)/steps   
    else: # use passed base
        return math.log(num)/(steps * math.log(log_base))       
 
"""
gradient_finder_in_range() calculates the minimum gradient found in the
range between frm and to.
minimum defines the upper limit for minimum
pingtime is the time peried in seconds after which the current
number analyzed is shown
"""
             
def find_gradient_in_range(frm, to, minimum = 1, log_base = 2, pingtime = 60, verbose = False):
    min = minimum
    print("log_" + str(log_base) + "_(n) / steps")
    starttime = time.time()
    for i in range(frm, to, 1):
        grad = gradient_calculator(i, log_base = 2)
        
        if verbose: 
            if (time.time() - starttime) > pingtime:
                print(str(datetime.datetime.now())) 
                print("Current number: " + str(i))
                starttime = time.time()
                print("Gradient:       " + str(grad))
            if (grad < min):
                print("Minimum found at " + str(i) + " is " + str(grad))
                min = grad
                
"""
same as gradient_finder_in_range() but uses a sequence of numbers 
instead of a range as input
"""
               
def find_gradient_in_sequence(seq, log_base = 2, verbose = False):
    min = 1
    min_val = 1
    print("log_" + str(log_base) + "_(n) / steps")
    for num in seq:
        if verbose: print(num)
        grad = gradient_calculator(num, log_base)
        
        if (grad < min):
            print("Minimum found at " + str(num) + " is " + str(grad))
            min = grad
            min_val = num
    print("Minimum " + str(min) + " achieved by number " + str(min_val))
           
    
# OPERATIONS for visualization
    
       
"""
visualize single Collatz sequence as a tree
uses visualize_collatz_trees()
"""

def visualize_collatz_tree(num): 
    visualize_collatz_trees(num, num)

    
# visualize range of collatz sequences as a tree
def visualize_collatz_trees(frm, to):
    tree = Tree()
    tree.create_node("1", 1)
    edges = set()
    for n in range(frm, to+1):
        CSeq = calc_collatz_sequence_opt(n)
        for i in range(len(CSeq)-1, 0, -1):
            if (CSeq[i-1], CSeq[i]) not in edges: 
                tree.create_node(str(CSeq[i-1]), CSeq[i-1], CSeq[i])
        edges = edges.union(seq_to_set(CSeq))
    tree.show()
    

"""
visualize_collatz_sequences_lengths_and_maxima() calculates for
a range the lengths of Collatz Sequences
range spans from 1 <= lower_bound and to upper_bound
illustrates lengths and maximas of analyzed conjectures
"""

def visualize_collatz_sequences_lengths_and_maxima(lower_bound, upper_bound, connected = False):
    if lower_bound < 1:
        raise ValueError("only positive numers allowed as lower_bound")
    if upper_bound < lower_bound:
        raise valueError("upper_bound must be greater or equal than lower_bound")
    x=[]
    y1=[]
    y2=[]
    if connected:
        sep = "o-" 
    else:
        sep = "o"
        
    rng = range(lower_bound, upper_bound)
    for i in rng:
        a = calc_collatz_sequence(i)
        x.append(i)
        y1.append(max(a))
        y2.append(len(a))
    
    figure, axes = plt.subplots(nrows = 2, ncols = 1)
        
    plt.subplot(2,1,1)
    plt.plot(x,y1,sep, color = 'cyan')
    plt.title('Collatz Conjecture - Maxima')
    plt.xlabel('Number')
    plt.ylabel('Maximum of Collatz conjecture')
    
    plt.subplot(2,1,2)
    plt.plot(x,y2,sep, color = 'orange')
    plt.title('Collatz Conjecture - Lengths')
    plt.xlabel('Number')
    plt.ylabel('Length of Collatz conjecture')
    
    figure.tight_layout(pad=3.0)
    
    plt.show()
    
    
"""
visualize_collatz_sequences_modulo4() calculates Collatz 
visualize_collatz_sequences in the range between lower_bound 
and upper_bound (lowerBound 1 not allowed)
connected = false => print points 
connected = True  => and connect these points
The diagrams show the distribution of nummber in the Collatz 
Sequences wrt. modulo 4 distribution
"""

def visualize_collatz_sequences_modulo4(lower_bound, upper_bound, connected = False):
    if lower_bound < 2:
        raise ValueError("1 or lower not allowed as lower_bound")
    if upper_bound < lower_bound:
        raise valueError("upper_bound must be greater or equal than lower_bound")
    x=[]
    n0=[]
    n1=[]
    n2=[]
    n3=[]
    if connected:
        sep = "o-" 
    else:
        sep = "o"
    iters = upper_bound - lower_bound + 1
    rng = range(lower_bound, upper_bound)
    for i in rng:
        val0, val1, val2, val3 = reduce_opt(i)
        
        sum = val0 + val1 + val2 + val3
        x.append(i)
        if sum > 0:
            n0.append(val0 / (sum))
            n1.append(val1 / (sum))
            n2.append(val2 / (sum))
            n3.append(val3 / (sum))
    
    figure, axes = plt.subplots(nrows=2, ncols=2)

    # Subplot 1:
    plt.subplot(2,2,1)
    plt.plot(x,n0,sep, color = 'green')
    plt.title("modulo 4 == 0")
    plt.xlabel('Number')
    plt.ylabel('Occurrences in %')
    
    # Subplot 2:
    plt.subplot(2,2,2)
    plt.plot(x,n1,sep, color = 'red')
    plt.title("modulo 4 == 1")
    plt.xlabel('Number')
    plt.ylabel('Occurrences in %')
    
    # Subplot 3:
    plt.subplot(2,2,3)
    plt.plot(x,n2,sep, color = 'blue')
    plt.title("modulo 4 == 2")
    plt.xlabel('Number')
    plt.ylabel('Occurrences in %')
    
    # Subplot 4:
    plt.subplot(2,2,4)
    plt.plot(x,n3,sep, color = 'magenta')
    plt.title("modulo 4 == 3")
    plt.xlabel('Number')
    plt.ylabel('Occurrences in %')
    
    figure.tight_layout(pad=3.0)
    
    plt.show()
     
"""
the function visualize_collatz_sequences() and its helper functions
collatz(), transforms(), and color_picker() visualize 
Collatz-Sequences graphically
"""


def visualize_collatz_sequences(runs):
    # calculate collatz sequence 
    def collatz(n):
        if n == 1:                             
            cseq = [1]
        elif n % 2 == 0:
            cseq = collatz(n // 2) + [n]
        elif n % 2 == 1:
            cseq = collatz(3 * n + 1) + [n]
        return cseq

    # pick random color
    def color_picker():
        return np.random.choice(['#FF5E02','red',     
                'blue','#6400FF','#E10060','#02D1FF'])
            

    # calculate mathematical representation
    def mapping(arr):
        seq = [0]
        val = [0]
        rad = 0
        even = -.54 * (np.pi / 180 )
        odd  =  1.2 * (np.pi / 180 )
        for i in range(1, len(arr)):
            if arr[i] % 2 == 0:
                seq.append(seq[i - 1] + np.sin(rad+even))
                rad = rad + even            
            else:
                seq.append(seq[i - 1] + np.sin(rad+odd))
                rad = rad + odd
            val.append(val[i - 1] + np.cos(rad))
        return val,seq


    seen = {}
    sequence_lengths = []
    for i in range(1, runs):
        length = collatz(i)
        sequence_lengths.append(len(length))   
    plt.figure(figsize = (20,20))
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    
    for i in range(1, runs):
        length = collatz(i)    
        sequence_lengths.append(length)
        x,y = mapping(np.array(length))
        ax.set_facecolor('black')
        ax.plot(x,y, alpha=0.15, color=color_picker());    
    plt.show()
    
    
"""
visualize_collatz_threads_in_range() draws all 
collatz threads in a given range
"""
    
def visualize_collatz_sequences_in_range(frm, to, base = 3, delta = 1):
    if (even(base) != even(delta)):
        raise ValueError("base and delta must be both evil or odd")
    seen = {}
    sequence_lengths = []
    delta = to - frm + 1
    for i in range(0, delta):
        length = collatz(frm + i)
        sequence_lengths.append(len(length))   
    plt.figure(figsize = (20,20))
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    
    for i in range(0, delta):
        length = collatz(frm + i)    
        sequence_lengths.append(length)
        x,y = mapping(np.array(length))
        #ax.set_facecolor('black')
        ax.plot(x,y, alpha=0.15, color=color_picker());    
    plt.show()
    

# Loops that allow to interactively enter number
# and see the Collatz sequences they produce
     
""" 
version with base*n+delta instead of 3n+1 
MAXLOOPS denotes maximum number of loops after which 
calculation sStopAsyncIteration
"""
              
def collatz_loop_general(MAXLOOPS=10000, base = 3, delta = 1):
    if even(delta) != even(base):
        print("BASE and DELTA must both be odd or evil")
        return
    print("COLLATZ SEQUENCE CALCULATOR FOR base * n + delta")
    print("================================================")
    print("BASE  is {0:6d}".format(base))
    print("DELTA is {0:6d}".format(delta))
    print("You may enter positive or negative integer values!")
    print("Enter 'q'' to stop loop")
        
    while True:
        print()
        value = input("Enter an integer value (or q to stop) --->")
        try:
            n = int(value)
            reduce_general(n, MAXLOOPS, base, delta)
        except:
            if value == "q":
                break
            else:
                continue

"""
collatz_loop_binary() calculates standard 3n + 1 Collatz sequences
and displays numbers in binary mode.
"""

def collatz_loop_binary():
    print("COLLATZ SEQUENCE CALCULATOR (BINARY MODE)")
    print("=========================================")
    print("You may enter positive or negative integer values!")
    print("Enter 'q'' to stop loop")
    
    while True:
        print()
        value = input("Enter an integer value (or q to stop) --->")
        try:
            n = int(value)
            reduce(n, binary = True)
        except:
            if value == "q":
                break
            else:
                continue

        
# DEMO code: runs only when executed as a program, not as a module
def run_demos():
    print("Demo of Collatz Library in Python (c) 2022 by Michael Stal")
    
    def demo1():
        print("*** Demo  1: Calculation of collatz sequence")
        for i in range(10):
            print("Collatz Sequence of " + str(i+1))
            print(calc_collatz_sequence(i+1))
   
    def demo2():
        print("*** Demo  2: Search range for event horizons")
        search_for_ehs(create_range(-1, -10000), base = 3, delta = 1)
        
    def demo3():
        print("*** Demo  3: Collatz loop using binary format")
        collatz_loop_binary()
        
    def demo4():
        print("*** Demo  4: Collatz loop")
        collatz_loop_general(1000000, 3, 1)
    
    def demo5():
        print("*** Demo  5: Search range for collatz sequences")
        search_range(range(1000, 1010))
        
    def demo6():
        print("*** Demo  6: Calculate gradient in sequence or range")
        numbers = list()
        for i in range(2,12,1): numbers.append(i)
        find_gradient_in_sequence(numbers, log_base = 2, verbose = True)
        find_gradient_in_range(pow(2,68), pow(2,68) + 500, minimum = 0.1, log_base = 2,  pingtime = 120, verbose = True)
     
    def demo7():
        print("*** Demo  7: Calculate collatz optimized")                               
        reduce_opt(99997557553177553177975399977777555555555553332, True)
        
    def demo8():
        print("*** Demo  8: Visualize collatz conjectures as trees")
        visualize_collatz_trees(frm = 1, to = 7)
        #visualize_collatz_tree(83)

    def demo9():
        print("*** Demo  9: Visualize collatz sequences in 2D")
        visualize_collatz_sequences(1000)
        #visualize_collatz_sequences_in_range(1000000000000000000000000000000, 1000000000000000000000000010000)

    def demo10a():
        print("*** Demo 10a: Visualize statistics of collatz sequences - modulo")
        visualize_collatz_sequences_modulo4(2, 20, Tree)
        
    def demo10b():
        print("*** Demo 10b: Visualize statistics of collatz sequences - lengths and maxima")
        visualize_collatz_sequences_lengths_and_maxima(1, 20, True)
    
    
    def demo11():
        print("*** Demo 11: Measure and compare runtimes of algorithms gradient and gradient_opt")
        lambda_list = [gradient, gradient_opt]
        timelist = runtime_checker(lambda_list, 12345677765432887651777655543332111323233344499987776543332119988776543211123332211777665543332123445500877765887776555433321112348887655543231122334455566667777445511188777655554333221118877009876543212347777661999876555432123456789000000008876112399887776543212233454566778888999998760076588776554322119988877766655544321, 1000)
        print("%s seconds" % timelist[0])
        print("%s seconds" % timelist[1])
        print("runtime ratio second/first = " + str(100 * timelist[1]/timelist[0])+"%")
        print("Benefit = " + str(100-100*(timelist[1]/timelist[0])) + "%") 
    
    def demo12():
        print("*** Demo 12: Use generator to create collatz sequence lazily")
        for i in collatz_generator(3):
            print(i)
            
    def demo13():
        print("*** Demo 13: Find Collatz sequence with maximum length in range")
        search_max_in_range(range(1,100000))
        
    def demo14():
        print("*** Demo 14: Calculate succ and run a check() function on the result")
        n = 157
        check = lambda x: x % 8
        while True:
            (n_new,c) = succ_with_check(n, check)
            n = n_new 
            print("Check result (n % 8): " + str(c))
            if (n == 1): break
            
    # run demo functions in demo_list
    
    
    demo_list = [ demo1, demo2, demo3, demo4, demo5, demo6, demo7,
                  demo8, demo9, demo10a, demo10b, demo11, demo12,
                  demo13, demo14 ]
    for fun in demo_list:
        ignore = input(" Press <return> to continue---> ")
        print()
        fun() # call demo
        
    print()
    print("DEMOS completed")
    
    
if __name__ == "main":
    print("Started from command line using CLI")
    run_demos()

else:
    print("Imported as a library")
    run_demos()


