# A Collatz Library

In the 30s of the last century, German mathematician Lothar Collatz came up with the following problem:

Take a natural number n.

If n is even, divide it by 2: n_new = n / 2.

If n is odd, multiply it by 3 and add 1: n_new = 3 x n + 1.

Do the same with the result.

Repeating this algorithm leads to a sequence (the Collatz sequence).

Example:

7 is odd      => 7 * 3 + 1 = 22

22 is even    => 22 / 2 = 11

11 is odd     => 11 * 3 + 1 = 34

34 is even    => 34 / 2 = 17

17 is odd     => 17 * 3 + 1 = 52

52 is even    => 52 / 2 = 26

26 is even    => 26 / 2 = 13

13 is odd     => 13 * 3 + 1 = 40

40 is even    => 40 / 2 = 20

20 is even    => 20 / 2 = 10

10 is even    => 10 / 2 = 5

5 is odd      => 5 * 3 + 1 = 16

16 is odd     => 16 / 2 = 8

8 is even     => 8 / 2 = 4

4 is even     => 4 / 2 = 2

2 is even     => 2 / 2 = 1

1 is odd      => 3 * 1 + 1 = 4 


We end up in a loop (4, 2, 1)

Collatz assumed that this behavior is the same for all natural numbers.
And indeed, all number checked up to pow(2,68) seem to support this assumption.

Unfortunately, nobody was able to prove or disprove this assumption so far.
It might even be the case that we can never prove the assumption. According to
Turing and Gödel there must be some problems trhat fall into this category.

Since I am fascinated by the Collatz consumption, I built my own Python library to study
Collatz sequenzes from different angles using different views. I am providing this library
for sharing my enthusiasm with others.

I added a demo to the library so that it can also be executed as a standalone program.
The demo also illustrates how the library might be used.



