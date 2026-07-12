OPENQASM 3;

include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cp(pi/2) q[1], q[0];
h q[1];
swap q[0], q[1];

c = measure q;
