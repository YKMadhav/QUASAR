OPENQASM 3;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

h q[0];
rx(0.5) q[1];
ry(1.2) q[2];
cx q[0], q[1];
swap q[1], q[2];
ccx q[0], q[1], q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
