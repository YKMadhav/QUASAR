OPENQASM 3;

include "stdgates.inc";

qubit[4] q;
bit[4] c;

h q[0];
x q[1];
y q[2];
z q[3];

cx q[0], q[1];
cz q[1], q[2];
swap q[2], q[3];

rx(pi/3) q[0];
ry(pi/5) q[1];
rz(pi/7) q[2];

c = measure q;
