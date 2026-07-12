OPENQASM 3;

include "stdgates.inc";

qubit[6] q;
bit[6] c;

// Initial gates
h q[0];
h q[1];
h q[2];
x q[3];
y q[4];
z q[5];

// CX chain
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[5], q[0];

// Additional entangling gates
cx q[0], q[2];
cx q[2], q[4];
cx q[4], q[1];
cx q[1], q[3];
cx q[3], q[5];
cx q[5], q[2];

// Three-qubit gates
ccx q[0], q[1], q[2];
ccx q[3], q[4], q[5];

// SWAP gates
swap q[0], q[5];
swap q[1], q[4];

// Increase circuit depth
h q[0];
x q[1];
y q[2];
z q[3];

rx(pi/4) q[4];
rz(pi/8) q[5];

h q[1];
x q[2];
y q[3];
z q[4];

rx(pi/5) q[5];
rz(pi/6) q[0];

h q[0];
h q[2];
h q[4];

cx q[0], q[3];
cx q[1], q[5];

// Measurements
c = measure q;
