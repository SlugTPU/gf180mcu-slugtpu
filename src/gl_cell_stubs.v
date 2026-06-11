// Functional stubs for standard cells missing from the PDK verilog models.
// Drive-strength variants (e.g. _4) are functionally identical to the modeled
// variant (_2); only the drive current differs, which is irrelevant in simulation.

// Temporary fix until Verilog model is pushed on AS 3v3 library

module gf180mcu_as_sc_mcu7t3v3__dfxtp_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	input CLK,
	input D,
	output Q
);
reg state;
always @(posedge CLK) state <= D;
assign Q = state;
endmodule
