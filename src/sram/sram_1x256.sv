`timescale 1 ps / 1 ps
module sram_1x256
    (
`ifdef USE_POWER_PINS
    inout wire VDD,
    inout wire VSS,
`endif
    input clk_i
    ,input rst_i
    ,input [7:0] addr_i 
    ,input [7:0] wr_data_i
    ,input en_i //chip enable
    ,input rw_mode_i //1 for writing, 0 for reading
    ,output [7:0] rd_data_o
    );

    gf180mcu_ocd_ip_sram__sram256x8m8wm1
    sram_inst(
`ifdef USE_POWER_PINS
        .VDD(VDD),
        .VSS(VSS),
`endif
        .CLK(clk_i),
        .CEN(~en_i),
        .GWEN(~rw_mode_i),
        .WEN('0),
        .A(addr_i),
        .D(wr_data_i),
        .Q(rd_data_o)
    );

endmodule
