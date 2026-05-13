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

    logic [7:0]  addr_dly;
    logic [7:0] wr_data_dly;
    logic        rw_mode_dly;

`ifdef SIM
    assign #200 addr_dly = addr_i;
    assign #200 wr_data_dly = wr_data_i;
    assign #200 rw_mode_dly = rw_mode_i;
`else
`ifdef SIM_TOP
    assign #2 addr_dly = addr_i;
    assign #2 wr_data_dly = wr_data_i;
    assign #2 rw_mode_dly = rw_mode_i;
`else
    assign addr_dly = addr_i;
    assign wr_data_dly = wr_data_i;
    assign rw_mode_dly = rw_mode_i;
`endif
`endif

    gf180mcu_ocd_ip_sram__sram256x8m8wm1
    sram_inst(
`ifdef USE_POWER_PINS
        .VDD(VDD),
        .VSS(VSS),
`endif
        .CLK(clk_i),
        .CEN(~en_i),
        .GWEN(~rw_mode_dly),
        .WEN('0),
        .A(addr_dly),
        .D(wr_data_dly),
        .Q(rd_data_o)
    );

endmodule