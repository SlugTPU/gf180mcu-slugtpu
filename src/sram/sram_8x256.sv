module sram_8x256 
    (input clk_i
    ,input rst_i
    ,input [7:0] addr_i  // every 64 bits is one address - NOT single byte addressable
    ,input [63:0] wr_data_i
    ,input en_i //chip enable
    ,input rw_mode_i //1 for writing, 0 for reading
    ,output [63:0] rd_data_o
    );

    logic [7:0]  addr_dly;
    logic [63:0] wr_data_dly;
    logic        rw_mode_dly;

    always @(addr_i) addr_dly = #200 addr_i;
    always @(wr_data_i) wr_data_dly = #200 wr_data_i;
    always @(rw_mode_i) rw_mode_dly = #200 rw_mode_i;


    genvar i;
    generate
        for (i = 0; i < 8 ; i++ ) begin
            gf180mcu_ocd_ip_sram__sram256x8m8wm1
            sram_mod_i(
                .CLK(clk_i),
                .CEN(~en_i),
                .GWEN(~rw_mode_dly),
                .WEN('0),
                .A(addr_dly),
                .D(wr_data_dly[i*8 +7 : i*8]),
                .Q(rd_data_o[i*8 +7 : i*8])
            );
        end
    endgenerate

endmodule