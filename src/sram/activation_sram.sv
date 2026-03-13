module activation_sram 
#(
    parameter counter_width = 8,
    parameter address_width = 8,
    parameter data_width = 64
)
(
    input clk_i,
    input rst_i,

    input  downstream_ready_i,
    output logic downstream_ready_o,
    output rd_valid_o,

    input  [address_width-1:0] addr_i,
    input  [counter_width-1:0] transaction_amount_i,
    input  transaction_rw_mode_i,
    input  load_valid_i,
    output load_ready_o,

    input  [data_width-1:0] wr_data_i,
    output [data_width-1:0] rd_data_o
);

    logic[address_width-1:0] sram_addr;
    logic sram_rw_mode;

    sram_8x256
    activation_sram_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),
        .addr_i(sram_addr),
        .wr_data_i(wr_data_i),
        .rd_data_o(rd_data_o),
        .en_i(~rst_i),
        .rw_mode_i(sram_rw_mode)
    );


    memory_transaction
        #()
    transaction_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .sram_addr_o(sram_addr),
        .sram_rw_mode_o(sram_rw_mode),

        .downstream_ready_i(downstream_ready_i),
        .ready_o(downstream_ready_o),
        .rd_valid_o(rd_valid_o),

        .addr_i(addr_i),
        .transaction_amount_i(transaction_amount_i),
        .transaction_rw_mode_i(transaction_rw_mode_i),
        .load_valid_i(load_valid_i),
        .load_ready_o(load_ready_o)
    );
    
endmodule
