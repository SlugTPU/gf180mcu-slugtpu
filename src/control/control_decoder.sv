module control_decoder #(
    parameter int SRAM_ADDR_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DRAM_COUNTER_WIDTH = 8,
    parameter int DRAM_DATA_WIDTH = 64
) (
    input clk_i,
    input rst_i,

    output instruction_ready_o,
    input [7:0] instruction_data_i,
    input instruction_valid_i,

    // DMA/DRAM control signals
    output [DRAM_ADDR_WIDTH-1:0] dram_start_addr_o,
    output [DRAM_COUNTER_WIDTH-1:0] dram_word_count_o,
    output dram_we_o, // 0=DRAM->stream, 1=stream->DRAM
    output start_i,
    input busy_o,
    input done_o,

    // DRAM IO streams
    // Read to SRAM from DRAM
    input dram2sram_valid_i,
    input [DRAM_DATA_WIDTH-1:0] dram2sram_data_i,
    output dram2sram_ready_o,

    // Write from SRAM to DRAM
    output sram2dram_valid_o,
    output [DRAM_DATA_WIDTH-1:0] sram2dram_data_o,
    input sram2dram_ready_i

    // TODO: SPI out instruction
);

    compute_core #(
        .DATA_WIDTH (8),
        .ACC_WIDTH  (32),
        .N          (8),
        .BUS_WIDTH  (64),
        .address_width (SRAM_ADDR_WIDTH),
        .counter_width (SRAM_ADDR_WIDTH)
    ) compute_core_inst (
        .clk_i                        (clk_i),
        .rst_i                        (rst_i),

        .load_bias_en_i               (),
        .load_zp_en_i                 (),
        .load_scale_en_i              (),
        .relu_enable_i                (),
  
        .act_enable_i                 (),
        .weight_enable_i              (),

        .act_addr_i                   (),
        .act_transaction_amount_i     (),
        .act_transaction_rw_mode_i    (),
        .act_load_valid_i             (),
        .act_load_ready_o             (),

        .weight_addr_i                (),
        .weight_transaction_amount_i  (),
        .weight_transaction_rw_mode_i (),
        .weight_load_valid_i          (),
        .weight_load_ready_o          (),

        .act_wr_data_i                (),
        .act_wr_valid_i               (),
        .act_downstream_ready_o       (),

        .act_rd_data_o                (),
        .act_rd_valid_o               (),
        .act_rd_ready_i               (),

        .weight_wr_data_i             (),
        .weight_wr_valid_i            (),
        .weight_downstream_ready_o    (),

        .weight_rd_data_o             (),
        .weight_rd_valid_o            (),
        .weight_rd_ready_i            ()
    );
    
endmodule