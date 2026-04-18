module compute_core #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    parameter int N = 8,
    parameter int BUS_WIDTH = 64,
    parameter int address_width = 8,
    parameter int counter_width = 8
) (
    input clk_i,
    input rst_i,

    // scalar stage
    input load_bias_en_i,
    input load_zp_en_i,
    input load_scale_en_i,
    // mxu
    input act_enable_i,
    input weight_enable_i,

    input  [address_width-1:0] act_addr_i,
    input  [counter_width-1:0] act_transaction_amount_i,
    input  act_transaction_rw_mode_i,
    input  act_load_valid_i,
    output act_load_ready_o,

    input  [address_width-1:0] weight_addr_i,
    input  [counter_width-1:0] weight_transaction_amount_i,
    input  weight_transaction_rw_mode_i,
    input  weight_load_valid_i,
    output weight_load_ready_o,

    //Control Unit to SRAM
    input [BUS_WIDTH-1:0] act_wr_data_i,
    input act_wr_valid_i,
    output act_downstream_ready_o,

    output [BUS_WIDTH-1:0] act_rd_data_o,
    output act_rd_valid_o,
    input  act_rd_ready_i,

    input [BUS_WIDTH-1:0] weight_wr_data_i,
    input weight_wr_valid_i,
    output weight_downstream_ready_o,

    //debug signals
    output [BUS_WIDTH-1:0] weight_rd_data_o,
    output weight_rd_valid_o,
    input weight_rd_ready_i
);

    logic weight_downstream_ready_i, weight_ready_l;
    assign weight_downstream_ready_i = weight_wr_valid_i | weight_ready_l | weight_rd_ready_i;

    logic signed [ACC_WIDTH-1:0] psum [N-1:0];
    logic psum_valid [N-1:0];
    logic any_psum_valid;
    always_comb begin
        any_psum_valid = 1'b0;
        for (int i = 0; i < N; i++)
            any_psum_valid |= psum_valid[i];
    end

    sram_8x256_full #()
    weight_sram_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .downstream_ready_i(weight_downstream_ready_i),
        .downstream_ready_o(weight_downstream_ready_o),
        .rd_valid_o(weight_rd_valid_o),

        .addr_i(weight_addr_i),
        .transaction_amount_i(weight_transaction_amount_i),
        .transaction_rw_mode_i(weight_transaction_rw_mode_i),
        .load_valid_i(weight_load_valid_i),
        .load_ready_o(weight_load_ready_o),

        .wr_data_i(weight_wr_data_i),
        .rd_data_o(weight_rd_data_o)
    );

    // ACTIVATION SRAM IS IN HERE
    scalar_stage_sram #()
    scalar_stage_sram_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .load_bias_en_i(load_bias_en_i),
        .load_zp_en_i(load_zp_en_i),
        .load_scale_en_i(load_scale_en_i),

        .data_i(psum),
        .data_valid_i(any_psum_valid),
        .data_ready_o(),

        .addr_i(act_addr_i),
        .transaction_amount_i(act_transaction_amount_i),
        .transaction_rw_mode_i(act_transaction_rw_mode_i),
        .load_valid_i(act_load_valid_i),
        .load_ready_o(act_load_ready_o),

        .rd_data_o(act_rd_data_o),
        .rd_valid_o(act_rd_valid_o),
        .rd_ready_i(act_rd_ready_i | act_enable_i),

        .wr_data_i(act_wr_data_i),
        .wr_valid_i(act_wr_valid_i),
        .downstream_ready_o(act_downstream_ready_o)
    );

    mxu #()
    mxu_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .act_enable_i(act_enable_i),
        .act_valid_i(act_rd_valid_o),
        .act_bus_i(act_rd_data_o),

        .weight_enable_i(weight_enable_i),
        .weight_valid_i(weight_rd_valid_o),
        .weight_bus_i(weight_rd_data_o),
        .weight_ready_o(weight_ready_l),
        
        .psum_o(psum),
        .psum_valid_o(psum_valid)
    );
    
endmodule