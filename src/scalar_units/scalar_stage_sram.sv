module scalar_stage_sram #(
    parameter int N = 8,
    parameter int BUS_W = 64,
    parameter int DATA_W = 32,
    parameter int FIXED_SHIFT = 16,
    parameter int DATA_Q = 8,
    parameter int counter_width = 8,
    parameter int address_width = 8
) (
    input clk_i,
    input rst_i,

    input load_bias_en_i,
    input load_zp_en_i,
    input load_scale_en_i,

    input signed [DATA_W-1:0] data_i [N-1:0],
    input  data_valid_i,
    output data_ready_o,

    input  [address_width-1:0] addr_i,
    input  [counter_width-1:0] transaction_amount_i,
    input  transaction_rw_mode_i,
    input  load_valid_i,
    output load_ready_o,

    //Control Unit to SRAM
    output [BUS_W-1:0] rd_data_o,
    output rd_valid_o,
    input  rd_ready_i,

    input [BUS_W-1:0] wr_data_i,
    input wr_valid_i,
    output downstream_ready_o
);

    logic signed [DATA_Q-1:0] quantized_data [N-1:0];

    logic quantized_ready, quantized_valid;

    //SRAM Control Signals
    //ASSUME NO DATA RACES
    logic sram_downstream_ready_in;
    assign sram_downstream_ready_in = rd_ready_i | wr_valid_i | quantized_valid;
    logic sram_downstream_ready_out;
    assign downstream_ready_o = sram_downstream_ready_out;

    logic signed [BUS_W-1:0] quantized_flattened, sram_wr_data;
    genvar i;
    generate
        for (i = 0; i < N ; i++) begin
            assign quantized_flattened[i*DATA_Q + (DATA_Q-1):i*DATA_Q] = quantized_data[i];
            //always @(quantized_data[i]) quantized_flattened[i*DATA_Q + (DATA_Q-1):i*DATA_Q] = #200 quantized_data[i];
        end
    endgenerate

    always_comb begin
        sram_wr_data = '0;
        if(quantized_valid)
            sram_wr_data = quantized_flattened;
        else if(wr_valid_i)
            sram_wr_data = wr_data_i;
    end

    activation_sram
        #()
    activation_sram_inst (
        .clk_i(clk_i),
        .rst_i(rst_i),

        .downstream_ready_i(sram_downstream_ready_in),
        .downstream_ready_o(sram_downstream_ready_out),
        .rd_valid_o(rd_valid_o),

        .addr_i(addr_i),
        .transaction_amount_i(transaction_amount_i),
        .transaction_rw_mode_i(transaction_rw_mode_i),
        .load_valid_i(load_valid_i),
        .load_ready_o(load_ready_o),

        .wr_data_i(sram_wr_data),
        .rd_data_o(rd_data_o)
    );

    scalar_stage
        #()
    scalar_stage_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .read_bus_i(rd_data_o),
        .load_valid_i(rd_valid_o),
        .load_bias_en_i(load_bias_en_i),
        .load_zp_en_i(load_zp_en_i),
        .load_scale_en_i(load_scale_en_i),

        .data_i(data_i),
        .data_valid_i(data_valid_i),
        .data_ready_o(data_ready_o),

        .data_o(quantized_data),
        .data_valid_o(quantized_valid),
        .data_ready_i(sram_downstream_ready_out)
    );
    
endmodule
