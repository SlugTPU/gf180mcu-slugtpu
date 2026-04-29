module control_decoder #(
    parameter int SRAM_ADDR_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DRAM_COUNTER_WIDTH = 8,
    parameter int DRAM_DATA_WIDTH = 64,
    parameter int INST_MAX_WIDTH_BYTES = 8
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
    input sram2dram_ready_i,

    output logic [2:0] tpu_state_o,
    output logic INTERNAL_ERROR_O
);
    logic [INST_MAX_WIDTH_BYTES * 8 - 1 : 0] inst_q, inst_d;
    // Scalar stage controls
    logic                        load_bias_en_q, load_bias_en_d;
    logic                        load_zp_en_q, load_zp_en_d;
    logic                        load_scale_en_q, load_scale_en_d;
    logic                        relu_enable_q, relu_enable_d;

    // MXU enables
    logic                        act_enable_q, act_enable_d;
    logic                        weight_enable_q, weight_enable_d;

    // Activation SRAM control
    logic [SRAM_ADDR_WIDTH-1:0]  act_addr;
    logic [SRAM_ADDR_WIDTH-1:0]  act_transaction_amount;
    logic                        act_transaction_rw_mode;
    logic                        act_load_valid;
    logic                        act_load_ready;

    // Weight SRAM control
    logic [SRAM_ADDR_WIDTH-1:0]  weight_addr;
    logic [SRAM_ADDR_WIDTH-1:0]  weight_transaction_amount;
    logic                        weight_transaction_rw_mode;
    logic                        weight_load_valid;
    logic                        weight_load_ready;

    // Activation write channel
    logic [63:0]                 act_wr_data;
    logic                        act_wr_valid;
    logic                        act_downstream_ready;

    // Activation read channel
    logic [63:0]                 act_rd_data;
    logic                        act_rd_valid;
    logic                        act_rd_ready;

    // Weight write channel
    logic [63:0]                 weight_wr_data;
    logic                        weight_wr_valid;
    logic                        weight_downstream_ready;

    // Weight read channel
    logic [63:0]                 weight_rd_data;
    logic                        weight_rd_valid;
    logic                        weight_rd_ready;

    /*
    TPU STATE
    */
    enum bit[2:0] 
        {RST        = 2'b00    // rst_i is high
        ,IDLE       = 2'b01    // after reset or after exit
        ,INIT_PC    = 2'b10    // Initalizing Program counter
        ,COMPUTE    = 2'b11    // In compute mode
        }
    tpu_state_t;
    /*
    Instruction FSM (Mealy)
    Determine Opcode -> LOAD -> WAIT -> EXECUTE -> return to first state
    */
    enum bit[3:0] 
        {IDLE           = 3'b000,
         LOAD_OPCODE    = 3'b001,
         LOAD_ALL       = 3'b010,
         WAIT           = 3'b011,
         EXECUTE        = 3'b100
        }
    decoder_state_t;
    logic [3:0] decoder_state_l;
    /*
    OP CODES
    codes that start with 1 depend on act_load_ready
    codes that start with 01 depend on weight_load_ready
    codes that end in 1 depend on dma singals
    */
    enum bit[4:0] 
        {EXIT               = 4'b0000
        ,DRAM2SRAM_ACT      = 4'b1101
        ,DRAM2SRAM_WEIGHT   = 4'b0101
        ,SRAM2DRAM          = 4'b1111
        ,LOAD_BIAS          = 4'b1000
        ,LOAD_ZP            = 4'b1100
        ,LOAD_SCALE         = 4'b1110
        ,MATMUL             = 4'b1010
        ,LOAD_WEIGHTS       = 4'b0110
        ,PIPELINE_SETUP     = 4'b0010
        }
    op_code_t;

    always_ff @( posedge clk_i ) begin : tpu_state_block
        if(rst_i)
            tpu_state_o <= RST;
        // TODO: else
    end

    always_ff @( posedge clk_i ) begin : decoder_state_block
        if(rst_i)
            decoder_state_l <= RST;
        // TODO: else
    end

    always_ff @( posedge clk_i ) begin : enable_block
        if(rst_i) begin
            load_bias_en_q <= '0;
            load_zp_en_q <= '0;
            load_scale_en_q <= '0;
            relu_enable_q <= '0;
            act_enable_q <= '0;
            weight_enable_q <= '0;
        end
        else begin
            load_bias_en_q <= load_bias_en_d;
            load_zp_en_q <= load_zp_en_d;
            load_scale_en_q <= load_scale_en_d;
            relu_enable_q <= relu_enable_d;
            act_enable_q <= act_enable_d;
            weight_enable_q <= weight_enable_d;
        end
    end

    always_ff @( posedge clk_i ) begin : instruction_dff
        if(rst_i) begin
            inst_q <= '0;
        end else begin
            inst_q <= inst_d;
        end
    end

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

        .load_bias_en_i               (load_bias_en_q),
        .load_zp_en_i                 (load_zp_en_q),
        .load_scale_en_i              (load_scale_en_q),
        .relu_enable_i                (relu_enable_q),

        .act_enable_i                 (act_enable_q),
        .weight_enable_i              (weight_enable_q),

        .act_addr_i                   (act_addr),
        .act_transaction_amount_i     (act_transaction_amount),
        .act_transaction_rw_mode_i    (act_transaction_rw_mode),
        .act_load_valid_i             (act_load_valid),
        .act_load_ready_o             (act_load_ready),

        .weight_addr_i                (weight_addr),
        .weight_transaction_amount_i  (weight_transaction_amount),
        .weight_transaction_rw_mode_i (weight_transaction_rw_mode),
        .weight_load_valid_i          (weight_load_valid),
        .weight_load_ready_o          (weight_load_ready),

        .act_wr_data_i                (act_wr_data),
        .act_wr_valid_i               (act_wr_valid),
        .act_downstream_ready_o       (act_downstream_ready),

        .act_rd_data_o                (act_rd_data),
        .act_rd_valid_o               (act_rd_valid),
        .act_rd_ready_i               (act_rd_ready),

        .weight_wr_data_i             (weight_wr_data),
        .weight_wr_valid_i            (weight_wr_valid),
        .weight_downstream_ready_o    (weight_downstream_ready),

        .weight_rd_data_o             (weight_rd_data),
        .weight_rd_valid_o            (weight_rd_valid),
        .weight_rd_ready_i            (weight_rd_ready)
    );
    
endmodule