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

    input [2:0] tpu_state_i,
    output tpu_exit_o,
    output logic INTERNAL_ERROR_O
);
    logic [INST_MAX_WIDTH_BYTES * 8 - 1 : 0] inst_q, inst_d;
    
    logic [SRAM_ADDR_WIDTH-1:0] inst_sram_addr, inst_result_sram_addr;
    logic [DRAM_ADDR_WIDTH-1:0] inst_dram_addr;
    logic [3:0] inst_relu_mode;
    logic [1:0] inst_load_count_q, inst_load_count_d;
    op_code_t inst_opcode;
    logic [DRAM_COUNTER_WIDTH-1:0] inst_pipeline_amount , inst_dma_amount ;

    assign inst_sram_addr           = inst_q[11:4];
    assign inst_dram_addr           = inst_q[23:12];
    assign inst_result_sram_addr    = inst_q[19:12];
    assign inst_relu_mode           = inst_q[7:4];
    assign inst_pipeline_amount     = inst_q[15:8];
    assign inst_dma_amount          = inst_q[31:24];

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

    assign act_wr_data = dram2sram_data_i;
    assign weight_wr_data = dram2sram_data_i;
    assign sram2dram_data_o = (act_enable_q) ? act_wr_data : weight_wr_data;

    assign act_wr_valid = (act_enable_q) ? dram2sram_valid_i : '0;
    assign weight_wr_valid = (weight_enable_q) ? dram2sram_valid_i : '0;
    assign dram2sram_ready_o = (act_enable_q) ? act_downstream_ready : weight_downstream_ready;

    assign act_rd_ready = (act_enable_q) ? sram2dram_ready_i : '0;
    assign weight_rd_ready = (weight_enable_q) ? sram2dram_ready_i : '0;
    assign sram2dram_valid_o = (act_enable_q) ? act_rd_valid : weight_rd_valid;
    assign inst_opcode = inst_q[3:0];

    /*
    Instruction FSM (Mealy)
    Determine Opcode -> LOAD -> WAIT -> EXECUTE -> return to first state
    */
    enum bit[1:0] 
        {LOAD_OPCODE    = 2'b00,
         LOAD_ALL       = 2'b01,
         WAIT           = 2'b10,
         WAIT_MATMUL    = 2'b11
        }
    decoder_state_t;
    decoder_state_t decoder_state_q, decoder_state_d;
    /*
    OP CODES
    codes that start with 1 depend on act_load_ready
    codes that start with 01 depend on weight_load_ready
    codes that end in 1 depend on dma singals
    matmul is unique because its a two part instruction
    */
    enum bit[4:0] 
        {EXIT               = 4'b0000

        ,SRAM2DRAM          = 4'b1111
        ,DRAM2SRAM_ACT      = 4'b1101
        ,DRAM2SRAM_WEIGHT   = 4'b0101

        ,LOAD_WEIGHTS       = 4'b0110

        ,LOAD_BIAS          = 4'b1000
        ,LOAD_ZP            = 4'b1100
        ,LOAD_SCALE         = 4'b1010
        ,PIPELINE_SETUP     = 4'b1110

        ,MATMUL             = 4'b0001
        }
    op_code_t;

    always_ff @( posedge clk_i ) begin : d_q_block
        if(rst_i) begin
            decoder_state_q <= LOAD_OPCODE;
            inst_q <= '0;

            load_bias_en_q <= '0;
            load_zp_en_q <= '0;
            load_scale_en_q <= '0;
            relu_enable_q <= '0;
            act_enable_q <= '0;
            weight_enable_q <= '0;
            inst_load_count_q <= '0;
        end
        else begin
            decoder_state_q <= decoder_state_d;
            inst_q <= inst_d;

            load_bias_en_q <= load_bias_en_d;
            load_zp_en_q <= load_zp_en_d;
            load_scale_en_q <= load_scale_en_d;
            relu_enable_q <= relu_enable_d;
            act_enable_q <= act_enable_d;
            weight_enable_q <= weight_enable_d;
            inst_load_count_q <= inst_load_count_d;
        end
    end

    logic do_execute;

    always_comb begin : instruction_blk
        inst_d = inst_q;
        inst_load_count_d = inst_load_count_q;
        do_execute = '0;
        tpu_exit_o = '0;
        decoder_state_d = decoder_state_q;
        case(decoder_state_q)
            LOAD_OPCODE : begin
                if (instruction_valid_i) begin
                    inst_d[7 : 0] = instruction_data_i;
                    inst_load_count_d = 2'b001;
                    decoder_state_d = LOAD_ALL;
                end
            end
            LOAD_ALL : begin
                if (instruction_valid_i) begin
                    if (inst_opcode[3] = 1'b1 || inst_opcode == LOAD_WEIGHTS) begin
                        decoder_state_d = WAIT;
                    end
                    if (inst_opcode[0] == 1'b1 && inst_load_count_d == 2'b11) begin // DRAM
                        decoder_state_d = WAIT;
                    end
                    if (inst_opcode == MATMUL && inst_load_count_d == 2'b10) begin
                        decoder_state_d = WAIT;
                    end
                    inst_d[inst_load_count_q*8 +: 8] = instruction_data_i;
                    inst_load_count_d = inst_load_count_d + 1'b1;
                end
            end
            WAIT : begin
                if (inst_opcode[3:2] == 2'b01 && weight_load_ready) begin
                   do_execute = '1;
                   decoder_state_d = LOAD_OPCODE;
                end
                if (inst_opcode[3] == 1'b1 && act_load_ready) begin
                   do_execute = '1;
                   decoder_state_d = LOAD_OPCODE;
                end
                if (inst_opcode == MATMUL && act_load_ready) begin
                   do_execute = '1;
                   decoder_state_d = WAIT_MATMUL;
                end
                if (inst_opcode == EXIT) begin
                    decoder_state_d = LOAD_OPCODE;
                    tpu_exit_o = '1;
                end
            end
            WAIT_MATMUL : begin
                if (act_load_ready) begin
                    do_execute = '1;
                    decoder_state_d = LOAD_OPCODE;
                end
            end
        endcase
    end

    always_comb begin : execute_blk
        if (do_execute == 1'b1) begin
            case (inst_opcode)
                // EXIT: TODO
                SRAM2DRAM : begin
                    
                end
                DRAM2SRAM_ACT : begin
                    
                end
                DRAM2SRAM_WEIGHT : begin
                    
                end
                LOAD_WEIGHTS : begin
                    
                end
                LOAD_BIAS : begin
                    
                end
                LOAD_ZP : begin
                    
                end
                LOAD_SCALE : begin
                    
                end
                PIPELINE_SETUP : begin
                    
                end
                MATMUL : begin
                    
                end
            endcase
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