module control_top #(
    parameter int SRAM_ADDR_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DRAM_COUNTER_WIDTH = 8,
    parameter int DRAM_DATA_WIDTH = 64,
    parameter int CONTROL_WIDTH = 8
) (
    input clk_i,
    input rst_i,

    // DMA/DRAM control signals
    output logic [DRAM_ADDR_WIDTH-1:0] dram_start_addr_o,
    output logic [DRAM_COUNTER_WIDTH-1:0] dram_word_count_o,
    output logic dram_we_o,
    output logic dram_start_o,
    input  dma_busy_i,
    input  dma_done_i,

    // DRAM IO streams
    // Read from DRAM
    input  dram2sram_valid_i,
    input  [DRAM_DATA_WIDTH-1:0] dram2sram_data_i,
    output logic dram2sram_ready_o,

    // Write to DRAM
    output logic sram2dram_valid_o,
    output logic [DRAM_DATA_WIDTH-1:0] sram2dram_data_o,
    input  sram2dram_ready_i,

    // SPI signals
    input  [DRAM_ADDR_WIDTH-1:0] pc_in,
    input  pc_valid_i,
    output pc_ready_o,

    // TPU STATE
    output logic [1:0] tpu_state_o,
    output logic INTERNAL_ERROR_O
);

    logic buffer_ready_in, buffer_ready_out, buffer_valid_in, buffer_valid_out;
    logic instruction_valid, instruction_ready, tpu_exit;

    logic [DRAM_ADDR_WIDTH-1:0] pc_out;

    logic [DRAM_ADDR_WIDTH-1:0]   dec_dram_start_addr;
    logic [DRAM_COUNTER_WIDTH-1:0] dec_dram_word_count;
    logic dec_dram_we;
    logic dec_dram_start;
    logic dec_dram2sram_valid;
    logic dec_dram2sram_ready;
    logic dec_sram2dram_valid;
    logic [7:0] buffer_data_out, instruction_data;
    logic dec_sram2dram_ready;
    logic sram_is_full;

    typedef enum bit [1:0] {
        RST      = 2'b00,
        IDLE     = 2'b01,
        INIT_PC  = 2'b10,
        COMPUTE  = 2'b11
    } tpu_state_t;

    tpu_state_t tpu_state_q, tpu_state_d;

    assign tpu_state_o = tpu_state_q;
    assign pc_ready_o  = (tpu_state_q == IDLE);

    always_comb begin : tpu_state_logic
        tpu_state_d = tpu_state_q;
        case (tpu_state_q)
            RST: begin
                if (~rst_i)
                    tpu_state_d = IDLE;
            end
            IDLE: begin
                if (pc_valid_i)
                    tpu_state_d = INIT_PC;
            end
            INIT_PC: begin
                if (~dma_busy_i& sram_is_full)
                    tpu_state_d = COMPUTE;
            end
            COMPUTE: begin
                if (tpu_exit)
                    tpu_state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i) begin : tpu_state_dff
        if (rst_i)
            tpu_state_q <= RST;
        else
            tpu_state_q <= tpu_state_d;
    end

    logic decoder_owns_dma;
    assign decoder_owns_dma = dec_dram_start;

    logic [DRAM_ADDR_WIDTH-1:0]    buf_dram_start_addr;
    logic [DRAM_COUNTER_WIDTH-1:0] buf_dram_word_count;
    logic buf_dram_start, dec_inst_ready;

    assign buf_dram_start_addr = pc_out;
    assign buf_dram_word_count = (tpu_state_q == INIT_PC) ? 8'd255 : 8'd1;
    assign buf_dram_start = ~dec_dram_start & ~dma_busy_i & buffer_ready_out &
                            (tpu_state_q == INIT_PC || tpu_state_q == COMPUTE);
    assign instruction_ready = (tpu_state_q == COMPUTE) ? dec_inst_ready : 1'b0;

    always_comb begin : dram_access_mux
        if (decoder_owns_dma) begin
            dram_start_addr_o  = dec_dram_start_addr;
            dram_word_count_o  = dec_dram_word_count;
            dram_we_o          = dec_dram_we;
            dram_start_o       = dec_dram_start;

            dram2sram_ready_o  = dec_dram2sram_ready;
            dec_dram2sram_valid = dram2sram_valid_i;

            sram2dram_valid_o  = dec_sram2dram_valid;
            dec_sram2dram_ready = sram2dram_ready_i;

            buffer_valid_in    = 1'b0;

        end else begin
            dram_start_addr_o  = buf_dram_start_addr;
            dram_word_count_o  = buf_dram_word_count;
            dram_we_o          = 1'b0;  
            dram_start_o       = buf_dram_start;

            dram2sram_ready_o  = buffer_ready_out;
            buffer_valid_in    = dram2sram_valid_i;

            dec_dram2sram_valid = 1'b0;
            sram2dram_valid_o  = 1'b0;
            dec_sram2dram_ready = 1'b0;
        end
    end

    control_sram #(
        .sram_width_p(CONTROL_WIDTH)
    ) control_sram_inst (
        .clk_i   (clk_i),
        .rst_i   (rst_i),

        .wr_data_i (buffer_data_out),
        .valid_i   (buffer_valid_out),
        .ready_o   (buffer_ready_in),

        .ready_i   (instruction_ready),
        .rd_data_o (instruction_data),
        .valid_o   (instruction_valid),

        .is_full_o (sram_is_full)
    );

    control_buffer #(
        .DRAM_WIDTH      (DRAM_DATA_WIDTH),
        .CONTROL_WIDTH   (CONTROL_WIDTH),
        .DRAM_ADDR_WIDTH (DRAM_ADDR_WIDTH)
    ) control_buffer_inst (
        .clk_i       (clk_i),
        .rst_i       (rst_i),

        .pc_in       (pc_in),
        .pc_valid_i  (pc_valid_i),

        .wr_data_i   (dram2sram_data_i),
        .wr_valid_i  (buffer_valid_in),
        .wr_ready_o  (buffer_ready_out),

        .pc_out      (pc_out),

        .rd_data_o   (buffer_data_out),
        .rd_valid_o  (buffer_valid_out),
        .rd_ready_i  (buffer_ready_in)
    );

    control_decoder #(
        .SRAM_ADDR_WIDTH    (SRAM_ADDR_WIDTH),
        .DRAM_ADDR_WIDTH    (DRAM_ADDR_WIDTH),
        .DRAM_COUNTER_WIDTH (DRAM_COUNTER_WIDTH),
        .DRAM_DATA_WIDTH    (DRAM_DATA_WIDTH)
    ) control_decoder_inst (
        .clk_i               (clk_i),
        .rst_i               (rst_i),

        .instruction_ready_o (dec_inst_ready),
        .instruction_data_i  (instruction_data),
        .instruction_valid_i (instruction_valid),

        .dram_start_addr_o   (dec_dram_start_addr),
        .dram_word_count_o   (dec_dram_word_count),
        .dram_we_o           (dec_dram_we),
        .dram_start_i        (dec_dram_start),
        .dma_busy            (dma_busy_i),
        .dma_done            (dma_done_i),

        .dram2sram_valid_i   (dec_dram2sram_valid),
        .dram2sram_data_i    (dram2sram_data_i),
        .dram2sram_ready_o   (dec_dram2sram_ready),

        .sram2dram_valid_o   (dec_sram2dram_valid),
        .sram2dram_data_o    (sram2dram_data_o),
        .sram2dram_ready_i   (dec_sram2dram_ready),

        .tpu_state_i         (tpu_state_o),
        .tpu_exit_o          (tpu_exit),
        .INTERNAL_ERROR_O    (INTERNAL_ERROR_O)
    );

endmodule