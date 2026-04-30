module control_top #(
    parameter int SRAM_ADDR_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DRAM_COUNTER_WIDTH = 8,
    parameter int DRAM_DATA_WIDTH = 64,
    parameter int CONTROL_WIDTH = 8,
) (
    input clk_i,
    input rst_i,

    // DMA/DRAM control signals
    output [DRAM_ADDR_WIDTH-1:0] dram_start_addr_o,
    output [DRAM_COUNTER_WIDTH-1:0] dram_word_count_o,
    output dram_we_o, // 0=DRAM->stream, 1=stream->DRAM
    output start_i,
    input busy_o,
    input done_o,

    // DRAM IO streams
    // Read from DRAM
    input dram2sram_valid_i,
    input [DRAM_DATA_WIDTH-1:0] dram2sram_data_i,
    output dram2sram_ready_o,

    // Write to DRAM
    output sram2dram_valid_o,
    output [DRAM_DATA_WIDTH-1:0] sram2dram_data_o,
    input sram2dram_ready_i

    // SPI signals
    input [DRAM_ADDR_WIDTH-1:0] pc_in,
    input pc_valid_i,
    output pc_ready_o,

    // TPU STATE
    output [1:0] tpu_state_o,
    output INTERNAL_ERROR_O
);
    logic buffer_ready_in, buffer_ready_out, buffer_valid_in, buffer_valid_out;
    logic decoder_ready_in, decoder_ready_out, decoder_valid_in, decoder_valid_out;
    logic instruction_valid, instruction_ready;
    always_comb begin : dram_access_mux
        
    end
    /*
    TPU STATE
    */
    typedef enum bit[1:0] 
        {RST        = 2'b00    // rst_i is high
        ,IDLE       = 2'b01    // after reset or after exit
        ,INIT_PC    = 2'b10    // Initalizing Program counter
        ,COMPUTE    = 2'b11    // In compute mode
        }
    tpu_state_t;
    tpu_state_t tpu_state_d;
    assign pc_ready_o = tpu_state_o == IDLE;

    // This should be a case based FSM
    always_comb begin : tpu_state_logic
        tpu_state_d = tpu_state_q;
        if(tpu_state_q == RST & ~rst_i) begin
            tpu_state_d = IDLE;
        end
        if(pc_valid_i & tpu_state_q == IDLE) begin
            tpu_state_d = INIT_PC;
        end
        if(tpu_state_q == INIT_PC & ~buffer_ready_out)begin
            tpu_state_d = COMPUTE;
        end
    end

    always_ff @( posedge clk_i ) begin : tpu_state_dff
        if(rst_i) begin
            tpu_state_o <= RST;
        end
        else begin
            tpu_state_o <= tpu_state_d;
        end
    end
    
    control_sram #(
        .sram_width_p(CONTROL_WIDTH)
    ) control_sram_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .wr_data_i(),
        .valid_i(buffer_valid_out),
        .ready_o(buffer_ready_in),

        .ready_i(instruction_ready),
        .rd_data_o(),
        .valid_o(instruction_valid),

        .is_full_o()
    );

    control_buffer #(
        .DRAM_WIDTH       (DRAM_DATA_WIDTH),
        .CONTROL_WIDTH    (CONTROL_WIDTH),
        .DRAM_ADDR_WIDTH  (DRAM_ADDR_WIDTH)
    ) control_buffer_inst (
        .clk_i            (clk_i),
        .rst_i            (rst_i),

        .pc_in            (pc_in),
        .pc_valid_i       (pc_valid_i),

        .wr_data_i        (dram2sram_data_i),
        .wr_valid_i       (buffer_valid_in),
        .wr_ready_o       (buffer_ready_out),

        .pc_out           (),

        .rd_data_o        (),
        .rd_valid_o       (buffer_valid_out),
        .rd_ready_i       (buffer_ready_in)
    );

    control_decoder #(
        .SRAM_ADDR_WIDTH   (SRAM_ADDR_WIDTH),
        .DRAM_ADDR_WIDTH   (DRAM_ADDR_WIDTH),
        .DRAM_COUNTER_WIDTH(DRAM_COUNTER_WIDTH),
        .DRAM_DATA_WIDTH   (DRAM_DATA_WIDTH)
    ) control_decoder_inst (
        .clk_i                (clk_i),
        .rst_i                (rst_i),

        .instruction_ready_o  (instruction_ready),
        .instruction_data_i   (),
        .instruction_valid_i  (instruction_ready),

        .dram_start_addr_o    (),
        .dram_word_count_o    (),
        .dram_we_o            (),
        .start_i              (),
        .busy_o               (),
        .done_o               (),

        .dram2sram_valid_i    (decoder_valid_in),
        .dram2sram_data_i     (dram2sram_data_i),
        .dram2sram_ready_o    (decoder_ready_out),

        .sram2dram_valid_o    (decoder_valid_out),
        .sram2dram_data_o     (sram2dram_data_o),
        .sram2dram_ready_i    (decoder_ready_in)
    );
endmodule