module control_top #(
    parameter int SRAM_ADDR_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DRAM_COUNTER_WIDTH = 8,
    parameter int DRAM_DATA_WIDTH = 64,
    parameter int TPU_STATE_WIDTH = 3,
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
    // TODO: SPI out instruction

    // TPU STATE
    output logic [TPU_STATE_WIDTH-1:0] tpu_state_o
    // output INTERNAL_ERROR_O
);
    enum bit[TPU_STATE_WIDTH-1:0] 
        {RST        = 3'b000    // rst_i is high
        ,IDLE       = 3'b001    // after reset or after exit
        ,INIT_PC    = 3'b010    // Initalizing Program counter
        ,COMPUTE    = 3'b011    // In compute mode
        ,TO_SPI     = 3'b100}   // Sending activation address to SPI
    tpu_state_t;

    always_ff @( posedge clk_i ) begin : state_block
        if(rst_i)
            tpu_state_o <= RST;
        // TODO: else
    end
    
    control_sram #(
        .sram_width_p(CONTROL_WIDTH)
    ) control_sram_inst(
        .clk_i(clk_i),
        .rst_i(rst_i),

        .wr_data_i(),
        .valid_i(),
        .ready_o(),

        .ready_i(),
        .rd_data_o(),
        .valid_o(),

        .is_full_o()
    );

    control_buffer #(
        .DRAM_WIDTH       (),
        .CONTROL_WIDTH    (),
        .DRAM_ADDR_WIDTH  (),
        .DEPTH_LOG2_P     ()
    ) control_buffer_inst (
        .clk_i            (),
        .rst_i            (),

        .pc_in            (),
        .pc_valid_i       (),

        .wr_data_i        (),
        .wr_valid_i       (),
        .wr_ready_o       (),

        .pc_out           (),

        .rd_data_o        (),
        .rd_valid_o       (),
        .rd_ready_i       ()
    );

    control_decoder #(
        .SRAM_ADDR_WIDTH   (),
        .DRAM_ADDR_WIDTH   (),
        .DRAM_COUNTER_WIDTH(),
        .DRAM_DATA_WIDTH   ()
    ) control_decoder_inst (
        .clk_i                (),
        .rst_i                (),

        .instruction_ready_o  (),
        .instruction_data_i   (),
        .instruction_valid_i  (),

        .dram_start_addr_o    (),
        .dram_word_count_o    (),
        .dram_we_o            (),
        .start_i              (),
        .busy_o               (),
        .done_o               (),

        .dram2sram_valid_i    (),
        .dram2sram_data_i     (),
        .dram2sram_ready_o    (),

        .sram2dram_valid_o    (),
        .sram2dram_data_o     (),
        .sram2dram_ready_i    ()
    );
endmodule