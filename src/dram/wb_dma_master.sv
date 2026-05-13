/* 
TPU side driver for the DRAM bus. On start_i, moves word_count_i

DataW bit words between DRAM wb slave and a valid/ready stream:
   we_i=0  read : DRAM -> (rd_data_o, rd_valid_o/rd_ready_i)
   we_i=1  write: (wr_data_i, wr_valid_i/wr_ready_o) -> DRAM

 data stream side backpressure. wb side classic

 note: this is a frontend for the eventual Gmem2Smem / Smem2Gmem. Hook the SRAM's memory_transaction onto the stream side. */

module wb_dma_master #(
    parameter AddrW = 32,
    parameter DataW = 32
)(
    input  logic               clk_i,
    input  logic               rst_i,

    // control
    input  logic [AddrW-1:0]   start_addr_i, // byte addr, DataW-aligned
    input  logic [15:0]        word_count_i,
    input  logic               we_i, // 0=DRAM->stream, 1=stream->DRAM
    input  logic               start_i,
    output logic               busy_o,
    output logic               done_o,

    // stream side
    
    output logic [DataW-1:0]   rd_data_o, //read
    output logic               rd_valid_o,
    input  logic               rd_ready_i,

    input  logic [DataW-1:0]   wr_data_i, //write
    input  logic               wr_valid_i,
    output logic               wr_ready_o,

    // wb master
    output logic [AddrW-1:0]   wb_adr_o,
    output logic [DataW-1:0]   wb_dat_o,
    input  logic [DataW-1:0]   wb_dat_i,
    output logic               wb_we_o,
    output logic               wb_stb_o,
    output logic               wb_cyc_o,
    output logic [DataW/8-1:0] wb_sel_o,
    input  logic               wb_ack_i
);

    // localparam BYTE_W = DataW / 8;
    localparam BYTE_W = 1;

    typedef enum logic [1:0] {
        IDLE,
        REQ,    // drive cyc/stb (write waits on wr_valid but read can always go)
        HOLD    // read only. present captured word until consumer accepts
    } state_e;

    state_e              state_q, state_d;
    logic [AddrW-1:0]    addr_q,  addr_d;
    logic [15:0]         count_q, count_d;
    logic                we_q;
    logic [DataW-1:0]    rd_data_q;

    logic last_word;
    assign last_word = (count_q == 16'd1);

    assign busy_o = (state_q != IDLE);
    assign done_o = busy_o && (state_d == IDLE);

    assign wb_adr_o = addr_q;
    assign wb_sel_o = '1;
    assign wb_we_o  = we_q;
    assign wb_dat_o = wr_data_i; // only when we_q && wr_valid_i

    always_comb begin
        state_d    = state_q;
        addr_d     = addr_q;
        count_d    = count_q;
        wb_stb_o   = 1'b0;
        wb_cyc_o   = 1'b0;
        wr_ready_o = 1'b0;
        rd_valid_o = 1'b0;
        rd_data_o  = rd_data_q;

        case (state_q)
            IDLE: begin
                if (start_i && word_count_i != 16'd0) state_d = REQ;
            end

            REQ: begin
                if (we_q) begin
                    wb_cyc_o   = wr_valid_i;
                    wb_stb_o   = wr_valid_i;
                    wr_ready_o = wb_ack_i;
                    // only when the producer->DMA and DMA->slave
                    // handshakes both complete in the same cycle
                    if (wb_ack_i && wr_valid_i) begin
                        addr_d  = addr_q + BYTE_W;
                        count_d = count_q - 16'd1;
                        if (last_word) state_d = IDLE;
                        else           state_d = REQ;
                    end
                end else begin
                    // read: fetch, then hold for consumer
                    wb_cyc_o = 1'b1;
                    wb_stb_o = 1'b1;
                    if (wb_ack_i) state_d = HOLD;
                end
            end

            HOLD: begin
                rd_valid_o = 1'b1;
                if (rd_ready_i) begin
                    addr_d  = addr_q + BYTE_W;
                    count_d = count_q - 16'd1;
                    if (last_word) state_d = IDLE;
                    else           state_d = REQ;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            state_q   <= IDLE;
            addr_q    <= '0;
            count_q   <= '0;
            we_q      <= 1'b0;
            rd_data_q <= '0;
        end else begin
            state_q <= state_d;

            // latch control on kick, else follow fsm
            if (state_q == IDLE && start_i && word_count_i != 16'd0) begin
                addr_q  <= start_addr_i;
                count_q <= word_count_i;
                we_q    <= we_i;
            end else begin
                addr_q  <= addr_d;
                count_q <= count_d;
            end

            // capture read data on ack
            if (state_q == REQ && !we_q && wb_ack_i) begin
                rd_data_q <= wb_dat_i;
            end
        end
    end

endmodule