/* 
TPU side driver for the DRAM bus. This one is built for Wishbone B4 pipelined

Same ctrl and stream interface as classic wb_dma_mater, but now stb can be asserted every cycle and multiple requests can be in flight

Read data is buffered in a small internal FIFO so the bus can run ahead of a backpressuring consumer.
*/

module wb_dma_master #(
    parameter AddrW = 32,
    parameter DataW = 32,
    parameter MAX_OUTSTANDING = 4 //must be power of 2 that is >=2
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
    input  logic               wb_ack_i,
    input  logic               wb_stall_i
);

    localparam BYTE_W = DataW / 8;
    localparam FIFO_AW = $clog2(MAX_OUTSTANDING);

    typedef enum logic [0:0] {
        IDLE,
        ACTIVE
    } state_e;

    state_e              state_q;

    logic [AddrW-1:0] req_addr_q;
    logic [15:0]      req_sent_q;
    logic [15:0]      ack_rcvd_q;
    logic [15:0]      word_count_q;
    logic             we_q;

    logic all_req_sent, all_ack_rcvd;
    assign all_req_sent = (req_sent_q == word_count_q);
    assign all_ack_rcvd = (ack_rcvd_q == word_count_q);

    //Read FIFO (circular buffer
    
    logic [DataW-1:0]  fifo_mem [MAX_OUTSTANDING];
    logic [FIFO_AW:0]  fifo_wptr_q, fifo_rptr_q;
 
    logic fifo_empty;
    assign fifo_empty = (fifo_wptr_q == fifo_rptr_q);
 
    logic fifo_push, fifo_pop;
    assign fifo_push = wb_ack_i & ~we_q & (state_q == ACTIVE);
    assign fifo_pop  = rd_valid_o & rd_ready_i;

    //decremented on request issue, incremented on consumer pop. credit + in_flight + fifo_occupancy = MAX_OUTSTANDING
    logic [$clog2(MAX_OUTSTANDING):0] credit_q;

    logic req_accepted;
    logic can_issue;
 
    assign req_accepted = wb_stb_o & ~wb_stall_i;
    assign can_issue    = (state_q == ACTIVE) & ~all_req_sent &
                          (we_q ? wr_valid_i : (credit_q > 0));

    assign busy_o = (state_q != IDLE);

    logic go_idle;
    assign go_idle = all_ack_rcvd & (we_q | fifo_empty);
    assign done_o  = (state_q == ACTIVE) & go_idle;

    assign wb_adr_o = req_addr_q;
    assign wb_sel_o = '1;
    assign wb_we_o  = we_q;
    assign wb_dat_o = wr_data_i; // combinational pass-through for writes
    assign wb_cyc_o = (state_q == ACTIVE);
    assign wb_stb_o = can_issue;

    assign rd_valid_o = ~fifo_empty;
    assign rd_data_o  = fifo_mem[fifo_rptr_q[FIFO_AW-1:0]];
    assign wr_ready_o = req_accepted & we_q;

    always_comb begin
        state_d = state_q;
        req_addr_d  = req_addr_q;
        req_sent_d  = req_sent_q;
        ack_rcvd_d  = ack_rcvd_q;
        word_count_d = word_count_q;
        we_d        = we_q;
        fifo_wptr_d = fifo_wptr_q;
        fifo_rptr_d = fifo_rptr_q;
        credit_d    = credit_q;

        case (state_q)
            IDLE: begin
                if (start_i && word_count_i != 16'd0) begin
                    state_d      = ACTIVE;
                    req_addr_d   = start_addr_i;
                    word_count_d = word_count_i;
                    we_d         = we_i;
                    req_sent_d   = '0;
                    ack_rcvd_d   = '0;
                    fifo_wptr_d  = '0;
                    fifo_rptr_d  = '0;
                    credit_d     = MAX_OUTSTANDING[FIFO_AW:0];
                end
            end

            ACTIVE: begin
                // request side, so advance on accept
                if (req_accepted) begin
                    req_addr_d = req_addr_q + AddrW'(BYTE_W);
                    req_sent_d = req_sent_q + 16'd1;
                end

                // response side: count acks
                if (wb_ack_i) begin
                    ack_rcvd_d = ack_rcvd_q + 16'd1;
                end

                // read FIFO pointers
                if (fifo_push) begin
                    fifo_wptr_d = fifo_wptr_q + 1;
                end
                if (fifo_pop) begin
                    fifo_rptr_d = fifo_rptr_q + 1;
                end

                // credit counter
                case ({(req_accepted & ~we_q), fifo_pop})
                    2'b10:   credit_d = credit_q - 1;
                    2'b01:   credit_d = credit_q + 1;
                    default: ; // both or neither: no net change
                endcase

                if (go_idle) begin
                    state_d = IDLE;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            state_q      <= IDLE;
            req_addr_q   <= '0;
            req_sent_q   <= '0;
            ack_rcvd_q   <= '0;
            word_count_q <= '0;
            we_q         <= 1'b0;
            fifo_wptr_q  <= '0;
            fifo_rptr_q  <= '0;
            credit_q     <= MAX_OUTSTANDING[FIFO_AW:0];
        end else begin
            state_q      <= state_d;
            req_addr_q   <= req_addr_d;
            req_sent_q   <= req_sent_d;
            ack_rcvd_q   <= ack_rcvd_d;
            word_count_q <= word_count_d;
            we_q         <= we_d;
            fifo_wptr_q  <= fifo_wptr_d;
            fifo_rptr_q  <= fifo_rptr_d;
            credit_q     <= credit_d;
        end
    end

    // for FIFO memory array
    always_ff @(posedge clk_i) begin
        if (fifo_push) begin
            fifo_mem[fifo_wptr_q[FIFO_AW-1:0]] <= wb_dat_i;
        end
    end

endmodule