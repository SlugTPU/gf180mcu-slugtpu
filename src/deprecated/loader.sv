module loader #(
    parameter int N          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int K          = 4          // inner dimension for GEMM tile: A[NxK] * B[KxN]
)(
    input  logic                                        clk_i,
    input  logic                                        rst_i,

    // control
    input  logic                                        start_i,
    input  logic                                        buf_sel_i,
    output logic                                        busy_o,
    output logic                                        done_o,

    // Flattened tile inputs (Icarus-compatible):
    //   A_flat: A[N][K] packed as [N*K*DATA_WIDTH-1:0], row-major (A[0][0] at LSB)
    //   B_flat: B[K][N] packed as [K*N*DATA_WIDTH-1:0], row-major (B[0][0] at LSB)
    input  logic [N*K*DATA_WIDTH-1:0]                   A_flat,
    input  logic [K*N*DATA_WIDTH-1:0]                   B_flat,

    // outputs to systolic_array
    output logic [N*DATA_WIDTH-1:0]                     act_out,    // act_in[N]
    output logic [N*DATA_WIDTH-1:0]                     weight_out, // weight_in[N]
    output logic [N-1:0]                                weight_we,
    output logic [N-1:0]                                buf_sel_out
);

    // ── Helper functions to index flattened arrays ──────────────────────────
    // A_flat layout: A[row][ki] lives at bits [(row*K + ki)*DW +: DW]
    // B_flat layout: B[ki][col] lives at bits [(ki*N + col)*DW +: DW]

    typedef enum logic [1:0] {IDLE, LOAD_W, STREAM_A, DONE} state_t;
    state_t state_q, state_d;

    logic [$clog2(K+1)-1:0] k_q, k_d;
    logic [$clog2(N+1)-1:0] flush_q, flush_d;

    // Extract a single DATA_WIDTH-wide element from a flattened array
    function automatic logic [DATA_WIDTH-1:0] get_A(
        input logic [N*K*DATA_WIDTH-1:0] flat,
        input int                         row,
        input int                         ki
    );
        int base;
        base   = (row * K + ki) * DATA_WIDTH;
        get_A  = flat[base +: DATA_WIDTH];
    endfunction

    function automatic logic [DATA_WIDTH-1:0] get_B(
        input logic [K*N*DATA_WIDTH-1:0] flat,
        input int                         ki,
        input int                         col
    );
        int base;
        base  = (ki * N + col) * DATA_WIDTH;
        get_B = flat[base +: DATA_WIDTH];
    endfunction

    // ── Combinational logic ─────────────────────────────────────────────────
    always_comb begin
        state_d  = state_q;
        k_d      = k_q;
        flush_d  = flush_q;

        done_o   = 1'b0;
        busy_o   = (state_q != IDLE);

        for (int c = 0; c < N; c++) begin
            buf_sel_out[c]                              = buf_sel_i;
            act_out   [c*DATA_WIDTH +: DATA_WIDTH]      = '0;
            weight_out[c*DATA_WIDTH +: DATA_WIDTH]      = '0;
            weight_we [c]                               = 1'b0;
        end

        unique case (state_q)
            IDLE: begin
                if (start_i) begin
                    k_d     = '0;
                    flush_d = '0;
                    state_d = LOAD_W;
                end
            end

            LOAD_W: begin
                for (int col = 0; col < N; col++) begin
                    weight_out[col*DATA_WIDTH +: DATA_WIDTH] = get_B(B_flat, int'(k_q), col);
                    weight_we[col]                           = 1'b1;
                end

                if (k_q == K[$clog2(K+1)-1:0] - 1) begin
                    k_d     = '0;
                    flush_d = '0;
                    state_d = STREAM_A;
                end else begin
                    k_d = k_q + 1;
                end
            end

            STREAM_A: begin
                if (k_q < K[$clog2(K+1)-1:0]) begin
                    for (int row = 0; row < N; row++) begin
                        act_out[row*DATA_WIDTH +: DATA_WIDTH] = get_A(A_flat, row, int'(k_q));
                    end
                    k_d = k_q + 1;
                end else begin
                    // flush zeros (defaults already set above)
                    if (flush_q == (N[$clog2(N+1)-1:0] - 2)) begin
                        state_d = DONE;
                    end else begin
                        flush_d = flush_q + 1;
                    end
                end
            end

            DONE: begin
                done_o  = 1'b1;
                state_d = IDLE;
            end

            default: state_d = IDLE;
        endcase
    end

    // ── Sequential logic ────────────────────────────────────────────────────
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            state_q <= IDLE;
            k_q     <= '0;
            flush_q <= '0;
        end else begin
            state_q <= state_d;
            k_q     <= k_d;
            flush_q <= flush_d;
        end
    end

endmodule