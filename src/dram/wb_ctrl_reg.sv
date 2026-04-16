// wb control register facing host

module wb_ctrl_reg (
    input  logic        clk_i,
    input  logic        rst_i,

    // wb slave
    input  logic [31:0] adr_i,
    input  logic [31:0] dat_i,
    output logic [31:0] dat_o,
    input  logic        we_i,
    input  logic        stb_i,
    input  logic        cyc_i,
    input  logic [3:0]  sel_i,
    output logic        ack_o,

    output logic        tpu_active_o
);

    logic [0:0] ctrl_q;

    assign tpu_active_o = ctrl_q[0];
    assign dat_o        = {31'b0, ctrl_q[0]};

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ack_o <= 1'b0;
        end else begin
            ack_o <= cyc_i & stb_i & ~ack_o;
        end
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ctrl_q <= 1'b0;
        end else if (cyc_i & stb_i & we_i & ~ack_o & sel_i[0]) begin
            ctrl_q[0] <= dat_i[0];
        end
    end

endmodule
