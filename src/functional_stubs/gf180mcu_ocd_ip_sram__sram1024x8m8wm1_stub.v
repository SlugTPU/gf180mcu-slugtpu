module gf180mcu_ocd_ip_sram__sram1024x8m8wm1(
    inout wire VDD,
    inout wire VSS,
    input wire CLK,
    input wire CEN,
    input wire GWEN,
    input wire [7:0] WEN,
    input wire [7:0] D,
    input wire [9:0] A,
    output wire [7:0] Q
);
    reg [9:0] mem [0:1023];
    reg [7:0] qo_reg;

    assign Q = qo_reg;

    integer i;
    initial begin
        for (i = 0; i < 1024; i = i + 1)
            mem[i] = 8'd0;
        qo_reg = 8'd0;
    end

    always @(posedge CLK) begin
        if (!CEN && !GWEN)
            mem[A] <= D;
        else if (!CEN && GWEN)
            qo_reg <= mem[A];
    end
endmodule
