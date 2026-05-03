<<<<<<< HEAD
// wb_to_axi4.sv
// Wishbone B3 classic -> AXI4 bridge
//
// Optimised for spibone burst writes:
//   - ACKs WB as soon as AW+W channels both accepted
//   - Drains BRESP independently, never stalls the burst
//   - Read path waits for RDATA before ACK (required — data must return)
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module wb_to_axi4 (
    input  logic        clk,
    input  logic        rst,

    // Wishbone slave (from wb_decoder port 0)
    input  logic [31:0] wb_adr_i,
    input  logic [31:0] wb_dat_i,
    output logic [31:0] wb_dat_o,
    input  logic        wb_we_i,
    input  logic        wb_stb_i,
    input  logic        wb_cyc_i,
    output logic        wb_ack_o,

    // AXI4 master -> sdram_axi inport_*

    // Write address channel
    output logic        axi_awvalid,
    input  logic        axi_awready,
    output logic [31:0] axi_awaddr,
    output logic [3:0]  axi_awid,
    output logic [7:0]  axi_awlen,
    output logic [1:0]  axi_awburst,

    // Write data channel
    output logic        axi_wvalid,
    input  logic        axi_wready,
    output logic [31:0] axi_wdata,
    output logic [3:0]  axi_wstrb,
    output logic        axi_wlast,

    // Write response channel
    input  logic        axi_bvalid,
    output logic        axi_bready,
    input  logic [1:0]  axi_bresp,

    // Read address channel
    output logic        axi_arvalid,
    input  logic        axi_arready,
    output logic [31:0] axi_araddr,
    output logic [3:0]  axi_arid,
    output logic [7:0]  axi_arlen,
    output logic [1:0]  axi_arburst,

    // Read data channel
    input  logic        axi_rvalid,
    output logic        axi_rready,
    input  logic [31:0] axi_rdata,
    input  logic        axi_rlast,
    input  logic [1:0]  axi_rresp
);

    // Fixed AXI signals — single beat per WB transaction
    assign axi_awid    = 4'h0;
    assign axi_awlen   = 8'h00;   // 1 beat
    assign axi_awburst = 2'b01;   // INCR
    assign axi_wstrb   = 4'hF;    // all byte lanes
    assign axi_wlast   = 1'b1;
    assign axi_arid    = 4'h0;
    assign axi_arlen   = 8'h00;
    assign axi_arburst = 2'b01;

    // Always accept write responses and read data
    assign axi_bready  = 1'b1;
    assign axi_rready  = 1'b1;

    // ------------------------------------------------------------------
    // Write path
    // ------------------------------------------------------------------
    logic aw_done, w_done;

    logic aw_accepted, w_accepted;
    assign aw_accepted = axi_awvalid && axi_awready;
    assign w_accepted  = axi_wvalid  && axi_wready;

    logic wr_staged;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            axi_awvalid <= '0;
            axi_wvalid  <= '0;
            axi_awaddr  <= '0;
            axi_wdata   <= '0;
            aw_done     <= '0;
            w_done      <= '0;
            wr_staged   <= '0;
        end else begin
            if (aw_accepted) begin axi_awvalid <= '0; aw_done <= '1; end
            if (w_accepted)  begin axi_wvalid  <= '0; w_done  <= '1; end

            if (aw_done && w_done) begin
                aw_done <= '0;
                w_done  <= '0;
            end

            // Cycle 1: latch data, set staged flag
            if (wb_cyc_i && wb_stb_i && wb_we_i && !axi_awvalid && !aw_done && !wr_staged) begin
                axi_awaddr <= wb_adr_i;
                axi_wdata  <= wb_dat_i;
                wr_staged  <= '1;
            end

            // Cycle 2: data is now settled, assert valids
            if (wr_staged) begin
                axi_awvalid <= '1;
                axi_wvalid  <= '1;
                wr_staged   <= '0;
            end
        end
    end

    // ACK fires as soon as both channels complete handshake
    logic wb_write_ack;
    assign wb_write_ack = (aw_accepted || aw_done) && (w_accepted || w_done);

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------
    logic rd_pending;
    logic rd_data_ready;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            axi_arvalid <= '0;
            axi_araddr  <= '0;
            wb_dat_o    <= '0;
            rd_pending  <= '0;
            rd_data_ready <= '0;
        end else begin
            if (axi_arvalid && axi_arready) begin
                axi_arvalid <= '0;
                rd_pending  <= '1;
            end

            if (axi_rvalid) begin
                wb_dat_o   <= axi_rdata;
                rd_pending <= '0;
                rd_data_ready <= '1;
            end else if (rd_data_ready) begin
                rd_data_ready <= '0;
            end

            if (wb_cyc_i && wb_stb_i && !wb_we_i && !axi_arvalid && !rd_pending) begin
                axi_arvalid <= '1;
                axi_araddr  <= wb_adr_i;
            end
        end
    end

    logic wb_read_ack;
    assign wb_read_ack = rd_data_ready;

    // ------------------------------------------------------------------
    // Combined WB ACK
    // ------------------------------------------------------------------
    assign wb_ack_o = wb_we_i ? wb_write_ack : wb_read_ack;

endmodule

`default_nettype wire
