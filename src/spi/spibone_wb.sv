// spibone_wb.sv
// Wishbone master driven by SPI slave (spibone protocol).
//
// Protocol (big-endian, CS held low for entire burst):
//   Write: 0x00 | ADDR[AdrW-1:0] | DATA[DataW-1:0] [DATA ...]
//   Read:  0x01 | ADDR[AdrW-1:0] | (clock dummy bytes) -> DATA[DataW-1:0] [...]
//
// Address is sent in ceil(AdrW/8) bytes MSB-first.
// Data is sent/received in ceil(DataW/8) bytes MSB-first.
// Burst: address auto-increments by DataW/8 bytes after each word.
//
// Sampling strategy:
//   SCK is registered once (sck_r). Edge detected as sck_r & ~prev_sck_r.
//   MOSI is registered once (mosi_r). Sampled at sck_rise using mosi_r
//   which holds the value from the previous clock — this is the stable MOSI
//   value that was set up before SCK rose, satisfying SPI Mode 0 setup time.
//   CS is registered twice for clean deassert detection.
//
//   clk_i must be >= 4x SCK for reliable edge detection.
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module spibone_wb #(
    parameter int AdrW  = 32,
    parameter int DataW = 32
) (
    input  logic        clk_i,
    input  logic        rst_i,

    // SPI pins
    input  logic        spi_sck_i,
    input  logic        spi_mosi_i,
    output logic        spi_miso_o,
    input  logic        spi_cs_n_i,

    // Wishbone master
    output logic [AdrW-1:0]    wb_adr_o,
    output logic [DataW-1:0]   wb_dat_o,
    input  logic [DataW-1:0]   wb_dat_i,
    output logic               wb_we_o,
    output logic               wb_stb_o,
    output logic               wb_cyc_o,
    output logic [DataW/8-1:0] wb_sel_o,
    input  logic               wb_ack_i
);

    localparam int AdrBytes  = (AdrW  + 7) / 8;
    localparam int DataBytes = (DataW + 7) / 8;
    // Internal shift registers are padded to a whole number of bytes
    localparam int AdrBits  = AdrBytes  * 8;
    localparam int DataBits = DataBytes * 8;

    // Byte counter widths
    localparam int AdrCntW  = $clog2(AdrBytes)  + 1;
    localparam int DataCntW = $clog2(DataBytes) + 1;

    // ------------------------------------------------------------------
    // Input registers
    // ------------------------------------------------------------------
    logic sck_r, prev_sck_r;
    logic mosi_r;
    logic cs_n_r, cs_n_rr;

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            sck_r      <= 1'b0;
            prev_sck_r <= 1'b0;
            mosi_r     <= 1'b0;
            cs_n_r     <= 1'b1;
            cs_n_rr    <= 1'b1;
        end else begin
            prev_sck_r <= sck_r;
            sck_r      <= spi_sck_i;
            mosi_r     <= spi_mosi_i;
            cs_n_r     <= spi_cs_n_i;
            cs_n_rr    <= cs_n_r;
        end
    end

    logic sck_rise, sck_fall, cs_active, cs_deassert;
    assign sck_rise   =  sck_r & ~prev_sck_r;
    assign sck_fall   = ~sck_r &  prev_sck_r;
    assign cs_active  = ~cs_n_rr;
    assign cs_deassert = cs_n_r & ~cs_n_rr;

    // ------------------------------------------------------------------
    // Receive shift register
    // ------------------------------------------------------------------
    logic [7:0] rx_shreg;
    logic [2:0] rx_bitcnt;
    logic       rx_byte_done;
    logic       rx_byte_boundary_fall;
    logic [7:0] rx_byte_data;

    assign rx_byte_done          = sck_rise & (rx_bitcnt == 3'd7);
    assign rx_byte_boundary_fall = sck_fall & (rx_bitcnt == 3'd0);
    assign rx_byte_data          = {rx_shreg[6:0], spi_mosi_i};

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i || !cs_active) begin
            rx_shreg  <= '0;
            rx_bitcnt <= '0;
        end else if (sck_rise) begin
            rx_shreg  <= {rx_shreg[6:0], spi_mosi_i};
            rx_bitcnt <= rx_bitcnt + 3'd1;
        end
    end

    // ------------------------------------------------------------------
    // Transmit shift register
    // ------------------------------------------------------------------
    logic [7:0] tx_shreg;
    logic       tx_load;
    logic [7:0] tx_load_val;

    assign spi_miso_o = tx_shreg[7];

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i || !cs_active)
            tx_shreg <= '0;
        else if (tx_load)
            tx_shreg <= tx_load_val;
        else if (sck_fall)
            tx_shreg <= {tx_shreg[6:0], 1'b0};
    end

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Combinational helpers for FSM
    // ------------------------------------------------------------------
    // Next shifted value when a new byte is received into addr_reg / data_reg
    wire [AdrBits-1:0]  addr_shifted = {addr_reg[AdrBits-9:0], rx_byte_data};
    wire [DataBits-1:0] data_shifted = {data_reg[DataBits-9:0], rx_byte_data};

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_CMD,
        S_ADDR,
        S_WB_READ,
        S_TX,
        S_TX_BURST,
        S_DATA,
        S_WB_WRITE
    } state_t;

    state_t                   state;
    logic                     is_read;
    logic [AdrBits-1:0]       addr_reg;   // accumulates address bytes
    logic [DataBits-1:0]      data_reg;   // accumulates write-data bytes
    logic [DataBits-1:0]      rd_data;    // holds WB read result
    logic [AdrCntW-1:0]       addr_cnt;
    logic [DataCntW-1:0]      data_cnt;   // used for both TX and DATA phases

    // sel is always full-word
    assign wb_sel_o = '1;

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            state       <= S_IDLE;
            is_read     <= '0;
            addr_reg    <= '0;
            data_reg    <= '0;
            rd_data     <= '0;
            addr_cnt    <= '0;
            data_cnt    <= '0;
            wb_adr_o    <= '0;
            wb_dat_o    <= '0;
            wb_we_o     <= '0;
            wb_stb_o    <= '0;
            wb_cyc_o    <= '0;
            tx_load     <= '0;
            tx_load_val <= '0;
        end else begin
            tx_load <= '0;

            // Deassert WB strobe on ack
            if (wb_ack_i) begin
                wb_stb_o <= '0;
                wb_cyc_o <= '0;
            end

            // CS deasserted — abort and return to idle
            if (cs_deassert || !cs_active) begin
                state    <= S_IDLE;
                wb_stb_o <= '0;
                wb_cyc_o <= '0;
            end else begin

                case (state)

                    S_IDLE: begin
                        if (cs_active)
                            state <= S_CMD;
                    end

                    // ---- Command byte: bit[0]=0 write, bit[0]=1 read ----
                    S_CMD: begin
                        if (rx_byte_done) begin
                            is_read  <= rx_byte_data[0];
                            addr_cnt <= '0;
                            state    <= S_ADDR;
                        end
                    end

                    // ---- Address bytes MSB first (AdrBytes total) ----
                    S_ADDR: if (rx_byte_done) begin
                        addr_reg <= addr_shifted;
                        addr_cnt <= addr_cnt + 1;
                        if (addr_cnt == AdrCntW'(AdrBytes - 1)) begin
                            if (is_read) begin
                                wb_adr_o <= addr_shifted[AdrW-1:0];
                                wb_we_o  <= '0;
                                wb_stb_o <= '1;
                                wb_cyc_o <= '1;
                                state    <= S_WB_READ;
                            end else begin
                                data_cnt <= '0;
                                state    <= S_DATA;
                            end
                        end
                    end

                    // ---- Wait for WB read ack ----
                    S_WB_READ: begin
                        if (wb_ack_i) begin
                            rd_data  <= wb_dat_i;
                            data_cnt <= '0;
                            state    <= S_TX;
                        end
                    end

                    // ---- Transmit DataBytes bytes MSB first ----
                    S_TX: if (rx_byte_boundary_fall) begin
                        tx_load     <= '1;
                        tx_load_val <= rd_data[(DataCntW'(DataBytes - 1) - data_cnt) * 8 +: 8];
                        data_cnt    <= data_cnt + 1;
                        if (data_cnt == DataCntW'(DataBytes - 1))
                            state <= S_TX_BURST;
                    end

                    // ---- Burst: increment addr and start next WB read ----
                    S_TX_BURST: if (rx_byte_boundary_fall) begin
                        addr_reg <= AdrBits'(addr_reg[AdrW-1:0] + AdrW'(DataBytes));
                        wb_adr_o <= addr_reg[AdrW-1:0] + AdrW'(DataBytes);
                        wb_we_o  <= '0;
                        wb_stb_o <= '1;
                        wb_cyc_o <= '1;
                        data_cnt <= '0;
                        state    <= S_WB_READ;
                    end

                    // ---- Receive write data bytes MSB first (DataBytes total) ----
                    S_DATA: if (rx_byte_done) begin
                        data_reg <= data_shifted;
                        data_cnt <= data_cnt + 1;
                        if (data_cnt == DataCntW'(DataBytes - 1)) begin
                            wb_dat_o <= data_shifted[DataW-1:0];
                            wb_adr_o <= addr_reg[AdrW-1:0];
                            wb_we_o  <= '1;
                            wb_stb_o <= '1;
                            wb_cyc_o <= '1;
                            state    <= S_WB_WRITE;
                        end
                    end

                    // ---- Wait for WB write ack, loop for burst ----
                    S_WB_WRITE: begin
                        if (wb_ack_i) begin
                            addr_reg <= AdrBits'(addr_reg[AdrW-1:0] + AdrW'(DataBytes));
                            data_cnt <= '0;
                            state    <= S_DATA;
                        end
                    end

                    default: state <= S_IDLE;

                endcase
            end
        end
    end

endmodule

`default_nettype wire
