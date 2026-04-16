//* SPI Mode 0 Slave (CPOL=0, CPHA=0)
//* - Shift in on SCLK rising edge while CS_n is low
//* - Shift out MSB first on MISO
//*
//* FIX: rx_valid remains asserted after byte completion until the
//* first SCLK rising edge of the next byte. This guarantees the host
//* or testbench can observe the completed byte despite synchronizer
//* latency between SPI pins and clk domain.

module spi_slave (
    input  wire clk,
    input  wire rst,

    input  wire sclk,
    input  wire mosi,
    input  wire cs_n,
    output wire miso,

    output logic [7:0] rx_data,
    output logic       rx_valid,

    input  wire [7:0] tx_data,
    input  wire       tx_valid
);

    //* Synchronize asynchronous SPI inputs into clk domain
    logic sclk_q1, sclk_q2;
    logic cs_q1,   cs_q2;
    logic mosi_q1, mosi_q2;

    always_ff @(posedge clk) begin
        if (rst) begin
            sclk_q1 <= 1'b0;  sclk_q2 <= 1'b0;
            cs_q1   <= 1'b1;  cs_q2   <= 1'b1;
            mosi_q1 <= 1'b0;  mosi_q2 <= 1'b0;
        end else begin
            sclk_q1 <= sclk;    sclk_q2 <= sclk_q1;
            cs_q1   <= cs_n;    cs_q2   <= cs_q1;
            mosi_q1 <= mosi;    mosi_q2 <= mosi_q1;
        end
    end

    wire sclk_sync = sclk_q2;
    wire cs_n_sync = cs_q2;
    wire mosi_sync = mosi_q2;

    //* Detect rising edge of synchronized SCLK
    logic sclk_sync_d;
    always_ff @(posedge clk) begin
        if (rst) sclk_sync_d <= 1'b0;
        else     sclk_sync_d <= sclk_sync;
    end

    wire sclk_rise = (sclk_sync == 1'b1) && (sclk_sync_d == 1'b0);

    //* Shift registers and bit counter
    logic [7:0] shift_rx;
    logic [7:0] shift_tx;
    logic [2:0] bit_cnt;

    //* MSB-first transmit
    assign miso = shift_tx[7];

    always_ff @(posedge clk) begin
        if (rst) begin
            bit_cnt  <= 3'd0;
            shift_rx <= 8'd0;
            shift_tx <= 8'd0;
            rx_data  <= 8'd0;
            rx_valid <= 1'b0;

        end else begin

            if (cs_n_sync) begin
                //* Idle state (CS high)
                bit_cnt <= 3'd0;

                if (tx_valid)
                    shift_tx <= tx_data;

            end else begin
                //* Active SPI transaction
                if (sclk_rise) begin

                    //* Clear valid flag at start of next byte
                    if (bit_cnt == 3'd0)
                        rx_valid <= 1'b0;

                    //* Sample MOSI on rising edge (Mode 0)
                    shift_rx <= {shift_rx[6:0], mosi_sync};

                    //* Shift transmit register
                    shift_tx <= {shift_tx[6:0], 1'b0};

                    if (bit_cnt == 3'd7) begin
                        //* Byte complete
                        rx_data  <= {shift_rx[6:0], mosi_sync};
                        rx_valid <= 1'b1;
                        bit_cnt  <= 3'd0;

                        if (tx_valid)
                            shift_tx <= tx_data;

                    end else begin
                        bit_cnt <= bit_cnt + 3'd1;
                    end
                end
            end
        end
    end

endmodule
