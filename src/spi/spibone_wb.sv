// COMMAND:
// 0x00: READ_SINGLE
// 0x01: WRITE_SINGLE

module spibone_wb #(
    parameter int addr_w_p  = 32,
    parameter int data_w_p = 64
) (
    input logic                   clk_i,
    input logic                   rst_i,

    // SPI pins
    input logic                   spi_sck_i,
    input logic                   spi_mosi_i,
    output logic                  spi_miso_o,
    input logic                   spi_cs_n_i,

    // Wishbone master
    output logic [addr_w_p-1:0]   wb_adr_o,
    output logic [data_w_p-1:0]   wb_dat_o,
    input logic [data_w_p-1:0]    wb_dat_i,
    output logic                  wb_we_o,
    output logic                  wb_stb_o,
    output logic                  wb_cyc_o,
    output logic [data_w_p/8-1:0] wb_sel_o,
    input logic                   wb_ack_i
);
    typedef enum logic [3:0] {
        S_IDLE,
        S_CMD,
        S_ADDR,
        S_WR_PAYLOAD,
        S_WB_READ,
        S_WB_WRITE,
        S_READ_TX, // send back data at WB address to master
        S_ACK
    } state_t;

   typedef enum logic [7:0] {
      READ_SINGLE = 8'h00,
      WRITE_SINGLE = 8'h10
   } cmd_t;

   localparam int addr_bytes_lp = addr_w_p / 8;
   localparam int data_bytes_lp = data_w_p / 8;

   localparam int addr_cnt_w_lp  = $clog2(addr_bytes_lp)  + 1;
   localparam int data_cnt_w_lp  = $clog2(data_bytes_lp) + 1;

   localparam logic [7:0] byte_stall_lp = 8'hFF;

   logic [addr_cnt_w_lp - 1 : 0] addr_cntr_d, addr_cntr_q;
   logic [data_cnt_w_lp - 1 : 0] data_cntr_d, data_cntr_q;

   cmd_t cmd_reg_d, cmd_reg_q;
   logic [data_w_p - 1:0] data_reg_d, data_reg_q;
   logic [addr_w_p - 1:0] addr_reg_d, addr_reg_q;
   logic                  wb_ack_reg_d, wb_ack_reg_q;

   state_t state_q, state_d;
   wire active;
   wire  [7:0] byte_rx;
   logic [7:0] byte_tx;
   wire       byte_stb;

   spi_slave slave_inst (
      .clk_i(clk_i),
      .rst_i(rst_i),

      .spi_sck_async_i(spi_sck_i),
      .spi_cs_async_ni(spi_cs_n_i),
      .spi_mosi_async_i(spi_mosi_i),
      .spi_miso_o(spi_miso_o),

      .active_o(active),

      .byte_rx_o(byte_rx),
      .byte_tx_i(byte_tx),
      .byte_stb_o(byte_stb)
   );

   // always_ff @(posedge clk_i) begin
   //    if (rst_i || state_q != S_ADDR) begin
   //       addr_cntr_q <= '0;
   //    end else begin
   //       addr_cntr_q <= addr_cntr_d + 1;
   //    end
   // end

   // always_ff @(posedge clk_i) begin
   //    if (rst_i || state_q != S_WR_PAYLOAD) begin
   //       data_cntr_q <= '0;
   //    end else begin
   //       data_cntr_q <= data_cntr_d + 1;
   //    end
   // end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         addr_reg_q <= '0;
         data_reg_q <= '0;
         wb_ack_reg_q <= '0;
         addr_cntr_q <= '0;
         data_cntr_q <= '0;
         cmd_reg_q <= '0;
      end else begin
         addr_reg_q <= addr_reg_d;
         data_reg_q <= data_reg_d;
         wb_ack_reg_q <= wb_ack_reg_d;
         addr_cntr_q <= addr_cntr_d;
         data_cntr_q <= data_cntr_d;
         cmd_reg_q <= cmd_reg_d;
      end
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         state_q <= '0;
      end else begin
         state_q <= state_d;
      end
   end

   always_comb begin
      state_d = state_q;
      addr_reg_d = addr_reg_q;
      data_reg_d = data_reg_q;
      cmd_reg_d = cmd_reg_q;
      wb_ack_reg_d = wb_ack_reg_q;

      wb_adr_o = '0;
      wb_dat_o = '0;
      wb_we_o = '0;
      wb_stb_o = '0;
      wb_cyc_o = '0;
      wb_sel_o = '0;

      addr_cntr_d = addr_cntr_q;
      data_cntr_d = data_cntr_q;

      case (state_q)
      S_IDLE: begin
         byte_tx = '0;

         addr_reg_d = '0;
         data_reg_d = '0;
         cmd_reg_d = '0;
         wb_ack_reg_d = '0;

         if (active) begin
            state_d = S_CMD;
         end
      end
      S_CMD: begin
         byte_tx = '0;

         if (!active) begin
            state_d = S_IDLE;
         end else if (byte_stb) begin
            cmd_reg_d = byte_rx;
            if (byte_rx == READ_SINGLE || byte_rx == WRITE_SINGLE) begin
               state_d = S_ADDR;
            end
         end
      end
      S_ADDR: begin
         byte_tx = '0;

         if (!active) begin
            state_d = S_IDLE;
         end else if (byte_stb) begin
            addr_reg_d = { addr_reg_q[addr_w_p-9:0], byte_rx };
            if (addr_cntr_q == addr_bytes_lp - 1) begin
               addr_cntr_d = '0;
               if (cmd_reg_q == READ_SINGLE) begin
                  state_d = S_WB_READ;
               end else begin
                  state_d = S_WR_PAYLOAD;
               end
            end else begin
               addr_cntr_d = addr_cntr_q + 1;
            end
         end
      end
      S_WR_PAYLOAD: begin
         byte_tx = '0;

         if (!active) begin
            state_d = S_IDLE;
         end else if (byte_stb) begin
            data_reg_d = { data_reg_q[data_w_p-9:0], byte_rx };
            if (data_cntr_q == data_bytes_lp - 1) begin
               data_cntr_d = '0;
               state_d = S_WB_WRITE;
            end else begin
               data_cntr_d = data_cntr_q + 1;
            end
         end
      end
      S_WB_WRITE: begin
         byte_tx = byte_stall_lp;
         wb_sel_o = '1;

         if (!wb_ack_reg_q && wb_ack_i) begin
            wb_ack_reg_d = 1'b1;
         end else begin
            wb_we_o  = 1'b1;
            wb_stb_o = 1'b1;
            wb_cyc_o = 1'b1;
            wb_adr_o = addr_reg_q;
            wb_dat_o = data_reg_q;
         end

         if (!active) begin
            wb_ack_reg_d = '0;
            state_d = S_IDLE;
         end else if (byte_stb) begin
            if (wb_ack_reg_q) begin
               wb_ack_reg_d = '0;
               state_d = S_ACK;
            end
         end
      end
      S_WB_READ: begin
         byte_tx = byte_stall_lp;
         wb_sel_o = '1;

         if (!wb_ack_reg_q && wb_ack_i) begin
            data_reg_d = wb_dat_i;
            wb_ack_reg_d = 1'b1;
         end else begin
            wb_we_o  = '0;
            wb_stb_o = 1'b1;
            wb_cyc_o = 1'b1;
            wb_adr_o = addr_reg_q;
         end

         if (!active) begin
            wb_ack_reg_d = '0;
            state_d = S_IDLE;
         end else if (byte_stb) begin
            if (wb_ack_reg_q) begin
               wb_ack_reg_d = '0;
               state_d = S_ACK;
            end
         end
      end
      S_ACK: begin
         byte_tx = 8'hAC;

         if (!active) begin
            state_d = S_IDLE;
         end else if (byte_stb) begin
            if (cmd_reg_q == READ_SINGLE) begin
               state_d = S_READ_TX;
            end else begin
               state_d = S_CMD;
            end
         end
      end
      S_READ_TX: begin
         byte_tx = data_reg_q[data_w_p - 1:data_w_p - 8];

         if (!active) begin
            state_d = S_IDLE;
         end else if (byte_stb) begin
            data_reg_d = {data_reg_q[data_w_p - 9:0], 8'h00};

            if (data_cntr_q == data_bytes_lp - 1) begin
               data_cntr_d = '0;
               data_reg_d = '0;
               addr_reg_d = '0;
               cmd_reg_d = '0;
               state_d = S_CMD;
            end else begin
               data_cntr_d = data_cntr_q + 1;
            end
         end
      end
      default: begin
         state_d = S_IDLE;
      end
      endcase
   end

endmodule
