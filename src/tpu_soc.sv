module tpu_soc #(
    parameter sys_clk_mhz_p   = 67
   ,parameter sdr_data_bits_p = 16
   ,parameter sdr_addr_bits_p = 13
   ,parameter dram_burst_p    = 4
) (
    input clk_i
   ,input rst_i

   // SPI slave interface
   ,input  spi_clk_i
   ,input  spi_cs_ni
   ,input  spi_mosi_i
   ,output spi_miso_o
   ,output spi_oe_o

   // SDRAM
   ,input  [sdr_data_bits_p - 1 :0] sdr_dq_i
   ,output logic [sdr_data_bits_p - 1:0] sdr_dq_o
   ,output logic sdr_dq_oe_o
   ,output logic [sdr_addr_bits_p - 1:0] sdr_addr_o
   ,output logic [1:0] sdr_ba_o
   ,output logic sdr_cke_o
   ,output logic sdr_cs_no
   ,output logic sdr_ras_no
   ,output logic sdr_cas_no
   ,output logic sdr_we_no
   ,output logic [1:0] sdr_dqm_o
);
   localparam _rows_lp = 8192;
   localparam _cols_lp = 512;
   localparam _banks_lp = 4;
   localparam dma_data_w_lp  = sdr_data_bits_p * dram_burst_p;
   localparam dma_addr_w_lp  = $clog2(_rows_lp) + $clog2(_banks_lp) + $clog2(_cols_lp) - $clog2(dram_burst_p);

   // -----------------------------------------------------------------------
   // WishBone bus wires (DMA master → SDRAM controller)
   // -----------------------------------------------------------------------
   wire [dma_addr_w_lp - 1:0]   wb_adr;
   wire [dma_data_w_lp - 1:0]   wb_dat_m2s;
   wire [dma_data_w_lp - 1:0]   wb_dat_s2m;
   wire                          wb_we;
   wire                          wb_stb;
   wire                          wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] wb_sel;
   wire                          wb_ack;

   // -----------------------------------------------------------------------
   // DMA control (TODO: connect to instruction decoder)
   // -----------------------------------------------------------------------
   wire [dma_addr_w_lp - 1:0] dma_start_addr;
   wire [15:0]                 dma_word_count;
   wire                        dma_we;
   wire                        dma_start;
   wire                        dma_busy;
   wire                        dma_done;


   // -----------------------------------------------------------------------
   // DMA stream (TODO: connect to systolic-array SRAM)
   // -----------------------------------------------------------------------
   wire [dma_data_w_lp - 1:0] dma_rd_data;
   wire                        dma_rd_valid;
   wire                        dma_rd_ready;
   wire [dma_data_w_lp - 1:0] dma_wr_data;
   wire                        dma_wr_valid;
   wire                        dma_wr_ready;

   // -----------------------------------------------------------------------
   // SPI register bus (TODO: connect to instruction decoder)
   // -----------------------------------------------------------------------
   wire [15:0] spi_oaddr;
   wire [7:0]  spi_odata;
   wire [7:0]  spi_idata;
   wire        spi_rdstb;
   wire        spi_wrstb;

   assign spi_idata = '0;

   housekeeping_spi spi (
      .reset               (rst_i),
      .SCK                 (spi_clk_i),
      .SDI                 (spi_mosi_i),
      .SDO                 (spi_miso_o),
      .CSB                 (spi_cs_ni),
      .sdoenb              (spi_oe_o),
      .idata               (spi_idata),
      .odata               (spi_odata),
      .oaddr               (spi_oaddr),
      .rdstb               (spi_rdstb),
      .wrstb               (spi_wrstb),
      .pass_thru_mgmt      (),
      .pass_thru_mgmt_delay(),
      .pass_thru_user      (),
      .pass_thru_user_delay(),
      .pass_thru_mgmt_reset(),
      .pass_thru_user_reset()
   );
   // TODO:
   instruction_decoder ();

   wb_dma_master #(
       .AddrW (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) dma (
       .clk_i        (clk_i),
       .rst_i        (rst_i),
       // control
       .start_addr_i (dma_start_addr),
       .word_count_i (dma_word_count),
       .we_i         (dma_we),
       .start_i      (dma_start),
       .busy_o       (dma_busy),
       .done_o       (dma_done),
       // stream
       .rd_data_o    (dma_rd_data),
       .rd_valid_o   (dma_rd_valid),
       .rd_ready_i   (dma_rd_ready),
       .wr_data_i    (dma_wr_data),
       .wr_valid_i   (dma_wr_valid),
       .wr_ready_o   (dma_wr_ready),
       // WB master
       .wb_adr_o     (wb_adr),
       .wb_dat_o     (wb_dat_m2s),
       .wb_dat_i     (wb_dat_s2m),
       .wb_we_o      (wb_we),
       .wb_stb_o     (wb_stb),
       .wb_cyc_o     (wb_cyc),
       .wb_sel_o     (wb_sel),
       .wb_ack_i     (wb_ack)
   );

   wb_mux_2to1 #() ();

   wb_sdr_mt48lc16m16a_7e #(
       .sys_clk_mhz_p   (sys_clk_mhz_p),
       .dram_burst_p     (dram_burst_p)
   ) sdram_ctrl (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       // WB slave
       .m_adr_i  (wb_adr),
       .m_dat_i  (wb_dat_m2s),
       .m_we_i   (wb_we),
       .m_stb_i  (wb_stb),
       .m_cyc_i  (wb_cyc),
       .m_sel_i  (wb_sel),
       .m_ack_o  (wb_ack),
       .m_dat_o  (wb_dat_s2m),
       // SDRAM pins
       .s_dq_i   (sdr_dq_i),
       .s_dq_o   (sdr_dq_o),
       .s_addr_o (sdr_addr_o),
       .s_ba_o   (sdr_ba_o),
       .s_cke_o  (sdr_cke_o),
       .s_cs_no  (sdr_cs_no),
       .s_ras_no (sdr_ras_no),
       .s_cas_no (sdr_cas_no),
       .s_we_no  (sdr_we_no),
       .s_dqm_o  (sdr_dqm_o),
       .oe_o     (sdr_dq_oe_o)
   );

   // wire up with systolic array
endmodule
