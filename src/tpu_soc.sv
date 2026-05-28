module tpu_soc #(
    parameter sys_clk_mhz_p   = 67
   ,parameter sdr_data_bits_p = 16
   ,parameter sdr_addr_bits_p = 13
   ,parameter dram_burst_p    = 4
   ,parameter dbg_n_words_p = 32
   ,parameter test_ram_words_p = 32

   ,parameter control_top_instruction_addr_width_p = 12
   ,parameter control_top_dram_counter_width_p = 8
) (
`ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
`endif
    input clk_i
   ,input rst_i

   // SPI slave interface (spibone — direct SDRAM + TPU register access)
   ,input  spi_clk_i
   ,input  spi_cs_ni
   ,input  spi_mosi_i
   ,output spi_miso_o

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
   ,output logic test_mode_o
);
   localparam _rows_lp = 8192;
   localparam _cols_lp = 512;
   localparam _banks_lp = 4;
   localparam dma_data_w_lp  = sdr_data_bits_p * dram_burst_p;
   localparam dma_addr_w_lp  = $clog2(_rows_lp) + $clog2(_banks_lp) + $clog2(_cols_lp) - $clog2(dram_burst_p);

   // -----------------------------------------------------------------------
   // WishBone bus wires
   // -----------------------------------------------------------------------

   // spibone → wb_decoder
   wire [31:0]               dec_adr;
   wire [dma_data_w_lp-1:0] dec_dat_w;
   wire [dma_data_w_lp-1:0] dec_dat_r;
   wire                      dec_we, dec_stb, dec_cyc, dec_ack;

   // wb_decoder port 0 → wb_mux m0 (SPI bridge path to DRAM)
   wire [dma_addr_w_lp - 1:0]   spi_wb_adr;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_r;
   wire                          spi_wb_we;
   wire                          spi_wb_stb;
   wire                          spi_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] spi_wb_sel;
   wire                          spi_wb_ack;
   assign spi_wb_sel = '1; // decoder has no sel output; always full-word

   // wb_decoder port 1 → tpu_regs (64-bit bus; dat narrowed to 32-bit at tpu_regs ports)
   wire [31:0]               tpureg_adr;
   wire [dma_data_w_lp-1:0] tpureg_wb_dat_w;
   wire [31:0]               tpureg_dat_r;
   wire [1:0]                tpureg_status;
   wire [31:0]               tpureg_pc_addr;
   wire                      tpureg_we, tpureg_stb, tpureg_cyc, tpureg_ack, tpureg_pc_stb;

   wire tpu_active;
   // TPU active is determined by state of compute controller
   assign tpu_active = tpureg_status[1];

   // DMA master → wb_mux m1
   wire [dma_addr_w_lp - 1:0]   dma_wb_adr;
   wire [dma_data_w_lp - 1:0]   dma_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   dma_wb_dat_r;
   wire                          dma_wb_we;
   wire                          dma_wb_stb;
   wire                          dma_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] dma_wb_sel;
   wire                          dma_wb_ack;

   // wb_mux output → wb_demux input (arbiter output bus)
   wire [dma_addr_w_lp - 1:0]   arb_wb_adr;
   wire [dma_data_w_lp - 1:0]   arb_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   arb_wb_dat_r;
   wire                          arb_wb_we;
   wire                          arb_wb_stb;
   wire                          arb_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] arb_wb_sel;
   wire                          arb_wb_ack;

   // wb_demux s0 → SDRAM controller
   wire [dma_addr_w_lp - 1:0]   sdram_wb_adr;
   wire [dma_data_w_lp - 1:0]   sdram_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   sdram_wb_dat_r;
   wire                          sdram_wb_we;
   wire                          sdram_wb_stb;
   wire                          sdram_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] sdram_wb_sel;
   wire                          sdram_wb_ack;

   // wb_demux s1 → test RAM
   wire [dma_addr_w_lp - 1:0]   tram_s_adr;
   wire [dma_data_w_lp - 1:0]   tram_s_dat_w;
   wire [dma_data_w_lp - 1:0]   tram_s_dat_r;
   wire                          tram_s_we;
   wire                          tram_s_stb;
   wire                          tram_s_cyc;
   wire [dma_data_w_lp/8 - 1:0] tram_s_sel;
   wire                          tram_s_ack;

   // CTRL[1] from tpu_regs
   wire test_mode;
   assign test_mode_o = test_mode;

   // -----------------------------------------------------------------------
   // DMA control
   // -----------------------------------------------------------------------
   wire [dma_addr_w_lp - 1:0]  dma_start_addr;
   wire                        dma_start;
   wire [control_top_dram_counter_width_p - 1:0] dma_word_count;
   wire                        dma_we;
   wire                        dma_busy;
   wire                        dma_done;

   // -----------------------------------------------------------------------
   // DMA stream
   // -----------------------------------------------------------------------
   wire [dma_data_w_lp - 1:0] dma_rd_data;
   wire                        dma_rd_valid;
   wire                        dma_rd_ready;
   wire [dma_data_w_lp - 1:0] dma_wr_data;
   wire                        dma_wr_valid;
   wire                        dma_wr_ready;

  //debug observation byte map
  // driven into tpu_regs and read by host via REG_DBG_ADDR/DBG_DATA over SPI
  logic [dbg_n_words_p*8 - 1:0] dbg_word;
  logic internal_error;

   // -----------------------------------------------------------------------
   // spibone_wb: SPI slave → Wishbone master
   // -----------------------------------------------------------------------
   spibone_wb #(
       .addr_w_p  (32),
       .data_w_p  (dma_data_w_lp)
   ) spi_wb_inst (
       .clk_i       (clk_i),
       .rst_i       (rst_i),
       .spi_sck_i   (spi_clk_i),
       .spi_mosi_i  (spi_mosi_i),
       .spi_miso_o  (spi_miso_o),
       .spi_cs_n_i  (spi_cs_ni),
       .wb_adr_o    (dec_adr),
       .wb_dat_o    (dec_dat_w),
       .wb_dat_i    (dec_dat_r),
       .wb_we_o     (dec_we),
       .wb_stb_o    (dec_stb),
       .wb_cyc_o    (dec_cyc),
       .wb_sel_o    (/* full-word, unused here */),
       .wb_ack_i    (dec_ack)
   );

   // -----------------------------------------------------------------------
   // wb_decoder: route spibone transactions to DRAM or TPU registers
   // -----------------------------------------------------------------------
   wb_decoder #(
       .DataW (dma_data_w_lp)
   ) spi_wb_decoder (
       .wbs_adr_i  (dec_adr),
       .wbs_dat_i  (dec_dat_w),
       .wbs_dat_o  (dec_dat_r),
       .wbs_we_i   (dec_we),
       .wbs_stb_i  (dec_stb),
       .wbs_cyc_i  (dec_cyc),
       .wbs_ack_o  (dec_ack),
       // port 0: DRAM (byte addr + 64-bit data)
       .wbm0_adr_o (spi_wb_adr),
       .wbm0_dat_o (spi_wb_dat_w),
       .wbm0_dat_i (spi_wb_dat_r),
       .wbm0_we_o  (spi_wb_we),
       .wbm0_stb_o (spi_wb_stb),
       .wbm0_cyc_o (spi_wb_cyc),
       .wbm0_ack_i (spi_wb_ack),
       // port 1: TPU registers (32-bit, zero-extended to DataW)
       .wbm1_adr_o (tpureg_adr),
       .wbm1_dat_o (tpureg_wb_dat_w),
       .wbm1_dat_i ({{(dma_data_w_lp-32){1'b0}}, tpureg_dat_r}),
       .wbm1_we_o  (tpureg_we),
       .wbm1_stb_o (tpureg_stb),
       .wbm1_cyc_o (tpureg_cyc),
       .wbm1_ack_i (tpureg_ack)
   );

   // -----------------------------------------------------------------------
   // tpu_regs: TPU control/status registers via spibone
   // -----------------------------------------------------------------------
   tpu_regs #(
        .DBG_N_WORDS (dbg_n_words_p)
   ) regs (
       .clk_i             (clk_i),
       .rst_i             (rst_i),
       .wb_adr_i          (tpureg_adr),
       .wb_dat_i          (tpureg_wb_dat_w[31:0]),
       .wb_dat_o          (tpureg_dat_r),
       .wb_we_i           (tpureg_we),
       .wb_stb_i          (tpureg_stb),
       .wb_cyc_i          (tpureg_cyc),
       .wb_ack_o          (tpureg_ack),
       .test_mode_o       (test_mode),
       .tpu_pc_addr_o     (tpureg_pc_addr),
       .tpu_pc_stb_o      (tpureg_pc_stb),
       .tpu_state_i       (tpureg_status),
       .dbg_word_i        (dbg_word)
   );

   // -----------------------------------------------------------------------
   // wb_dma_master: TPU's DMA master (mux m1)
   // -----------------------------------------------------------------------
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
       .wb_adr_o     (dma_wb_adr),
       .wb_dat_o     (dma_wb_dat_w),
       .wb_dat_i     (dma_wb_dat_r),
       .wb_we_o      (dma_wb_we),
       .wb_stb_o     (dma_wb_stb),
       .wb_cyc_o     (dma_wb_cyc),
       .wb_sel_o     (dma_wb_sel),
       .wb_ack_i     (dma_wb_ack)
   );

   // -----------------------------------------------------------------------
   // wb_mux_2to1: arbitrate between SPI bridge (m0) and DMA master (m1)
   // -----------------------------------------------------------------------
   wb_mux_2to1 #(
       .AdrW  (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) wb_mux (
       .clk_i        (clk_i),
       .rst_i        (rst_i),
       .tpu_active_i (tpu_active),
       // m0: SPI bridge
       .m0_adr_i     (spi_wb_adr),
       .m0_dat_w_i   (spi_wb_dat_w),
       .m0_dat_r_o   (spi_wb_dat_r),
       .m0_we_i      (spi_wb_we),
       .m0_stb_i     (spi_wb_stb),
       .m0_cyc_i     (spi_wb_cyc),
       .m0_sel_i     (spi_wb_sel),
       .m0_ack_o     (spi_wb_ack),
       // m1: DMA master
       .m1_adr_i     (dma_wb_adr),
       .m1_dat_w_i   (dma_wb_dat_w),
       .m1_dat_r_o   (dma_wb_dat_r),
       .m1_we_i      (dma_wb_we),
       .m1_stb_i     (dma_wb_stb),
       .m1_cyc_i     (dma_wb_cyc),
       .m1_sel_i     (dma_wb_sel),
       .m1_ack_o     (dma_wb_ack),
       // slave: arbiter output
       .s_adr_o      (arb_wb_adr),
       .s_dat_w_o    (arb_wb_dat_w),
       .s_dat_r_i    (arb_wb_dat_r),
       .s_we_o       (arb_wb_we),
       .s_stb_o      (arb_wb_stb),
       .s_cyc_o      (arb_wb_cyc),
       .s_sel_o      (arb_wb_sel),
       .s_ack_i      (arb_wb_ack)
   );

   // -----------------------------------------------------------------------
   // wb_demux_1to2: route mux output to either the SDRAM controller (s0) or the on-chip test RAM (s1), selected by tpu_regs.CTRL[1] = test_mode.
   // -----------------------------------------------------------------------
   wb_demux_1to2 #(
       .AdrW  (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) test_demux (
       .sel_i      (test_mode),
       // master: output of wb_mux_2to1
       .m_adr_i    (arb_wb_adr),
       .m_dat_w_i  (arb_wb_dat_w),
       .m_dat_r_o  (arb_wb_dat_r),
       .m_sel_i    (arb_wb_sel),
       .m_we_i     (arb_wb_we),
       .m_stb_i    (arb_wb_stb),
       .m_cyc_i    (arb_wb_cyc),
       .m_ack_o    (arb_wb_ack),
       // s0: SDRAM controller
       .s0_adr_o   (sdram_wb_adr),
       .s0_dat_w_o (sdram_wb_dat_w),
       .s0_dat_r_i (sdram_wb_dat_r),
       .s0_sel_o   (sdram_wb_sel),
       .s0_we_o    (sdram_wb_we),
       .s0_stb_o   (sdram_wb_stb),
       .s0_cyc_o   (sdram_wb_cyc),
       .s0_ack_i   (sdram_wb_ack),
       // s1: test RAM
       .s1_adr_o   (tram_s_adr),
       .s1_dat_w_o (tram_s_dat_w),
       .s1_dat_r_i (tram_s_dat_r),
       .s1_sel_o   (tram_s_sel),
       .s1_we_o    (tram_s_we),
       .s1_stb_o   (tram_s_stb),
       .s1_cyc_o   (tram_s_cyc),
       .s1_ack_i   (tram_s_ack)
   );


   // -----------------------------------------------------------------------
   // SDRAM controller
   // -----------------------------------------------------------------------
   wb_sdr_mt48lc16m16a_7e #(
       .sys_clk_mhz_p   (sys_clk_mhz_p),
       .dram_burst_p     (dram_burst_p)
   ) sdram_ctrl (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       // WB slave
       .m_adr_i  (sdram_wb_adr),
       .m_dat_i  (sdram_wb_dat_w),
       .m_we_i   (sdram_wb_we),
       .m_stb_i  (sdram_wb_stb),
       .m_cyc_i  (sdram_wb_cyc),
       .m_sel_i  (sdram_wb_sel),
       .m_ack_o  (sdram_wb_ack),
       .m_dat_o  (sdram_wb_dat_r),
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

   // -----------------------------------------------------------------------
   // wb_test_ram: small on-chip stand-in for SDRAM
   // -----------------------------------------------------------------------
   wb_test_ram #(
       .N_WORDS (test_ram_words_p),
       .AdrW    (dma_addr_w_lp),
       .DataW   (dma_data_w_lp)
   ) test_ram (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       .wb_adr_i (tram_s_adr),
       .wb_dat_i (tram_s_dat_w),
       .wb_dat_o (tram_s_dat_r),
       .wb_sel_i (tram_s_sel),
       .wb_we_i  (tram_s_we),
       .wb_stb_i (tram_s_stb),
       .wb_cyc_i (tram_s_cyc),
       .wb_ack_o (tram_s_ack)
   );


  control_top #(
     .EXTERNAL_DRAM_ADDR_WIDTH(dma_addr_w_lp),
     .DRAM_DATA_WIDTH(dma_data_w_lp),
     .DRAM_ADDR_WIDTH(control_top_instruction_addr_width_p),
     .DRAM_COUNTER_WIDTH(control_top_dram_counter_width_p)
  )
  control (
`ifdef USE_POWER_PINS
      .VDD(VDD),
      .VSS(VSS),
`endif
      .clk_i    (clk_i),
      .rst_i    (rst_i),

      .dram_start_addr_o(dma_start_addr),
      .dram_word_count_o(dma_word_count),
      .dram_we_o(dma_we),
      .dram_start_o(dma_start),
      .dma_busy_i(dma_busy),
      .dma_done_i(dma_done),

      .dram2sram_valid_i(dma_rd_valid),
      .dram2sram_data_i(dma_rd_data),
      .dram2sram_ready_o(dma_rd_ready),

      .sram2dram_valid_o(dma_wr_valid),
      .sram2dram_data_o(dma_wr_data),
      .sram2dram_ready_i(dma_wr_ready),

      .pc_in(tpureg_pc_addr[control_top_instruction_addr_width_p - 1:0]),
      .pc_valid_i(tpureg_pc_stb),
      .pc_ready_o(),

      .tpu_state_o(tpureg_status),
      .INTERNAL_ERROR_O(internal_error)
  );

  always_comb begin
        dbg_word = '0;
        dbg_word[ 8'h00*8 +: 8] = {3'b0, internal_error, dma_done, dma_busy,
                                     tpureg_status};
        dbg_word[ 8'h01*8 +: 8] = {4'b0, dma_start, dma_we,
                                     dma_rd_valid, dma_wr_valid};
        dbg_word[ 8'h02*8 +: 8] = {7'b0, tpureg_pc_stb};
        dbg_word[ 8'h03*8 +: 8] = {7'b0, test_mode};

        dbg_word[ 8'h10*8 +: 8] = tpureg_pc_addr[ 7: 0];
        dbg_word[ 8'h11*8 +: 8] = tpureg_pc_addr[15: 8];
        dbg_word[ 8'h12*8 +: 8] = tpureg_pc_addr[23:16];
        dbg_word[ 8'h13*8 +: 8] = tpureg_pc_addr[31:24];

        dbg_word[ 8'h14*8 +: 8] = dma_word_count[7:0];

        dbg_word[ 8'h1C*8 +: 8] = 8'h53; // 'S'
        dbg_word[ 8'h1D*8 +: 8] = 8'h4C; // 'L'
        dbg_word[ 8'h1E*8 +: 8] = 8'h55; // 'U'
        dbg_word[ 8'h1F*8 +: 8] = 8'h47; // 'G'
    end

endmodule
