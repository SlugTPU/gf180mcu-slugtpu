// SDRAM controller for mt48lc16m16a, with speed grade of 7e
module wb_sdr_mt48lc16m16a_7e #(
    parameter sys_clk_mhz_p = 67
   ,parameter burst_p = 4
   // below should never be modified
   ,parameter _data_bits_p = 16
   ,parameter _addr_bits_p = 13
   ,parameter _banks_p = 4
) (
    input clk_i
   ,input rst_i

   ,input [_addr_bits_p - 1:0]  m_adr_i
   ,input [_data_bits_p * burst_p - 1:0]  m_dat_i // input to bus
   ,input m_we_i, m_stb_i, m_cyc_i
   ,input [(_data_bits_p * burst_p) / 8 - 1:0]  m_sel_i
   ,output logic m_ack
   ,output logic [_data_bits_p * burst_p - 1:0]  m_dat_o // output to bus

   // for SDRAM
   ,output logic [_data_bits_p - 1:0] s_dq_o
   ,output logic [_addr_bits_p - 1:0] s_addr_o
   ,output logic [1:0] s_ba_o
   ,output logic s_cke_o
   ,output logic s_cs_no
   ,output logic s_ras_no
   ,output logic s_cas_no
   ,output logic s_we_no
   ,output logic s_dqm_o
   ,input  logic s_dq_i
);


   /** raw dram parameters from datasheet with unit conversions and maybe a bit of rounding */
   localparam tRP_us_lp = 0.015;
   localparam tRFC_us_lp = 0.066;
   // One distributed refresh every 7.8125us, which we round down to 7
   localparam tREF_dist_us_lp = 7;
   localparam init_wait_us_lp = 100;
   // minimum clock period is 7ns, and 1/7.0ns ~= 142MHz
   localparam min_period_mhz_lp = 142;
   localparam tMRD_wait_cycles_lp = 2;

   /* parameterized parameters */
   // For 7E, the minimum clock period for CL=2 is 7.5ns, or 133MHz.
   // The minimum clock period for CL=3 is 7.0ns
   localparam CL_lp = (sys_clk_mhz_p <= 133) ? 3'd2 : 3'd3;
   // cycles needed to wait for 100us
   localparam init_wait_cycles_lp = sys_clk_mhz_p * init_wait_us_lp;
   localparam tRP_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tRP_us_lp));
   localparam tRFC_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tRFC_us_lp));
   localparam tREF_dist_wait_cycles = sys_clk_mhz_p * tREF_dist_us_lp;

   typedef enum logic [4:0] {
       INIT_WAIT,
       INIT_NOP,
       INIT_PRECHARGE,
       INIT_PRECHARGE_WAIT,
       INIT_REFRESH_1,
       INIT_REFRESH_1_WAIT,
       INIT_REFRESH_2,
       INIT_REFRESH_2_WAIT,
       INIT_LOAD_MODE,
       INIT_LOAD_MODE_WAIT,
       AUTO_REFRESH,
       ACTIVE,
       READ,
       WRITE,
       PRECHARGE,
       IDLE
   } state_t;

   task automatic set_cmd_NOP ();
      s_cs_no = 1'b0;
      s_ras_no = 1'b1;
      s_cas_no = 1'b1;
      s_we_no  = 1'b1;
   endtask

   task automatic set_cmd_PRECHARGE_ALL ();
      s_cs_no = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b1;
      s_we_no  = 1'b0;
      // set A10 high for precharge all
      s_addr_o = 13'b0_0100_0000_0000;
   endtask

   task automatic set_cmd_AUTO_REFRESH();
      s_cs_no  = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b0;
      s_we_no  = 1'b1;
   endtask

   task automatic set_cmd_LOAD_MODE_REGISTER();
      s_cs_no = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b0;
      s_we_no = 1'b0;
   endtask

   int wait_counter_d, wait_counter_q;
   int auto_refresh_counter_d, auto_refresh_counter_q;
   logic dram_initialized;
   state_t state_d, state_q;

   initial begin
      assert(sys_clk_mhz_p <= min_period_mhz_lp) else
          $error("sys_clk_mhz_p exceeds minimum clock period of 7ns!");

      assert(burst_p == 1 || burst_p == 2 || burst_p == 4 || burst_p == 8) else
        $error("Invalid burst length given. Valid burst lengths are 1, 2, 4, and 8");
   end

   /** wait counter */
   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         // on reset, we go through the initialization process
         wait_counter_q <= init_wait_us_lp;
      end else begin
         wait_counter_q <= wait_counter_d;
      end
   end

   /** auto refresh counter after dram is initialized */
   always_comb begin
      if (!dram_initialized || auto_refresh_counter_q == 0) begin
         auto_refresh_counter_d = tREF_dist_us_lp;
      end else begin
         auto_refresh_counter_d = auto_refresh_counter_q - 1'b1;
      end
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         auto_refresh_counter_q <= '0;
      end else begin
         auto_refresh_counter_q <= auto_refresh_counter_d;
      end
   end

   /** state machine */
   always_comb begin
      s_cke_o = 1;
      wait_counter_d = '0;
      s_addr_o = '0;

      case (state_q)
      /** The init process
       1. Apply power, assert CKE LOW, provide stable clock
       2. Wait 100μs minimum
       3. Bring CKE HIGH, issue a NOP command
       4. Issue PRECHARGE ALL (A10 HIGH)
       5. Wait tRP
       6. Issue two AUTO REFRESH commands, waiting tRFC between each
       7. Issue LOAD MODE REGISTER command with your desired configuration (burst length, CAS latency, burst type)
       8. Wait tMRD before any further commands
      **/
      INIT_WAIT: begin
         dram_initialized = 1'b0;
         s_cke_o = 0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            state_d = state_q;
            wait_counter_d = wait_counter_q - 1'b1;
         end else begin
            state_d = INIT_NOP;
         end
      end
      INIT_NOP: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         state_d = INIT_PRECHARGE;
      end
      INIT_PRECHARGE: begin
         dram_initialized = 1'b0;
         set_cmd_PRECHARGE_ALL();

         state_d = INIT_PRECHARGE_WAIT;
         wait_counter_d = tRP_wait_cycles_lp - 1'b1;
      end
      INIT_PRECHARGE_WAIT: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            wait_counter_d = wait_counter_q - 1'b1;
            state_d = state_q;
         end else begin
            state_d = INIT_REFRESH_1;
         end
      end
      INIT_REFRESH_1: begin
         dram_initialized = 1'b0;
         set_cmd_AUTO_REFRESH();

         state_d = INIT_REFRESH_1_WAIT;
         wait_counter_d = tRFC_wait_cycles_lp - 1'b1;
      end
      INIT_REFRESH_1_WAIT: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            wait_counter_d = wait_counter_q - 1'b1;
            state_d = state_q;
         end else begin
            state_d = INIT_REFRESH_2;
         end
      end
      INIT_REFRESH_2: begin
         dram_initialized = 1'b0;
         set_cmd_AUTO_REFRESH();

         state_d = INIT_REFRESH_2_WAIT;
         wait_counter_d = tRFC_wait_cycles_lp- 1'b1;
      end
      INIT_REFRESH_2_WAIT: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            wait_counter_d = wait_counter_q - 1'b1;
            state_d = state_q;
         end else begin
            state_d = INIT_LOAD_MODE;
         end
      end
      INIT_LOAD_MODE: begin
         dram_initialized = 1'b0;
         set_cmd_LOAD_MODE_REGISTER();
         /**
          A[12:10] = Reserved
          A[9]     = Write Burst Mode { 0 = Programmed Burst Length }
          A[8:7]   = Op Mode { 0 = Standard Operation }
          A[6:4]   = CAS Latency
          A[3]     = Burst Type { 0 = Sequential }
          A[2:0]   = Burst Length
          **/
         s_addr_o = {1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, CL_lp[2:0], 1'b0, burst_p[2:0]};

         state_d = INIT_LOAD_MODE_WAIT;
         wait_counter_d = tMRD_wait_cycles_lp - 1'b1;
      end
      INIT_LOAD_MODE_WAIT: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            wait_counter_d = wait_counter_q - 1'b1;
            state_d = state_q;
         end else begin
            state_d = IDLE;
         end
      end
      IDLE: begin
         dram_initialized = 1'b1;
         set_cmd_NOP();
         if (auto_refresh_counter_q == 0) begin
            state_d = AUTO_REFRESH;
         end
         state_d = state_q;
      end
      default: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();
         state_d = state_q;
         $fatal("At an invalid state!");
      end
      endcase
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         state_q <= INIT_WAIT;
      end else begin
         state_q <= state_d;
      end
   end

endmodule
