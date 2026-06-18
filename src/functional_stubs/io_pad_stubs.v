// Functional stubs for gf180mcu_fd_io pad cells.
// Strips all #delay and specify blocks; only the signal-level behaviour is kept.
// Use in place of gf180mcu_fd_io.v for GLS where pad propagation delay is
// covered by SDF annotation and primitive delays are unwanted noise.

// Bidirectional 24 mA pad
module gf180mcu_fd_io__bi_24t (CS, SL, IE, OE, PU, PD, A, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  CS, SL, IE, OE, PU, PD, A;
    inout  PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;

    assign Y   = PAD & IE;
    assign PAD = OE ? A : 1'bz;
endmodule

// Input pad — CMOS threshold
module gf180mcu_fd_io__in_c (PU, PD, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  PU, PD, PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;

    assign Y = PAD;
endmodule

// Input pad — Schmitt trigger (functionally identical to in_c)
module gf180mcu_fd_io__in_s (PU, PD, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  PU, PD, PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;

    assign Y = PAD;
endmodule

// Analog signal pad — no digital logic
module gf180mcu_fd_io__asig_5p0 (ASIG5V, DVDD, DVSS, VDD, VSS);
    inout ASIG5V, DVDD, DVSS, VDD, VSS;
endmodule

// Corner pad — no logic
module gf180mcu_fd_io__cor (DVDD, DVSS, VDD, VSS);
    inout DVDD, DVSS, VDD, VSS;
endmodule

// Power / ground pads
module gf180mcu_fd_io__dvdd (DVDD, DVSS, VSS);
    inout  DVDD, DVSS, VSS;
    supply1 DVDD;
endmodule

module gf180mcu_fd_io__dvss (DVDD, DVSS, VDD);
    inout  DVDD, DVSS, VDD;
    supply0 DVSS;
endmodule

// Fill cells — no logic
module gf180mcu_fd_io__fill1 (DVDD, DVSS, VDD, VSS);
    inout DVDD, DVSS, VDD, VSS;
endmodule

module gf180mcu_fd_io__fill5 (DVDD, DVSS, VDD, VSS);
    inout DVDD, DVSS, VDD, VSS;
endmodule

module gf180mcu_fd_io__fill10 (DVDD, DVSS, VDD, VSS);
    inout DVDD, DVSS, VDD, VSS;
endmodule
