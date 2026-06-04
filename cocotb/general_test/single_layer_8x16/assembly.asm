dram2sram weights,      sram=0x00, dram=0x100, n=16
dram2sram activations,  sram=0x20, dram=0x300, n=12
dram2sram activations,  sram=0x30, dram=0x400, n=16
pipeline_setup relu=enable, k_tile=16
load_weights sram=0x00

load_bias   sram=0x20
load_zp     sram=0x24 # comment
load_scale  sram=0x28

matmul act=0x30, res=0x50

sram2dram sram=0x50, dram=0x500, n=8

exit