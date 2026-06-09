# SRAM macros

define_pdn_grid \
    -macro \
    -instances {
        i_chip_core.i_tpu_soc.control.control_sram_inst.control_sram_block.sram_inst
        i_chip_core.i_tpu_soc.control.control_decoder_inst.compute_core_inst.weight_sram_inst.sram_inst.sram_gen_blk_8x256
    } \
    -name sram_macros_NS \
    -starts_with POWER \
    -halo "$::env(PDN_HORIZONTAL_HALO) $::env(PDN_VERTICAL_HALO)"

add_pdn_connect \
    -grid sram_macros_NS \
    -layers "$::env(PDN_VERTICAL_LAYER) $::env(PDN_HORIZONTAL_LAYER)"

add_pdn_connect \
    -grid sram_macros_NS \
    -layers "$::env(PDN_VERTICAL_LAYER) Metal3"

# Add stripes on W/E edges of SRAM
add_pdn_stripe \
    -grid sram_macros_NS \
    -layer Metal4 \
    -width 1.36 \
    -offset 0.68 \
    -spacing 0.28 \
    -pitch 298.30 \
    -starts_with GROUND \
    -number_of_straps 2

# Since the above stripes block the top level PDN at Metal4, add some more stripes
# to improve the PDN's integrity and ensure a better connection for the macro.
add_pdn_stripe \
    -grid sram_macros_NS \
    -layer Metal4 \
    -width 4.00 \
    -offset 50.80 \
    -spacing 0.28 \
    -pitch 48.86 \
    -starts_with GROUND \
    -number_of_straps 5

#define_pdn_grid \
#    -macro \
#    -instances i_chip_core.sram_1 \
#    -name sram_macros_WE \
#    -starts_with POWER \
#    -halo "$::env(PDN_HORIZONTAL_HALO) $::env(PDN_VERTICAL_HALO)"

#add_pdn_connect \
#    -grid sram_macros_WE \
#    -layers "$::env(PDN_VERTICAL_LAYER) $::env(PDN_HORIZONTAL_LAYER)"
#
#add_pdn_connect \
#    -grid sram_macros_WE \
#    -layers "$::env(PDN_VERTICAL_LAYER) Metal3"
#
## Add stripes on W/E edges of SRAM
#add_pdn_stripe \
#    -grid sram_macros_WE \
#    -layer Metal4 \
#    -width 1.36 \
#    -offset 0.68 \
#    -spacing 0.28 \
#    -pitch 319.09 \
#    -starts_with POWER \
#    -number_of_straps 2
#
## Since the above stripes block the top level PDN at Metal4, add some more stripes
## to improve the PDN's integrity and ensure a better connection for the macro.
#add_pdn_stripe \
#    -grid sram_macros_WE \
#    -layer Metal4 \
#    -width 4.00 \
#    -offset 28.0 \
#    -spacing 0.28 \
#    -pitch 43.50 \
#    -starts_with GROUND \
#    -number_of_straps 7
