# Colours the main blocks in the Device view of the implemented design
# (Open Implemented Design, then run this script from the Tcl console).
# Encoder
highlight_objects -color green \
    [get_cells -hierarchical -filter {NAME =~ "*s4d_encoder_ecg_axis_0*"}]
# S4D core
highlight_objects -color red \
    [get_cells -hierarchical -filter {NAME =~ "*s4d_core_axis_0*"}]
# Decoder
highlight_objects -color blue \
    [get_cells -hierarchical -filter {NAME =~ "*s4d_decoder_axis_0*"}]
# AXI DMA + PS7
highlight_objects -color orange \
    [get_cells -hierarchical -filter {NAME =~ "*axi_dma_0*"}]
highlight_objects -color orange \
    [get_cells -hierarchical -filter {NAME =~ "*processing_system7*"}]
