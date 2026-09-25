`timescale 1ns / 1ps
`default_nettype none

module s4d_wload_axis #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer LOG2_NB  = 9,
    parameter integer NMAC     = 64,
    parameter integer LOG2_NPH = 2,
    parameter integer W_A      = 24,
    parameter integer W_B      = 16,
    parameter integer W_D      = 16,
    parameter integer W_BSH    = 5,
    parameter integer BN_W_W   = 16,
    parameter integer BN_W_B   = 32,
    parameter integer W_MACC   = 32,
    parameter integer W_W      = 12,
    parameter integer AXI_DW   = 64
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start,
    output reg                   busy,
    output reg                   done,

    // stream dei pesi: NBEAT beat consecutivi per layer, in ordine.
    // tlast e' accettato ma IGNORATO: il DMA puo' inviare i 6 layer in un
    // unico trasferimento; il conteggio interno fa da confine fra layer.
    input  wire                  s_axis_w_tvalid,
    output wire                  s_axis_w_tready,
    input  wire [AXI_DW-1:0]     s_axis_w_tdata,
    input  wire                  s_axis_w_tlast,

    output reg                   cfg_we,
    output reg  [LOG2_NU-1:0]    cfg_unit,
    output reg  [LOG2_NB-1:0]    cfg_addr,
    output reg  [2*W_A+2*W_B-1:0] cfg_data,
    output reg                   cfgd_we,
    output reg  [LOG2_H-1:0]     cfgd_ch,
    output reg  signed [W_D-1:0] cfgd_data,
    output reg  [W_BSH-1:0]      cfgd_bsh,

    output reg                   ld_bn_we_sh,
    output reg  [5:0]            ld_bn_sh,
    output reg                   ld_bn_we_w,
    output reg                   ld_bn_we_b,
    output reg  [LOG2_H-1:0]     ld_bn_addr,
    output reg  signed [BN_W_W-1:0] ld_bn_w,
    output reg  signed [BN_W_B-1:0] ld_bn_b,

    output reg                   ld_wo_we,
    output reg  [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,
    output reg  [3:0]            ld_wo_sel,
    output reg  [AXI_DW-1:0]     ld_wo_data,
    output reg                   ld_bo_we,
    output reg  [LOG2_H:0]       ld_bo_addr,
    output reg  signed [W_MACC-1:0] ld_bo_data
);

    localparam integer CW     = 2*W_A + 2*W_B;
    localparam integer NCOEF  = NUNITS * (1 << LOG2_NB);
    localparam integer NWO    = (1 << LOG2_NPH) * H;
    localparam integer WOBEAT = (NMAC*W_W) / AXI_DW;
    localparam integer TH     = 2*H;

    localparam integer H_END  = 1;
    localparam integer C_BEG  = H_END;
    localparam integer C_END  = C_BEG + 2*NCOEF;
    localparam integer D_END  = C_END + H;
    localparam integer NW_END = D_END + H;
    localparam integer NB_END = NW_END + H;
    localparam integer BO_END = NB_END + TH;
    localparam integer WO_END = BO_END + NWO*WOBEAT;
    localparam integer NBEAT  = WO_END;

    localparam integer PW = 15;
    assign s_axis_w_tready = busy;
    wire unused_tlast      = s_axis_w_tlast;

    reg [PW-1:0]      pos;

    reg [63:0] coef_lo;

    wire       r_hs = s_axis_w_tvalid & s_axis_w_tready;

    wire [PW-1:0] p_c  = pos - C_BEG[PW-1:0];
    wire [PW-1:0] p_d  = pos - C_END[PW-1:0];
    wire [PW-1:0] p_nw = pos - D_END[PW-1:0];
    wire [PW-1:0] p_nb = pos - NW_END[PW-1:0];
    wire [PW-1:0] p_bo = pos - NB_END[PW-1:0];
    wire [PW-1:0] p_wo = pos - BO_END[PW-1:0];

    reg [LOG2_NPH+LOG2_H-1:0] wo_row;
    reg [3:0]                 wo_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy          <= 1'b0;
            done          <= 1'b0;
            pos           <= {PW{1'b0}};
            wo_row        <= {(LOG2_NPH+LOG2_H){1'b0}};
            wo_sel        <= 4'd0;
            coef_lo       <= 64'd0;
            cfg_we        <= 1'b0;
            cfgd_we       <= 1'b0;
            ld_bn_we_sh   <= 1'b0;
            ld_bn_sh      <= 6'd0;
            ld_bn_we_w    <= 1'b0;
            ld_bn_we_b    <= 1'b0;
            ld_wo_we      <= 1'b0;
            ld_bo_we      <= 1'b0;
        end else begin
            done       <= 1'b0;
            cfg_we     <= 1'b0;
            cfgd_we    <= 1'b0;
            ld_bn_we_sh <= 1'b0;
            ld_bn_we_w <= 1'b0;
            ld_bn_we_b <= 1'b0;
            ld_wo_we   <= 1'b0;
            ld_bo_we   <= 1'b0;

            if (!busy) begin
                if (start) begin
                    busy    <= 1'b1;
                    pos     <= {PW{1'b0}};
                    wo_row  <= {(LOG2_NPH+LOG2_H){1'b0}};
                    wo_sel  <= 4'd0;
                end
            end else begin

                if (r_hs) begin
                    pos <= pos + 1'b1;

                    if (pos < H_END[PW-1:0]) begin
                        ld_bn_sh    <= s_axis_w_tdata[5:0];
                        ld_bn_we_sh <= 1'b1;
                    end else if (pos < C_END[PW-1:0]) begin
                        if (!p_c[0]) begin
                            coef_lo <= s_axis_w_tdata;
                        end else begin
                            cfg_data <= {s_axis_w_tdata[CW-64-1:0], coef_lo};
                            cfg_unit <= p_c[LOG2_NB+LOG2_NU:LOG2_NB+1];
                            cfg_addr <= p_c[LOG2_NB:1];
                            cfg_we   <= 1'b1;
                        end
                    end else if (pos < D_END[PW-1:0]) begin
                        cfgd_ch   <= p_d[LOG2_H-1:0];
                        cfgd_data <= $signed(s_axis_w_tdata[W_D-1:0]);
                        cfgd_bsh  <= s_axis_w_tdata[W_D+W_BSH-1:W_D];
                        cfgd_we   <= 1'b1;
                    end else if (pos < NW_END[PW-1:0]) begin
                        ld_bn_addr <= p_nw[LOG2_H-1:0];
                        ld_bn_w    <= $signed(s_axis_w_tdata[BN_W_W-1:0]);
                        ld_bn_we_w <= 1'b1;
                    end else if (pos < NB_END[PW-1:0]) begin
                        ld_bn_addr <= p_nb[LOG2_H-1:0];
                        ld_bn_b    <= $signed(s_axis_w_tdata[BN_W_B-1:0]);
                        ld_bn_we_b <= 1'b1;
                    end else if (pos < BO_END[PW-1:0]) begin
                        ld_bo_addr <= p_bo[LOG2_H:0];
                        ld_bo_data <= $signed(s_axis_w_tdata[W_MACC-1:0]);
                        ld_bo_we   <= 1'b1;
                    end else begin
                        ld_wo_addr <= wo_row;
                        ld_wo_sel  <= wo_sel;
                        ld_wo_data <= s_axis_w_tdata;
                        ld_wo_we   <= 1'b1;
                        if (wo_sel == WOBEAT[3:0] - 4'd1) begin
                            wo_sel <= 4'd0;
                            wo_row <= wo_row + 1'b1;
                        end else begin
                            wo_sel <= wo_sel + 4'd1;
                        end
                    end

                    if (pos == NBEAT[PW-1:0] - 1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
