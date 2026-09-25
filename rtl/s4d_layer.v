`timescale 1ns / 1ps
`default_nettype none

module s4d_layer #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,
    parameter integer LOG2_NCH = 4,
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,
    parameter integer NMAC     = 64,
    parameter integer LOG2_NPH = 2,

    parameter integer W_ACT   = 16,
    parameter integer F_ACT   = 11,
    parameter integer W_U     = 16,
    parameter integer F_U     = 11,
    parameter integer W_A     = 24,
    parameter integer F_A     = 22,
    parameter integer W_B     = 16,
    parameter integer F_B     = 15,
    parameter integer W_S     = 32,
    parameter integer F_Y     = 19,
    parameter integer W_D     = 16,
    parameter integer F_D     = 14,
    parameter integer W_BSH   = 5,
    parameter integer W_BACC  = 40,
    parameter integer W_SSM   = 16,
    parameter integer F_SSM   = 11,
    parameter integer W_W     = 12,
    parameter integer F_W     = 11,
    parameter integer W_MACC  = 32,
    parameter integer F_SIGI  = 11,
    parameter integer F_SIGO  = 14,
    parameter integer FIFO_LOG2 = 10,

    parameter integer LOADABLE = 0,
    parameter integer LD_W    = 64,
    parameter integer BN_SH       = 11,
    parameter integer BN_W_W      = 16,
    parameter integer BN_W_B      = 32,
    parameter integer BN_W_ACC    = 34,

    // s4d_layer e' UN layer fisico riusato per NLAYER ruoli (time-multiplexato):
    // non esiste "il" file giusto qui, chi lo istanzia sovrascrive sempre. I
    // default puntano al layer 0 solo perche' e' un file che esiste davvero,
    // non un placeholder inventato.
    parameter BN_W_FILE  = "l0_bn_w.mem",
    parameter BN_B_FILE  = "l0_bn_b.mem",
    parameter GELU_FILE  = "gelu_lut.mem",
    parameter SIG_FILE   = "sigmoid_lut.mem",
    parameter WOUT_FILE  = "l0_s4d_wout.mem",
    parameter BOUT_FILE  = "l0_s4d_bout.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,
    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    input  wire                    ld_bn_we_sh,
    input  wire [5:0]              ld_bn_sh,
    input  wire                    ld_bn_we_w,
    input  wire                    ld_bn_we_b,
    input  wire [LOG2_H-1:0]       ld_bn_addr,
    input  wire signed [BN_W_W-1:0] ld_bn_w,
    input  wire signed [BN_W_B-1:0] ld_bn_b,

    input  wire                    ld_wo_we,
    input  wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,
    input  wire [3:0]              ld_wo_sel,
    input  wire [LD_W-1:0]         ld_wo_data,
    input  wire                    ld_bo_we,
    input  wire [LOG2_H:0]         ld_bo_addr,
    input  wire signed [W_MACC-1:0] ld_bo_data,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_ACT-1:0] in_data,
    input  wire                    in_last_seq,

    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_ACT-1:0] out_data,
    output reg                     out_last_ch,
    output reg                     out_last_seq,

    output wire                    ovf,
    output reg                     err_bank_ovr
);

    // ------------------------------------------------------------------
    // BACKPRESSURE
    // oen congela lo stadio di uscita E il pop della FIFO del residual.
    // I due DEVONO muoversi insieme: se la FIFO avanzasse mentre
    // l'uscita e' ferma, il residual si disallineerebbe progressivamente
    // e otterresti un errore che sembra aritmetico. A full rate il bug
    // sarebbe invisibile.
    // ------------------------------------------------------------------
    wire oen = out_ready;

    localparam integer FDEPTH = (1 << FIFO_LOG2);

    reg signed [W_ACT-1:0] fifo    [0:FDEPTH-1];
    reg                    fifo_ls [0:FDEPTH-1];
    reg [FIFO_LOG2-1:0]    fwr, frd;
    reg [FIFO_LOG2:0]      fcnt;

    wire fifo_ready = (fcnt < FDEPTH - H);

    wire                   ln_valid;
    wire signed [W_U-1:0]  ln_data;
    wire                   ln_last_ch, ln_last_seq;
    wire                   ln_ready;
    wire                   bk_ready;

    s4d_affine #(
        .H(H), .LOG2_H(LOG2_H),
        .W_IN(W_ACT), .W_OUT(W_U),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .LOADABLE(LOADABLE),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_bn_we_w), .ld_we_b(ld_bn_we_b),
        .ld_addr(ld_bn_addr), .ld_w(ld_bn_w), .ld_b(ld_bn_b),
        .ld_we_sh(ld_bn_we_sh), .ld_sh(ld_bn_sh),
        .in_valid(in_valid && fifo_ready), .in_ready(ln_ready),
        .in_data(in_data), .in_last_seq(in_last_seq),
        .out_ready(bk_ready),
        .out_valid(ln_valid), .out_data(ln_data),
        .out_last_ch(ln_last_ch), .out_last_seq(ln_last_seq)
    );

    wire                     bk_valid;
    wire signed [W_SSM-1:0]  bk_data;
    wire [LOG2_H-1:0]        bk_ch;
    wire                     bk_last;

    wire ge_in_ready;
    wire mix_ready;

    s4d_biquad_bank #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD),
        .NCH(NCH), .LOG2_NCH(LOG2_NCH), .LOG2_NB(LOG2_NB),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
        .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
        .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_BACC),
        .W_OUT(W_SSM), .F_OUT(F_SSM)
    ) u_bank (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data), .cfgd_bsh(cfgd_bsh),
        .in_valid(ln_valid), .in_ready(bk_ready), .in_data(ln_data),
        .in_last_seq(ln_last_seq),
        .out_valid(bk_valid), .out_ready(ge_in_ready), .out_data(bk_data),
        .out_ch(bk_ch), .out_last(bk_last),
        .ovf(ovf)
    );

    wire                    ge_valid;
    wire signed [W_SSM-1:0] ge_data;
    reg  [2:0]              ge_last_d;

    // Con s4d_pwl dotato di in_ready/out_ready la catena
    // bank -> gelu -> mixing e' continua: questo caso non puo' piu'
    // verificarsi. Il flag resta come sentinella: se scatta, significa
    // che qualcuno ha rotto la propagazione del ready.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                       err_bank_ovr <= 1'b0;
        else if (ge_valid && !mix_ready)  err_bank_ovr <= 1'b1;
    end

    s4d_pwl #(
        .W_IN(W_SSM), .F_IN(F_SSM), .W_OUT(W_SSM), .F_OUT(F_SSM),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(1), .Y_LO_Q(0), .Y_HI_Q(0),
        .LUT_FILE(GELU_FILE)
    ) u_gelu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bk_valid), .in_ready(ge_in_ready), .in_data(bk_data),
        .out_ready(mix_ready),
        .out_valid(ge_valid), .out_data(ge_data)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)         ge_last_d <= 3'b0;
        else if (mix_ready) ge_last_d <= {ge_last_d[1:0], bk_last};
    end

    wire                    gl_valid;
    wire signed [W_ACT-1:0] gl_data;
    wire                    gl_last;

    s4d_mixing_glu #(
        .H(H), .LOG2_H(LOG2_H),
        .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_IN(W_SSM), .F_IN(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_ACC(W_MACC),
        .W_SIG(16), .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .W_A(16), .F_A(F_ACT),
        .W_OUT(W_ACT), .F_OUT(F_ACT),
        .LOADABLE(LOADABLE), .LD_W(LD_W),
        .W_FILE(WOUT_FILE), .B_FILE(BOUT_FILE), .SIG_FILE(SIG_FILE)
    ) u_mix (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_wo_we), .ld_addr_w(ld_wo_addr), .ld_sel_w(ld_wo_sel),
        .ld_data_w(ld_wo_data),
        .ld_we_b(ld_bo_we), .ld_addr_b(ld_bo_addr), .ld_data_b(ld_bo_data),
        .in_valid(ge_valid), .in_ready(mix_ready),
        .in_data(ge_data), .in_last(ge_last_d[2]),
        .out_ready(oen),
        .out_valid(gl_valid), .out_data(gl_data), .out_last(gl_last)
    );

    assign in_ready = ln_ready && fifo_ready;

    wire fifo_push = in_valid && in_ready;
    wire fifo_pop  = gl_valid && oen;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fwr  <= {FIFO_LOG2{1'b0}};
            frd  <= {FIFO_LOG2{1'b0}};
            fcnt <= {(FIFO_LOG2+1){1'b0}};
        end else begin
            if (fifo_push) fwr <= fwr + 1'b1;
            if (fifo_pop)  frd <= frd + 1'b1;
            fcnt <= fcnt + (fifo_push ? 1'b1 : 1'b0) - (fifo_pop ? 1'b1 : 1'b0);
        end
    end

    always @(posedge clk) if (rst_n && fifo_push) fifo[fwr] <= in_data;

    always @(posedge clk) if (fifo_push) fifo_ls[fwr] <= in_last_seq;

    localparam integer RW = W_ACT + 1;
    localparam signed [RW-1:0] RMAX = {{(RW-W_ACT+1){1'b0}}, {(W_ACT-1){1'b1}}};
    localparam signed [RW-1:0] RMIN = {{(RW-W_ACT+1){1'b1}}, {(W_ACT-1){1'b0}}};

    wire signed [RW-1:0] res_sum = {fifo[frd][W_ACT-1], fifo[frd]}
                                 + {gl_data[W_ACT-1],   gl_data};

    reg [LOG2_H-1:0] och;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_ACT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
            och          <= {LOG2_H{1'b0}};
        end else if (oen) begin
            out_valid    <= gl_valid;
            out_last_ch  <= gl_valid && (och == H-1);
            out_last_seq <= gl_valid && (och == H-1) && fifo_ls[frd];

            if (gl_valid) och <= (och == H-1) ? {LOG2_H{1'b0}} : (och + 1'b1);

            if      (res_sum > RMAX) out_data <= RMAX[W_ACT-1:0];
            else if (res_sum < RMIN) out_data <= RMIN[W_ACT-1:0];
            else                     out_data <= res_sum[W_ACT-1:0];
        end
    end

endmodule

`default_nettype wire
