`timescale 1ns / 1ps
`default_nettype none

module s4d_decoder #(
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer LOG2_L   = 10,      // sCIFAR (era 12/Speech Commands)
    parameter integer K        = 10,      // sCIFAR: 10 classi (era 35/Speech Commands)
    parameter integer LOG2_K   = 4,       // ceil(log2(10)) (era 6/Speech Commands)

    parameter integer W_ACT    = 16,
    parameter integer W_CIN    = 8,
    parameter integer W_CLS    = 8,
    parameter integer W_ACC    = 32,

    parameter integer F_OUT    = 11,
    parameter integer RQ_SHIFT = 6,

    parameter integer BN_SH       = 12,
    parameter integer BN_W_W      = 16,
    parameter integer BN_W_B      = 32,
    parameter integer BN_W_ACC    = 34,

    parameter BN_W_FILE  = "l6_bn_w.mem",
    parameter BN_B_FILE  = "l6_bn_b.mem",
    parameter W_FILE     = "cls_w.mem",
    parameter B_FILE     = "cls_b.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_ACT-1:0] in_data,
    input  wire                    in_last_seq,

    output wire                    logit_valid,
    output wire [LOG2_K-1:0]       logit_idx,
    output wire signed [W_ACC-1:0] logit_data,
    output wire                    logit_last,

    output wire                    class_valid,
    output wire [LOG2_K-1:0]       class_idx
);

    wire                    ln_valid;
    wire signed [W_ACT-1:0] ln_data;
    wire                    ln_last_ch;
    wire                    ln_last_seq;

    s4d_affine #(
        .H(H), .LOG2_H(LOG2_H),
        .W_IN(W_ACT), .W_OUT(W_ACT),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk          (clk),
        .rst_n        (rst_n),
        .ld_we_w      (1'b0),
        .ld_we_b      (1'b0),
        .ld_addr      ({LOG2_H{1'b0}}),
        .ld_w         ({BN_W_W{1'b0}}),
        .ld_b         ({BN_W_B{1'b0}}),
        .ld_we_sh     (1'b0),
        .ld_sh        (6'd0),
        .in_valid     (in_valid),
        .in_ready     (in_ready),
        .in_data      (in_data),
        .in_last_seq  (in_last_seq),
        .out_ready    (1'b1),
        .out_valid    (ln_valid),
        .out_data     (ln_data),
        .out_last_ch  (ln_last_ch),
        .out_last_seq (ln_last_seq)
    );

    wire                    pool_valid;
    wire signed [W_ACT-1:0] pool_data;
    wire                    pool_last;

    s4d_meanpool #(
        .H(H), .LOG2_H(LOG2_H), .LOG2_L(LOG2_L),
        .W_IN(W_ACT), .W_ACC(W_ACC), .W_OUT(W_ACT)
    ) u_pool (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (ln_valid),
        .in_data     (ln_data),
        .in_last_ch  (ln_last_ch),
        .in_last_seq (ln_last_seq),
        .out_valid   (pool_valid),
        .out_data    (pool_data),
        .out_last    (pool_last),
        .busy        ()
    );

    s4d_classifier #(
        .H(H), .LOG2_H(LOG2_H), .K(K), .LOG2_K(LOG2_K),
        .W_IN(W_ACT), .W_CIN(W_CIN), .W_CLS(W_CLS), .W_ACC(W_ACC),
        .RQ_SHIFT(RQ_SHIFT), .W_FILE(W_FILE), .B_FILE(B_FILE)
    ) u_cls (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (pool_valid),
        .in_data     (pool_data),
        .in_last     (pool_last),
        .out_valid   (logit_valid),
        .out_idx     (logit_idx),
        .out_logit   (logit_data),
        .out_last    (logit_last),
        .class_valid (class_valid),
        .class_idx   (class_idx)
    );


endmodule

`default_nettype wire
